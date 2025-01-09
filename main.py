from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import boto3
import json
import os
import logging
import nest_asyncio

from dotenv import load_dotenv
from openai import OpenAI
from langchain.memory import ConversationBufferWindowMemory  # Corrected import

# Import your utility functions
from utils import (
    find_match, 
    query_refiner, 
    get_conversation_string, 
    generate_rag_response, 
    init_pinecone
)

# Apply nest_asyncio
nest_asyncio.apply()

# 1. Load environment variables from .env
load_dotenv()

# 2. Fetch environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # default if not set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID")
GEN_MODEL_ID = os.getenv("GEN_MODEL_ID")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "8"))

# 3. Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 4. Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# 5. Initialize Pinecone Index
pinecone_index = init_pinecone()  # Make sure init_pinecone uses PINECONE_API_KEY, PINECONE_INDEX_NAME

# 6. Initialize the Bedrock client
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# 7. Set up environment variables for AWS keys (optional, if needed by other AWS services)
os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_KEY
os.environ['AWS_DEFAULT_REGION'] = AWS_REGION

# In-memory storage for chat sessions
chat_sessions: Dict[str, Dict[str, list]] = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

class ChatPayload(BaseModel):
    session_id: str  # Unique session ID for the chat
    message: str     # User input

def initialize_session(session_id):
    """Initialize a new chat session if it doesn't exist."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            'requests': [],
            'responses': ["How can I assist you?"],
            'buffer_memory': ConversationBufferWindowMemory(k=5, return_messages=True)
        }

@app.get("/", status_code=status.HTTP_200_OK)
async def health():
    """Health check endpoint."""
    return "ChainGuardianAssistant Check Successful"

@app.post("/agent_chat")
async def agent_chat(chat_payload: ChatPayload):
    """Main endpoint to handle user interactions."""
    session_id = chat_payload.session_id
    user_message = chat_payload.message

    # 1. Initialize session
    initialize_session(session_id)

    try:
        # 2. Gather conversation log
        conversation_string = get_conversation_string(session_id, chat_sessions)

        # 3. Refine the user query using the conversation log
        refined_query = query_refiner(conversation_string, user_message, bedrock_client)

        # 4. Retrieve relevant context from Pinecone
        context = find_match(refined_query, pinecone_index, bedrock_client)

        # 5. Generate the final response using RAG
        try:
            final_response = await generate_rag_response(refined_query, context, client)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            final_response = "Unexpected Error Occurred with Agent"
            refined_query = ""

        # 6. Update session history
        chat_sessions[session_id]['requests'].append(user_message)
        chat_sessions[session_id]['responses'].append(final_response)

        # 7. Return the final response
        return JSONResponse(content={
            "response": final_response
        }, status_code=200)

    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))