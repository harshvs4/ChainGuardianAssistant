from dotenv import load_dotenv
import os
import json
import logging
from openai import OpenAI
from pinecone import Pinecone
import boto3
import nest_asyncio

nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Access environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')
EMBED_MODEL_ID = os.getenv('EMBED_MODEL_ID')
GEN_MODEL_ID = os.getenv('GEN_MODEL_ID')
MODEL_NAME = os.getenv('MODEL_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS'))

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
def init_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logger.info(f"Index name: {PINECONE_INDEX_NAME}")
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        return None

pinecone_index = init_pinecone()

# Initialize Bedrock client
def init_bedrock_client():
    try:
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        return bedrock_client
    except Exception as e:
        logger.error(f"Error initializing Bedrock client: {e}")
        return None

bedrock_client = init_bedrock_client()

async def get_completion(client, prompt):
    """
    Generate a response from the language model using the provided client and prompt.
    Handles streaming output for real-time response processing.
    """
    try:
        # Define the system and user messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Call the OpenAI or Anthropic client to generate the response
        response = client.chat.completions.create(
            model="gpt-4",  # Replace with the correct model ID for your use case
            messages=messages,
            stream=True  # Enable streaming
        )

        # Initialize response content
        response_content = ""
        print("Generating response...")
        print(response)
        # Process the streamed response
        for chunk in response:
            try:
                choices = chunk.choices
                #print("Debug Choices:", choices)
                if choices:
                    content = choices[0].delta.content
                    response_content += content
                    print(content, end="", flush=True)
            except Exception as e:
                logger.error(f"Error processing stream chunk: {e}")
                continue

        return response_content

    except Exception as e:
        logger.error(f"An error occurred while generating completion: {e}")
        return f"An error occurred while generating completion: {e}"

# Utility method to refine user queries
def query_refiner(conversation, query, bedrock_client, logger=logger):
    prompt = f"""
    Refine the following user query strictly to improve clarity or remove ambiguity, without adding any additional context, assumptions, or new meaning. Do not introduce any new topics, keywords, or interpretations.

    CONVERSATION LOG: 
    {conversation}

    Original Query: {query}

    Refined Query (must retain original intent and meaning):
    """
    try:
        if logger:
            logger.info("Refining the query based on conversation context...")

        gen_request = {
            "inputText": prompt,
            "textGenerationConfig": {"maxTokenCount": 1024, "temperature": 0}
        }

        gen_response = bedrock_client.invoke_model(
            modelId=GEN_MODEL_ID,
            body=json.dumps(gen_request)
        )

        response_body = json.loads(gen_response["body"].read())
        refined_query = response_body.get('outputText', query)
        return refined_query

    except Exception as e:
        logger.error(f"Error refining query: {e}")
        return query

def find_match(input_text, pinecone_index, bedrock_client, logger=logger):
    """
    Find the most relevant matches for the input text using embeddings from Amazon Bedrock and Pinecone.
    Dynamically apply a metadata filter for 'news' or 'transactions' based on the user query.
    """
    try:
        if logger:
            logger.info("Generating embedding for input text...")

        # Step 1: Determine if the user likely wants transactions or news.
        # You can use more sophisticated regex or keywords if needed.
        lower_text = input_text.lower()
        if "transaction" in lower_text or "txid" in lower_text or "transactionid" in lower_text:
            dataset_filter = {"dataset": "transactions"}
        elif "news" in lower_text or "recent trends" in lower_text:
            dataset_filter = {"dataset": "news"}
        else:
            # If neither is explicitly mentioned, you can:
            # 1) Return everything (no filter), or
            # 2) Decide on a default filter. Below we do no filter.
            dataset_filter = None

        # Step 2: Generate embedding for the input query
        native_request = {"inputText": input_text}
        request = json.dumps(native_request)
        response = bedrock_client.invoke_model(modelId=EMBED_MODEL_ID, body=request)
        model_response = json.loads(response["body"].read())
        embedding = model_response["embedding"]

        # Step 3: Query Pinecone with or without a filter
        if logger:
            logger.info(f"Searching Pinecone with dataset filter: {dataset_filter}")

        # The 'filter' parameter accepts a dictionary that matches Pinecone metadata
        search_results = pinecone_index.query(
            vector=embedding,
            top_k=TOP_K_RESULTS,
            include_metadata=True,
            filter=dataset_filter  # <-- apply the filter here
        )

        # Step 4: Separate matches into "news" vs. "transactions"
        top_results = {
            "news": [],
            "transactions": []
        }

        for match in search_results["matches"]:
            ds_type = match["metadata"].get("dataset", "")
            if ds_type == "news":
                top_results["news"].append(match["metadata"])
            elif ds_type == "transactions":
                top_results["transactions"].append(match["metadata"])

        if logger:
            logger.info("Top results found:")

            #print the title of each news article
            for news in top_results.get("news", []):
                logger.info(news.get("title", ""))

        return top_results

    except Exception as e:
        logger.error(f"Error finding match: {e}")
        return {"news": [], "transactions": []}

async def generate_rag_response(refined_query, retrieved_data, client, logger=logger):
    """
    Generate a response using the refined query and data retrieved from Pinecone for the ChainGuardian AI use case.
    - If the user query involves news articles or 'recent trends,' summarize the relevant article(s).
    - If the user query involves transactions or a specific Transaction ID, summarize the relevant transaction(s).
    - Incorporate knowledge of how the Risk Category was derived from the LightGBM model snippet.
    Handle sensitive topics and blockchain-related queries professionally.
    """

    try:
        if logger:
            logger.info("Generating response using RAG for ChainGuardian AI...")

        # Separate news and transaction data
        news_data = retrieved_data.get("news", [])
        transaction_data = retrieved_data.get("transactions", [])

        # --- CASE 1: News Articles ---
        # This covers general news queries, "recent trends," or a specific news article request.
        if len(news_data) > 0:
            # Build a summary of all the news article metadata
            # Each metadata can contain keys like title, url, full_text, content, description, etc.
            articles_summary = []
            for item in news_data:
                title = item.get("title", "Untitled")
                url = item.get("url", "No URL")
                description = item.get("description", "No description available.")
                full_text = item.get("full_text", "No full text available.")
                content = item.get("content", "")

                # Create a short block for each article
                articles_summary.append(
                    f"**Title**: {title}\n"
                    f"**URL**: {url}\n"
                    f"**Description**: {description}\n\n"
                    f"**Content/Full Text (excerpt)**:\n{full_text[:500]}...\n"  # Truncate for brevity
                )

            # Join all articles into one consolidated list
            articles_str = "\n".join(articles_summary)

            final_prompt = f"""
                You are ChainGuardian AI, a smart assistant specializing in fraud detection, compliance, and blockchain-related data analysis.
                The user has asked about news articles or recent trends.

                Below are the relevant news articles retrieved from Pinecone:
                {articles_str}

                **Your Task**:
                - Summarize or explain these articles for the user.
                - Provide insights into any 'recent trends' the user might be interested in, specifically referencing the details provided.
                - Maintain a neutral, professional tone.

                User Query: {refined_query}

                **Response**:
                """

        # --- CASE 2: Transaction Data ---
        # The user asked about one or more transactions, or specifically a Transaction ID.
        elif len(transaction_data) > 0:
            # Summarize each transaction
            transactions_summary = []
            for tx in transaction_data:
                tx_id = tx.get("TransactionID", "Unknown")
                purpose = tx.get("Purpose", "N/A")
                risk = tx.get("Risk_Category", "Unknown")
                region = tx.get("Region", "N/A")
                timestamp = tx.get("Timestamp", "N/A")
                sender = tx.get("Sender", "N/A")
                receiver = tx.get("Receiver", "N/A")
                amount = tx.get("Amount", "N/A")
                currency = tx.get("Currency", "N/A")
                gasfee = tx.get("GasFee", "N/A")
                aml_kyc_verified = tx.get("AML_KYC_Verified", "N/A")
                geo_recv = tx.get("Geolocation_Receiver", "N/A")
                geo_send = tx.get("Geolocation_Sender", "N/A")

                # Build a short block for each transaction
                transactions_summary.append(
                    f"- **Transaction ID**: {tx_id}\n"
                    f"  **Purpose**: {purpose}\n"
                    f"  **Risk Category**: {risk}\n"
                    f"  **Region**: {region}\n"
                    f"  **Timestamp**: {timestamp}\n"
                    f"  **Sender**: {sender}\n"
                    f"  **Receiver**: {receiver}\n"
                    f"  **Amount**: {amount} {currency}\n"
                    f"  **Gas Fee**: {gasfee}\n"
                    f"  **AML/KYC Verified**: {aml_kyc_verified}\n"
                    f"  **Geolocation (Receiver)**: {geo_recv}\n"
                    f"  **Geolocation (Sender)**: {geo_send}\n"
                )

            tx_str = "\n".join(transactions_summary)

            # Explanation of how the risk category was calculated (using LGB model)
            # from the snippet you provided
            risk_explanation = """
                The Risk Category is determined by our LightGBM model trained on various transaction features, 
                including Amount, GasFee, Currency, Purpose, Region, AML/KYC Verification, and Geolocations. 
                During inference, each new transaction is label-encoded for categorical variables (e.g., currency, purpose, region) 
                and numeric features (Amount, GasFee) are scaled or checked for validity. 
                The model then outputs a risk label: Low, Medium, or High. 
                This helps identify potential fraud or suspicious activity based on patterns learned from historical data.
                """

            final_prompt = f"""
                You are ChainGuardian AI, a smart assistant specializing in blockchain compliance and fraud detection.
                The user has asked about a specific blockchain transaction or transactions.

                Below are the relevant transactions retrieved from Pinecone:
                {tx_str}

                **Risk Model Explanation**:
                {risk_explanation.strip()}

                User Query: {refined_query}

                **Response**:
                - Provide a concise summary of these transaction(s).
                - Offer context on why the risk category might be Low, Medium, or High, as inferred by our LightGBM model.
                - Maintain a neutral, professional tone.
                """

        # --- CASE 3: No Relevant Data ---
        else:
            final_prompt = f"""
                You are ChainGuardian AI, a smart assistant specializing in blockchain, fraud detection, and compliance. 
                The user query does not match any relevant data from Pinecone (no news articles or transactions found).

                User Query: {refined_query}

                **Response**:
                Please inform the user that no relevant data was found and suggest refining their query.
                """

        # Call the AI model to generate the final response
        response = await get_completion(client, final_prompt)
        return response

    except Exception as e:
        logger.error(f"Error generating RAG response for ChainGuardian AI: {e}")
        return "An error occurred while generating a response."

# Main conversation flow
def conversation_chain(input_text, session_id, logger=logger):
    try:
        # Get conversation log
        conversation_log = get_conversation_string(session_id)

        # Refine query
        refined_query = query_refiner(conversation_log, input_text, bedrock_client, logger)

        # Find matches
        retrieved_data = find_match(refined_query, pinecone_index, bedrock_client, logger)

        # Generate response
        final_response = generate_rag_response(refined_query, retrieved_data, client, logger)

        return final_response

    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        return "An error occurred during the conversation."

# Utility method to get conversation history
def get_conversation_string(session_id: str, chat_sessions: dict) -> str:
    conversation_string = ""
    if session_id in chat_sessions and 'requests' in chat_sessions[session_id] and 'responses' in chat_sessions[session_id]:
        length = min(len(chat_sessions[session_id]['requests']), len(chat_sessions[session_id]['responses']))
        for i in range(length):
            conversation_string += f"Human: {chat_sessions[session_id]['requests'][i]}\n"
            conversation_string += f"Bot: {chat_sessions[session_id]['responses'][i]}\n"
    return conversation_string