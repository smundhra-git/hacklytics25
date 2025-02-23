import os
import io
import json
import base64
import logging
import warnings
import asyncio
import re
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn
import requests  # for Slack token exchange
from urllib.parse import urlencode  

# Set environment variables for development (DO NOT use insecure settings in production)
os.environ["USER_AGENT"] = "my-app/1.0"
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

from dotenv import load_dotenv
load_dotenv()

# --- Imports from LangChain, Mistral (via Ollama interface), and Slack/Web APIs ---
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama  # Using Mistral via Ollama interface
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from bs4 import BeautifulSoup

# We use Slack's Web API endpoints (see https://api.slack.com/web) via requests.
# --- Persistent storage and global cache ---
DATA_DIR = "data"
SLACK_MESSAGES_DIR = os.path.join(DATA_DIR, "slack_messages")
PERSIST_DIR = os.path.join(DATA_DIR, "chroma_store")
os.makedirs(SLACK_MESSAGES_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)
vectorstore_cache = None
vector_store_ready = asyncio.Event()  # Signals when the vector store is built

# --- Configuration ---
# Use Mistral as the base model (via our Ollama interface)
EMBEDDING_MODEL = "mistral"  # Set to your actual Mistral model identifier
# For Slack messages, use a moderate chunk size.
LOCAL_CHUNK_SIZE = 512      # Recommended between 256 and 512 characters
LOCAL_CHUNK_OVERLAP = 256

# --- Initialize Models and Services ---
model_local = ChatOllama(model=EMBEDDING_MODEL, temperature=0.2)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# --- Slack OAuth Configuration ---
SLACK_CLIENT_ID = os.getenv("SLACK_CLIENT_ID")
SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET")
SLACK_REDIRECT_URI = os.getenv("SLACK_REDIRECT_URI", "https://localhost:8000/slack_oauth_callback")
# Scopes as a space-delimited string (make sure these scopes are granted in your Slack app)
SLACK_SCOPES = ("app_mentions:read calls:read canvases:read channels:read channels:history "
                "conversations.connect:read emoji:read files:read groups:read groups:history "
                "im:read im:history links:read metadata.message:read")

# --- Create FastAPI App ---
app = FastAPI()

# --- Helper Functions ---
def clean_html(raw_html: str) -> str:
    """Remove HTML tags and unescape HTML entities."""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def split_text(text: str, chunk_size: int = LOCAL_CHUNK_SIZE, overlap: int = LOCAL_CHUNK_OVERLAP) -> list:
    """Split text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def update_knowledge_graph(entity: str, role: str):
    """Update a local knowledge graph stored as JSON."""
    kg_path = os.path.join(DATA_DIR, "knowledge_graph.json")
    try:
        with open(kg_path, "r") as f:
            kg = json.load(f)
    except Exception:
        kg = {}
    kg[entity] = role
    with open(kg_path, "w") as f:
        json.dump(kg, f, indent=2)
    logging.info("Knowledge graph updated: %s -> %s", entity, role)

def update_knowledge_graph_auto(text: str):
    """Automatically update the knowledge graph using spaCy."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        for ent in doc.ents:
            update_knowledge_graph(ent.text, ent.label_)
    except Exception as e:
        logging.error("Failed to auto-update knowledge graph: %s", e)

# --- Slack Ingestion & Vector Store Build ---
async def build_vector_store():
    global vectorstore_cache
    all_chunks = []
    metadatas = []
    
    slack_files = os.listdir(SLACK_MESSAGES_DIR)
    if not slack_files:
        logging.info("No Slack message files found. Aborting vector store build.")
        return {"status": "No Slack messages found"}
    
    async def process_slack_file(filename):
        file_path = os.path.join(SLACK_MESSAGES_DIR, filename)
        data = await asyncio.to_thread(lambda: json.load(open(file_path, "r", encoding="utf-8")))
        texts = [msg.get("text", "") for msg in data if msg.get("text")]
        for t in texts:
            update_knowledge_graph_auto(t)
        chunks = []
        for t in texts:
            # Split and filter out empty chunks
            t_chunks = [chunk for chunk in split_text(t) if chunk.strip()]
            chunks.extend(t_chunks)
        return chunks, len(chunks)
    
    results = await asyncio.gather(*[process_slack_file(f) for f in slack_files])
    for idx, (chunks, count) in enumerate(results):
        if count > 0:
            all_chunks.extend(chunks)
            metadatas.extend([{"source": f"slack:{slack_files[idx]}"}] * count)
    
    if not all_chunks:
        logging.warning("No valid text chunks found from Slack messages.")
        return {"status": "No valid documents found for vector store"}
    
    vectorstore_cache = Chroma.from_texts(
        texts=all_chunks,
        embedding=OllamaEmbeddings(model=os.getenv("OLLAMA_MODEL", EMBEDDING_MODEL)),
        metadatas=metadatas,
        collection_name="slack_documents",
        persist_directory=PERSIST_DIR,
    )
    vector_store_ready.set()  # Signal that the vector store build is complete.
    return {"status": "Vector store built", "num_chunks": len(all_chunks)}

# --- Slack OAuth Endpoints ---
@app.get("/slack/login", response_class=HTMLResponse)
def slack_login():
    """Redirect the user to Slack's OAuth authorization URL."""
    params = {
        "client_id": SLACK_CLIENT_ID,
        "scope": SLACK_SCOPES,
        "redirect_uri": SLACK_REDIRECT_URI,
    }
    auth_url = f"https://slack.com/oauth/v2/authorize?{urlencode(params)}"
    return RedirectResponse(auth_url)

@app.get("/slack_oauth_callback")
def slack_oauth_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code in callback")
    
    response = requests.post(
        "https://slack.com/api/oauth.v2.access",
        data={
            "client_id": SLACK_CLIENT_ID,
            "client_secret": SLACK_CLIENT_SECRET,
            "code": code,
            "redirect_uri": SLACK_REDIRECT_URI,
        },
    )
    data = response.json()
    if not data.get("ok"):
        raise HTTPException(status_code=400, detail=f"Error from Slack: {data.get('error')}")
    
    # Extract bot token from bot_user.access_token
    bot_token = data.get("bot_user", {}).get("access_token")
    if not bot_token:
        raise HTTPException(status_code=400, detail="Bot token not found in response")
    
    token_path = os.path.join(DATA_DIR, "slack_token.json")
    with open(token_path, "w") as f:
        json.dump({"bot_token": bot_token}, f, indent=2)
    
    return {"message": "Slack OAuth successful. You can now ingest Slack data."}

def extract_message_text(message: dict) -> str:
    """
    Extracts text from a Slack message.
    Prioritizes the "text" field; if empty, attempts to extract text from "blocks".
    """
    # Prefer the standard "text" field if available.
    text = message.get("text", "").strip()
    if text:
        return text

    # If "text" is empty, try extracting text from "blocks"
    texts = []
    if "blocks" in message:
        for block in message["blocks"]:
            if block.get("type") in ["section", "rich_text"]:
                # If the block contains a "text" dict
                if "text" in block and isinstance(block["text"], dict):
                    block_text = block["text"].get("text", "")
                    if block_text:
                        texts.append(block_text.strip())
                # If the block contains an "elements" list with text parts
                elif "elements" in block:
                    for element in block["elements"]:
                        if isinstance(element, dict) and element.get("type") == "text":
                            block_text = element.get("text", "")
                            if block_text:
                                texts.append(block_text.strip())
    return "\n".join(texts)


@app.get("/slack_ingest")
async def slack_ingest():
    token_path = os.path.join(DATA_DIR, "slack_token.json")
    if not os.path.exists(token_path):
        raise HTTPException(
            status_code=400,
            detail="Slack token not found. Please login via /slack/login"
        )
    with open(token_path, "r") as f:
        slack_token = json.load(f)
    bot_token = slack_token.get("bot_token")
    if not bot_token:
        raise HTTPException(
            status_code=400,
            detail="Slack bot token missing in token file"
        )
    
    headers = {"Authorization": f"Bearer {bot_token}"}
    
    # Fetch public channels
    channels_response = requests.get(
        "https://slack.com/api/conversations.list",
        params={"types": "public_channel"},
        headers=headers
    )
    channels_data = channels_response.json()
    if not channels_data.get("ok"):
        raise HTTPException(
            status_code=400,
            detail=f"Error fetching channels: {channels_data.get('error')}"
        )
    
    # Process each channel's history
    for channel in channels_data.get("channels", []):
        channel_id = channel["id"]
        channel_name = channel.get("name", "unknown")
        messages = []
        cursor = None
        while True:
            params = {"channel": channel_id, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            history_response = requests.get(
                "https://slack.com/api/conversations.history",
                params=params,
                headers=headers
            )
            history_data = history_response.json()
            if not history_data.get("ok"):
                logging.error(f"Error fetching history for {channel_name}: {history_data.get('error')}")
                break
            messages.extend(history_data.get("messages", []))
            cursor = history_data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        
        # Save processed messages
        processed_messages = []
        for msg in messages:
            extracted_text = extract_message_text(msg)
            if extracted_text:
                msg["extracted_text"] = extracted_text
                processed_messages.append(msg)
        if processed_messages:
            with open(os.path.join(SLACK_MESSAGES_DIR, f"{channel_id}.json"), "w", encoding="utf-8") as f:
                json.dump(processed_messages, f, indent=2)
    
    return {"message": "Slack messages ingested successfully."}
# --- Build Vector Store Endpoint ---
@app.get("/build_store")
async def build_store_endpoint():
    """Synchronously build (or update) the vector store from ingested Slack messages."""
    response = await build_vector_store()  # Wait until build is complete.
    return response

# --- Query Endpoint ---
@app.get("/query")
async def query_endpoint(q: str):
    """
    Query the vector store.
    Retrieves enriched chunks from Slack messages and uses a hybrid RAG approach to generate context-aware answers.
    """
    global vectorstore_cache
    try:
        # Instantiate the chat LLM with a lower temperature.
        llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", EMBEDDING_MODEL), temperature=0.1)
        
        # Ensure the vector store is built.
        if not vector_store_ready.is_set():
            await build_vector_store()
        
        # Load the vector store.
        vectorstore_cache = Chroma(
            collection_name="documents",
            embedding_function=OllamaEmbeddings(model=os.getenv("OLLAMA_MODEL", EMBEDDING_MODEL)),
            persist_directory=PERSIST_DIR,
        )
        retriever = vectorstore_cache.as_retriever(search_kwargs={"k": 20})
        retrieved_docs = retriever.get_relevant_documents(q)
        enriched_context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        logging.info("Enriched Context:\n%s", enriched_context)
        
        # Create a custom prompt.
        custom_prompt_text = (
            "You are a knowledgeable executive assistant. Using the following context, answer the question accurately. "
            "If unsure, ask for clarification.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        custom_prompt = ChatPromptTemplate.from_template(custom_prompt_text)
        custom_llm_chain = LLMChain(llm=llm, prompt=custom_prompt)
        
        def run_custom_qa(question: str) -> str:
            docs = retriever.get_relevant_documents(question)
            enriched = "\n\n".join(
                f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
                for doc in docs
            )
            prompt_input = {"context": enriched, "question": question}
            return custom_llm_chain.run(prompt_input)
        
        result = await asyncio.to_thread(run_custom_qa, q)
        return {"query": q, "result": result, "context": enriched_context}
    except Exception as e:
        logging.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Query processing error")

# --- Feedback Endpoint ---
@app.post("/feedback")
async def feedback_endpoint(request: Request):
    """
    Collect user feedback.
    Expects JSON with:
      - query: the original query,
      - result: the system's answer,
      - correct_answer: (optional) the corrected answer provided by the user,
      - feedback: comments (e.g., "too vague", "incorrect").
    """
    try:
        data = await request.json()
        feedback_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": data.get("query", ""),
            "result": data.get("result", ""),
            "correct_answer": data.get("correct_answer", ""),
            "feedback": data.get("feedback", "")
        }
        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        else:
            feedback_data = []
        feedback_data.append(feedback_entry)
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2)
        logging.info("Feedback received: %s", feedback_entry)
        return {"message": "Feedback received. Thank you!"}
    except Exception as e:
        logging.error(f"Feedback processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Feedback processing error")

# --- Index Endpoint ---
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head>
        <title>Hybrid Slack RAG for Adam</title>
      </head>
      <body>
        <h1>Hybrid Slack RAG System for Adam</h1>
        <ul>
          <li><a href="/slack/login">Slack Login (OAuth)</a></li>
          <li><a href="/slack_ingest">Ingest Slack Messages</a></li>
          <li><a href="/build_store">Build/Update Vector Store</a></li>
          <li><a href="/query?q=Your+Question+Here">Query the System</a></li>
          <li><a href="/feedback">Submit Feedback</a></li>
        </ul>
      </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, 
                 ssl_keyfile="key.pem", 
                 ssl_certfile="cert.pem")
