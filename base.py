import os
import io
import json
import base64
import logging
import asyncio
import re
import requests
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn

# Set environment variables for development
os.environ["USER_AGENT"] = "my-app/1.0"
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["SCREAMING_FROG_ACCESS_TOKEN"] = "XXXXXXXXX"
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-***")

from dotenv import load_dotenv
load_dotenv()

# LangChain / LLama / Ollama / Vector Store
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Google / Slack / Mail / Drive
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import google.oauth2.credentials
from email.mime.text import MIMEText

# PDF Processing
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader

# Compute Engine / Data Store
from google.cloud import storage
from google.cloud import bigquery

# JWT Authentication
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException

# Slack API
import slack_sdk

# Textract
import boto3

# Legacy
import PyPDF2
from bs4 import BeautifulSoup

# Knowledge Graph / Feedback System
EMAILS_BASE_DIR = "data/emails"
DRIVE_DIR = "data/drive_files"
SLACK_MESSAGES_DIR = "data/slack_messages"
KNOWLEDGE_GRAPH_PATH = "data/knowledge_graph.json"
FEEDBACK_PATH = "data/feedback.json"
PERSIST_DIR = "data/chroma_store"
os.makedirs(EMAILS_BASE_DIR, exist_ok=True)
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(SLACK_MESSAGES_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# Cache and Events
vectorstore_cache = None
vector_store_ready = asyncio.Event()

# Configuration
EMBEDDING_MODEL = "mistral"
LOCAL_CHUNK_SIZE = 512
LOCAL_CHUNK_OVERLAP = 64

# Initialize Models
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=EMBEDDING_MODEL, temperature=0.1)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=LOCAL_CHUNK_SIZE, chunk_overlap=LOCAL_CHUNK_OVERLAP)

# Persistent Storage
os.makedirs(PERSIST_DIR, exist_ok=True)

# -- Google OAuth Configuration
GOOGLE_CLIENT_CONFIG = {
    "web": {
        "client_id": "1034447378302-1sv65j51ruq47k4eqgnpkuarb6en1nba.apps.googleusercontent.com",
        "client_secret": "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",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:8000/callback"]
    }
}

SLACK_CLIENT_ID = os.getenv("SLACK_CLIENT_ID")
SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET")
SLACK_REDIRECT_URI = os.getenv("SLACK_REDIRECT_URI", "https://localhost:8000/slack/callback")

# Helper Functions

def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def extract_email_text(email_data: dict) -> str:
    payload = email_data.get("payload", {})
    body = payload.get("body", {})
    data = body.get("data")
    if data:
        decoded_bytes = base64.urlsafe_b64decode(data.encode("ASCII"))
        return clean_html(decoded_bytes.decode("utf-8", errors="replace"))
    return clean_html(email_data.get("snippet", ""))

def extract_email_metadata(email_data: dict) -> dict:
    metadata = {}
    email_meta = {}
    email_meta["thread_id"] = email_data.get("threadId", "")
    
    headers = email_data.get("payload", {}).get("headers", [])
    metadata_parts = {}
    for header in headers:
        name = header.get("name", "").lower()
        value = header.get("value", "")
        if name in ["subject", "from", "to", "cc"]:
            metadata_parts[name] = value
    email_meta.update(metadata_parts)
    
    parts = email_data.get("payload", {}).get("parts", [])
    attachments = []
    for part in parts:
        if part.get("filename"):
            attachments.append(part["filename"])
    email_meta["attachments"] = len(attachments)
    metadata["email"] = email_meta
    
    pdf_meta = {}
    if attachments:
        pdf_meta["attachment_count"] = len(attachments)
    metadata["pdf"] = pdf_meta
    return metadata

def combine_email_text_and_metadata(email_data: dict) -> str:
    metadata = extract_email_metadata(email_data).get("email", {})
    subject = metadata.get("subject", "")
    sender = metadata.get("from", "")
    to_list = ", ".join(metadata.get("to", "").split(", "))
    body_text = extract_email_text(email_data)
    return f"Subject: {subject}\nFrom: {sender}\nTo: {to_list}\n\n{body_text}"

def split_email_chain(email_text: str) -> list:
    pattern = r"(?:\n-----Original Message-----\n)|(?:\nOn .+? wrote:)"
    splits = re.split(pattern, email_text)
    return [segment.strip() for segment in splits if segment.strip()]

async def build_vector_store():
    global vectorstore_cache
    all_chunks = []
    metadatas = []
    
    # Process emails
    email_folders = os.listdir(EMAILS_BASE_DIR)
    for folder in email_folders:
        chunks_path = os.path.join(EMAILS_BASE_DIR, folder, "chunks.json")
        if os.path.exists(chunks_path):
            with open(chunks_path, "r") as f:
                chunks = json.load(f)
            all_chunks.extend(chunks)
            metadatas.extend([{"source": f"email:{folder}"}] * len(chunks))
    
    # Process PDFs
    pdf_chunks_path = os.path.join("data", "processed_pdfs.json")
    if os.path.exists(pdf_chunks_path):
        with open(pdf_chunks_path, "r") as f:
            pdf_data = json.load(f)
        for filename, chunks in pdf_data.items():
            all_chunks.extend(chunks)
            metadatas.extend([{"source": f"pdf:{filename}"}] * len(chunks))
    
    # Process Slack messages
    slack_files = os.listdir(SLACK_MESSAGES_DIR)
    for filename in slack_files:
        file_path = os.path.join(SLACK_MESSAGES_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                messages = json.load(f)
            for message in messages:
                text = message.get("text", "")
                if text:
                    chunks = text_splitter.split_text(text)
                    all_chunks.extend(chunks)
                    metadatas.extend([{"source": f"slack:{filename}"}] * len(chunks))
        except Exception as e:
            logging.error(f"Error processing Slack file {filename}: {e}")
    
    vectorstore_cache = Chroma.from_texts(
        all_chunks,
        embeddings,
        metadatas=metadatas,
        collection_name="hybrid_documents",
        persist_directory=PERSIST_DIR
    )
    vector_store_ready.set()

def generate_query_prompt(query, context):
    return (
        f"You are an AI assistant providing **precise answers**.\n"
        f"Answer the following question **using only the provided context**.\n"
        f"Do not make up information or provide vague responses.\n"
        f"--- USER QUESTION ---\n"
        f"{query}\n"
        f"--- CONTEXT ---\n"
        f"{context}\n"
    )

def generate_email_prompt(recipient_query, query, context):
    return (
        f"You are an AI assistant generating a **professional email** based on the user's request.\n"
        f"Answer must be direct and professional, using context provided.\n"
        f"--- USER REQUEST ---\n"
        f"{query}\n"
        f"--- CONTEXT ---\n"
        f"{context}\n"
    )

def parse_email_response(response):
    lines = response.split("\n")
    subject = ""
    body = ""
    start_body = False
    for line in lines:
        if line.startswith("Subject:"):
            subject = line.split("Subject:")[1].strip()
        elif start_body:
            body += line + "\n"
        elif line.startswith("Dear") or line.startswith("Thank"):
            start_body = True
            body += line + "\n"
    return subject.strip(), body.strip()

def send_email(recipient_email, subject, message_body):
    try:
        credentials = None
        if os.path.exists("credentials.json"):
            credentials = google.oauth2.credentials.Credentials.from_authorized_user_file("credentials.json")
        else:
            return False
        
        service = build("gmail", "v1", credentials=credentials)
        message = MIMEText(message_body)
        message["to"] = recipient_email
        message["subject"] = subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        send_request = {"raw": raw_message}
        service.users().messages().send(userId="me", body=send_request).execute()
        return True
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        return False

def get_knowledge_graph_context():
    try:
        with open(KNOWLEDGE_GRAPH_PATH, "r") as f:
            knowledge_graph = json.load(f)
        formatted_data = [
            f"Name: {info['name']}, Role: {info['role']}, Email: {email}"
            for email, info in knowledge_graph.get("people", {}).items()
        ]
        return "\n".join(formatted_data)
    except Exception as e:
        return "No structured knowledge available."

def update_knowledge_graph(sender_email, sender_name, role):
    try:
        with open(KNOWLEDGE_GRAPH_PATH, "r+") as f:
            knowledge_graph = json.load(f)
            knowledge_graph.setdefault("people", {})
            knowledge_graph["people"][sender_email] = {
                "name": sender_name,
                "role": role
            }
            f.seek(0)
            json.dump(knowledge_graph, f, indent=2)
    except Exception as e:
        logging.error(f"Error updating knowledge graph: {e}")

def extract_role_from_content(text):
    role_keywords = {
        "vp": "VP",
        "vice president": "VP",
        "director": "Director",
        "manager": "Manager",
        "executive": "Executive",
        "head": "Head",
        "chief": "Chief"
    }
    for keyword in role_keywords:
        if keyword in text.lower():
            return role_keywords[keyword]
    return "Unknown Role"

def get_recipient_email(query):
    try:
        with open(KNOWLEDGE_GRAPH_PATH, "r") as f:
            knowledge_graph = json.load(f)
        query = query.lower().strip()
        words = query.split()
        subqueries = sorted([
            " ".join(combo) for i in range(1, len(words) + 1)
            for combo in combinations(words, i)
        ], key=len, reverse=True)
        for subquery in subqueries:
            for email, info in knowledge_graph.get("people", {}).items():
                name_lower = info.get("name", "").lower()
                role_lower = info.get("role", "").lower()
                if subquery in name_lower or subquery in role_lower:
                    return email
        return None
    except Exception as e:
        return None

# Slack-specific functions

def update_knowledge_graph_auto(text):
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
            update_knowledge_graph(ent.text.strip(), ent.text.strip(), ent.label_)
    except Exception as e:
        pass

# FastAPI Endpoints

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head>
        <title>Hybrid RAG System</title>
      </head>
      <body>
        <h1>Hybrid RAG System for Adam</h1>
        <ul>
          <li><a href="/login">Login with Google</a></li>
          <li><a href="/slack/login">Login with Slack</a></li>
          <li><a href="/build_store">[Re]build Vector Store</a></li>
          <li><a href="/query?q=What%20is%20the%20latest%20project%20update">Query the System</a></li>
          <li><a href="/slack_ingest">Ingest Slack Messages</a></li>
          <li><a href="/feedback">Submit Feedback</a></li>
        </ul>
      </body>
    </html>
    """

@app.get("/login")
def login():
    flow = Flow.from_client_config(GOOGLE_CLIENT_CONFIG, scopes=[
        "https://mail.google.com/",
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/gmail.send"
    ])
    flow.redirect_uri = GOOGLE_CLIENT_CONFIG["web"]["redirect_uris"][0]
    authorization_url, _ = flow.authorization_url(access_type="offline")
    return RedirectResponse(authorization_url)

@app.get("/callback")
async def callback(request: Request):
    flow = Flow.from_client_config(GOOGLE_CLIENT_CONFIG, scopes=[
        "https://mail.google.com/",
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/gmail.send"
    ])
    flow.redirect_uri = GOOGLE_CLIENT_CONFIG["web"]["redirect_uris"][0]
    authorization_response = str(request.url)
    flow.fetch_token(authorization_response=authorization_response)
    credentials = flow.credentials
    with open("credentials.json", "w") as f:
        f.write(credentials.to_json())
    await process_emails_and_drive(credentials)
    await process_pdf_attachments(credentials)
    return {"message": "Google OAuth completed and data processed. If you also use Slack, proceed to the Slack OAuth."}

@app.get("/slack/login")
def slack_login():
    auth_url = (
        "https://slack.com/oauth/v2/authorize"
        f"?client_id={SLACK_CLIENT_ID}"
        f"&scope=app_mentions:read,channels:history,chat:write,groups:history,im:history,mpim:history"
        f"&user_scope=identity.basic,identity.email"
        f"&redirect_uri={SLACK_REDIRECT_URI}"
    )
    return RedirectResponse(auth_url)

@app.get("/slack/callback")
def slack_callback(request: Request):
    code = request.query_params.get("code")
    data = {
        "client_id": SLACK_CLIENT_ID,
        "client_secret": SLACK_CLIENT_SECRET,
        "code": code,
        "redirect_uri": SLACK_REDIRECT_URI
    }
    response = requests.post("https://slack.com/api/oauth.v2.access", data=data)
    access_data = response.json()
    if access_data.get("ok"):
        with open("slack_credentials.json", "w") as f:
            json.dump(access_data, f)
    return {"message": "Slack OAuth completed. Ingest Slack messages to populate the database."}

@app.get("/slack_ingest")
async def ingest_slack_messages():
    if not os.path.exists("slack_credentials.json"):
        return {"error": "Slack credentials not found. Please authenticate via /slack/login"}
    
    with open("slack_credentials.json", "r") as f:
        slack_data = json.load(f)
    bearer_token = slack_data["access_token"]
    client = slack_sdk.WebClient(token=bearer_token)
    
    channels = client.conversations_list(exclude_archived=True, types="public_channel,private_channel,mpim,im")
    for channel in channels["channels"]:
        try:
            messages = []
            cursor = None
            while True:
                response = client.conversations_history(channel=channel["id"], cursor=cursor)
                messages += response["messages"]
                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
            filename = f"{channel['id']}.json"
            with open(os.path.join(SLACK_MESSAGES_DIR, filename), "w") as f:
                json.dump(messages, f)
        except Exception as e:
            logging.error(f"Error ingesting Slack channel {channel['id']}: {e}")
    return {"message": "Slack messages ingested successfully."}

@app.get("/build_store")
async def build_store_endpoint():
    await build_vector_store()
    return {"status": "Vector store built."}

@app.get("/query")
async def query_endpoint(request: Request, q: str):
    try:
        if not vector_store_ready.is_set():
            await build_vector_store()
        
        if not vectorstore_cache:
            vectorstore_cache = Chroma(
                collection_name="hybrid_documents",
                embedding_function=embeddings,
                persist_directory=PERSIST_DIR
            )
        
        retriever = vectorstore_cache.as_retriever(k=12)
        retrieved_docs = retriever.invoke(q)
        doc_sources = [doc.metadata['source'] for doc in retrieved_docs]
        doc_content = [doc.page_content for doc in retrieved_docs]
        
        knowledge_graph_context = get_knowledge_graph_context()
        enriched_context = f"Knowledge Graph:\n{knowledge_graph_context}\n\nRetrieved Documents:\n{' '.join(doc_content)}"
        
        if q.startswith("send email to"):
            recipient_query = re.sub(r"^\s*send email to\s+", "", q, flags=re.IGNORECASE).strip()
            recipient_email = get_recipient_email(recipient_query)
            if recipient_email:
                prompt = generate_email_prompt(recipient_query, q, enriched_context)
                response = llm.invoke(prompt)
                subject, body = parse_email_response(response)
                if send_email(recipient_email, subject, body):
                    return {"query": q, "result": f"Email sent to {recipient_email}"}
                else:
                    return {"query": q, "result": "Email sending failed."}
            else:
                return {"query": q, "result": "Recipient not found in knowledge graph."}
        else:
            prompt = generate_query_prompt(q, enriched_context)
            response = llm.invoke(prompt)
            return {"query": q, "result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback_endpoint(request: Request):
    try:
        data = await request.json()
        with open(FEEDBACK_PATH, "a+") as f:
            f.seek(0)
            try:
                feedback_list = json.load(f)
            except json.JSONDecodeError:
                feedback_list = []
            feedback_list.append(data)
            f.truncate(0)
            f.seek(0)
            json.dump(feedback_list, f, indent=2)
        return {"message": "Feedback received."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Admin Tools

async def process_emails_and_drive(credentials):
    gmail_service = build("gmail", "v1", credentials=credentials)
    drive_service = build("drive", "v3", credentials=credentials)
    
    # Process emails
    messages = gmail_service.users().messages().list(userId="me", maxResults=20).execute().get("messages", [])
    for msg in messages:
        msg_data = gmail_service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
        email_folder = os.path.join(EMAILS_BASE_DIR, msg["id"])
        os.makedirs(email_folder, exist_ok=True)
        with open(os.path.join(email_folder, "email.json"), "w") as f:
            json.dump(msg_data, f)
        email_text = combine_email_text_and_metadata(msg_data)
        chunks = text_splitter.split_text(email_text)
        with open(os.path.join(email_folder, "chunks.json"), "w") as f:
            json.dump(chunks, f)
    
    # Process drive files (PDFs)
    pdf_query = "mimeType='application/pdf'"
    drive_files = drive_service.files().list(q=pdf_query, spaces="drive", pageSize=5).execute().get("files", [])
    for file in drive_files:
        file_id = file["id"]
        request = drive_service.files().get_media(fileId=file_id)
        fp = io.BytesIO()
        downloader = MediaIoBaseDownload(fp, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        with open(os.path.join(DRIVE_DIR, file["name"]), "wb") as f:
            f.write(fp.getvalue())

async def process_pdf_attachments(credentials):
    gmail_service = build("gmail", "v1", credentials=credentials)
    messages = gmail_service.users().messages().list(userId="me", maxResults=20).execute().get("messages", [])
    for msg in messages:
        msg_data = gmail_service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
        for part in msg_data["payload"].get("parts", []):
            if part.get("mimeType") == "application/pdf":
                attachment_id = part["body"]["attachmentId"]
                attachment = gmail_service.users().messages().attachments().get(userId="me", messageId=msg["id"], id=attachment_id).execute()
                data = base64.urlsafe_b64decode(attachment["data"])
                filename = part["filename"]
                with open(os.path.join(DRIVE_DIR, filename), "wb") as f:
                    f.write(data)

# Run the application

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")