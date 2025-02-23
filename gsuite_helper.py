import os
from dotenv import load_dotenv
import base64
from bs4 import BeautifulSoup
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
import asyncio
from itertools import combinations
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain_ollama import ChatOllama 
from email.mime.text import MIMEText
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks

load_dotenv()

DATA_DIR = "data"
KNOWLEDGE_GRAPH_PATH = os.path.join(DATA_DIR, "knowledge_graph.json")
EMAILS_BASE_DIR = os.path.join(DATA_DIR, "emails")
DRIVE_DIR = os.path.join(DATA_DIR, "drive_files")
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.json")

# To store user feedback
os.makedirs(EMAILS_BASE_DIR, exist_ok=True)
os.makedirs(DRIVE_DIR, exist_ok=True)
if not os.path.exists(KNOWLEDGE_GRAPH_PATH):
    with open(KNOWLEDGE_GRAPH_PATH, "w") as f:
        json.dump({}, f)
if not os.path.exists(FEEDBACK_PATH):
    with open(FEEDBACK_PATH, "w") as f:
        json.dump([], f)

PERSIST_DIR = os.path.join("data", "chroma_store")
os.makedirs(PERSIST_DIR, exist_ok=True)

LOCAL_CHUNK_SIZE = 512      # Recommended between 256 and 512 characters
LOCAL_CHUNK_OVERLAP = 64

local_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=LOCAL_CHUNK_SIZE, chunk_overlap=LOCAL_CHUNK_OVERLAP
)

CLIENT_CONFIG = {
    "web": {
        "client_id": os.getenv("CLIENT_ID"),
        "client_secret": os.getenv("CLIENT_SECRET"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:8000/oauth2callback"]
    }
}

SCOPES = [
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/gmail.send"
]

vectorstore_cache = None

gmail_client = build("gmail", "v1")
drive_client = build("drive", "v3")


EMBEDDING_MODEL = "mistral" 
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=EMBEDDING_MODEL, temperature=0.1)

vector_store_ready = asyncio.Event() 

def extract_email_text(email_data: dict) -> str:
    """Extract the email body text (decoded if possible), then clean HTML."""
    payload = email_data.get("payload", {})
    body = payload.get("body", {})
    data = body.get("data")
    if data:
        try:
            decoded_bytes = base64.urlsafe_b64decode(data.encode("ASCII"))
            decoded_text = decoded_bytes.decode("utf-8", errors="replace")
            return clean_html(decoded_text)
        except Exception as e:
            pass
    return clean_html(email_data.get("snippet", ""))

def clean_html(raw_html: str) -> str:
    """Remove HTML tags and unescape HTML entities."""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def extract_role_from_content(text):
    """Extract role keywords from text using a simple keyword search."""
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

def update_knowledge_graph(sender_email, sender_name, role):
    """Update the knowledge graph with sender details."""
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
        pass


def combine_email_text_and_metadata(email_data: dict) -> str:
    """
    Combine email metadata (subject, sender, recipients) with the email body.
    If the email contains a chain, split it into individual messages.
    """
    metadata = extract_email_metadata(email_data).get("email", {})
    subject = metadata.get("subject", "")
    sender = metadata.get("from", "")
    to_list = ", ".join(metadata.get("to", []))
    body_text = extract_email_text(email_data)
    email_segments = split_email_chain(body_text)
    chain_text = "\n\n--- Email Separator ---\n\n".join(email_segments) if email_segments else body_text
    return f"Subject: {subject}\nFrom: {sender}\nTo: {to_list}\n\n{chain_text}"

def split_email_chain(email_text: str) -> list:
    """
    Split a long email chain into individual messages using common delimiters.
    """
    pattern = r"(?:\n-----Original Message-----\n)|(?:\nOn .+? wrote:)"
    splits = re.split(pattern, email_text)
    return [segment.strip() for segment in splits if segment.strip()]

def extract_email_metadata(email_data: dict) -> dict:
    """
    Extract metadata from an email and (if available) from its PDF attachments.
    Returns a dict with keys "email" and "pdf".
    """
    metadata = {}
    email_meta = {}
    email_meta["thread_id"] = email_data.get("threadId", "")
    
    headers = email_data.get("payload", {}).get("headers", [])
    subject = ""
    sender = ""
    to_list = []
    cc_list = []
    attachments_list = []
    for header in headers:
        hname = header.get("name", "").lower()
        if hname == "subject":
            subject = header.get("value", "")
        elif hname == "from":
            sender = header.get("value", "")
        elif hname == "to":
            to_list = [addr.strip() for addr in header.get("value", "").split(",")]
        elif hname == "cc":
            cc_list = [addr.strip() for addr in header.get("value", "").split(",")]
    email_meta["subject"] = subject
    email_meta["from"] = sender
    email_meta["to"] = to_list
    email_meta["cc"] = cc_list
    
    parts = email_data.get("payload", {}).get("parts", [])
    for part in parts:
        filename = part.get("filename", "")
        if filename:
            attachments_list.append(filename)
    email_meta["attachments"] = len(attachments_list)
    metadata["email"] = email_meta
    
    pdf_meta = {}
    pdf_attachment = None
    for att in attachments_list:
        if att.lower().endswith(".pdf"):
            pdf_attachment = att
            break
    if pdf_attachment:
        pdf_path = os.path.join(DRIVE_DIR, pdf_attachment)
        if os.path.exists(pdf_path):
            pdf_meta = {"path": pdf_path}  # Placeholder; customize as needed
    metadata["pdf"] = pdf_meta
    return metadata


async def build_vector_store():
    global vectorstore_cache
    all_chunks = []
    metadatas = []
    
    email_folders = os.listdir(EMAILS_BASE_DIR)
    for folder in email_folders:
        chunks_path = os.path.join(EMAILS_BASE_DIR, folder, "chunks.json")
        if os.path.exists(chunks_path):
            with open(chunks_path, "r") as f:
                chunks = json.load(f)
            all_chunks.extend(chunks)
            metadatas.extend([{"source": f"email:{folder}"}] * len(chunks))
    
    # Process PDFs from processed_data.json
    pdf_chunks_path = os.path.join(DATA_DIR, "processed_data.json")
    if os.path.exists(pdf_chunks_path):
        with open(pdf_chunks_path, "r") as f:
            pdf_data = json.load(f)
        for filename, chunks in pdf_data.items():
            all_chunks.extend(chunks)
            metadatas.extend([{"source": f"pdf:{filename}"}] * len(chunks))
    
    vectorstore_cache = Chroma.from_texts(
        all_chunks,
        embeddings,
        metadatas=metadatas,
        collection_name="documents",
        persist_directory=PERSIST_DIR
    )
    vector_store_ready.set()



def get_knowledge_graph_context():
    """Retrieve structured knowledge from `knowledge_graph.json` and format it for the model."""
    try:
        with open(KNOWLEDGE_GRAPH_PATH, "r") as f:
            knowledge_graph = json.load(f)
        
        formatted_data = []
        for email, info in knowledge_graph.get("people", {}).items():
            name = info.get("name", "Unknown")
            role = info.get("role", "Unknown Role")
            formatted_data.append(f"Name: {name}, Role: {role}, Email: {email}")

        return "\n".join(formatted_data) if formatted_data else "No structured knowledge available."

    except Exception as e:
        return "Error retrieving structured knowledge."
    
def get_recipient_email(query):
    """Retrieve recipient's email by checking every subcombination of words in `knowledge_graph.json`."""
    print(f"Original Query: {query}")  # Debugging

    try:
        with open(KNOWLEDGE_GRAPH_PATH, "r") as f:
            knowledge_graph = json.load(f)

        query = query.lower().strip()

        # Extract all subcombination phrases from the query
        words = query.split()
        subqueries = sorted(
            [" ".join(combo) for i in range(1, len(words) + 1) for combo in combinations(words, i)],
            key=len,
            reverse=True  # Prioritize longer matches
        )

        # Search for the best match in `knowledge_graph.json`
        for subquery in subqueries:
            for email, info in knowledge_graph.get("people", {}).items():
                name = info.get("name", "").lower().strip()
                role = info.get("role", "").lower().strip()

                if subquery == name or subquery == role:
                    return email  # Return the first best match

        return None  # No match found

    except Exception as e:
        return None

def parse_email_response(response):
    try:
        parts = response.split("--- Email Template ---", 1)[1]
        subject = parts.split("Subject:", 1)[1].split("Body:", 1)[0].strip()
        body = parts.split("Body:", 1)[1].split("--- End Email Template ---")[0].strip()
        return subject, body
    except:
        return "Confirmation of Your Request", response
    

def get_gmail_service():
    if os.path.exists("token.json"):
        return build("gmail", "v1", credentials=Credentials.from_authorized_user_file("token.json"))
    else:
        raise Exception("No credentials found. Please log in first.")




def generate_email_prompt(role, query, context):
    return (
        f"You are an AI assistant generating a professional email for {role}.\n"
        f"Your task is to write a **direct, relevant, and well-structured email** based ONLY on the user's request.\n"
        f"DO NOT include unnecessary details, filler text, or vague language.\n"
        f"Keep it **short and professional** while addressing the request directly.\n\n"
        f"--- USER REQUEST ---\n"
        f"{query}\n"
        f"--- CONTEXT (if relevant) ---\n"
        f"{context}\n\n"
        f"📝 **STRICT RESPONSE FORMAT:**\n"
        f"--- Email Template ---\n"
        f"Subject: [Short and precise subject]\n\n"
        f"Dear {role},\n\n"
        f"[Clear, professional, and to-the-point message. Keep it concise.]\n\n"
        f"Best regards,\n"
        f"Shlok\n"
        f"--- End Email Template ---"
    )

def send_email(recipient_email, subject, body):
    try:
        service = get_gmail_service()
        
        # Ensure `body` is plain text (extract from AIMessage if necessary)
        if isinstance(body, dict) and "content" in body:
            body = body["content"]  # Extract actual text
        elif hasattr(body, "content"):
            body = body.content  # Extract text from AIMessage

        message = MIMEText(body)
        message["to"] = recipient_email
        message["subject"] = subject

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        send_request = {"raw": raw_message}

        response = service.users().messages().send(userId="me", body=send_request).execute()

        return True
    except Exception as e:
        return False
    
def generate_query_prompt(query, context):
    return (
        f"You are an AI assistant providing **concise, factual, and precise** answers.\n"
        f"Your task is to answer the following question **ONLY using the provided context**.\n"
        f"DO NOT make up information, assume details, or add unnecessary text.\n"
        f"Keep the response **short, clear, and direct.**\n\n"
        f"--- USER QUESTION ---\n"
        f"{query}\n"
        f"--- CONTEXT (if relevant) ---\n"
        f"{context}\n\n"
        f"📝 **STRICT RESPONSE FORMAT:**\n"
        f"Answer in **one or two clear sentences**, without any extra explanation.\n"
        f"Only provide information that is available in the context."
    )

def extract_role_from_content(text):
    """Extract role keywords from text using a simple keyword search."""
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

def create_message(sender, to, subject, message_text):
    """Create a MIMEText message."""
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': encoded_message}

def send_message(service, user_id, message):
    """Send a MIMEText message via the Gmail API."""
    try:
        sent_message = service.users().messages().send(userId=user_id, body=message).execute()
        return {"status": "Message sent", "message_id": sent_message["id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error sending email")
    


def get_gmail_service():
    """Retrieve authenticated Gmail API service."""
    if not os.path.exists("token.json"):
        raise Exception("No credentials found. Please log in first.")

    with open("token.json", "r") as token_file:
        credentials_data = json.load(token_file)

    # Ensure refresh token exists
    if "refresh_token" not in credentials_data:
        raise Exception("Refresh token missing! Please log in again and grant full permissions.")

    credentials = Credentials.from_authorized_user_info(credentials_data, SCOPES)
    return build("gmail", "v1", credentials=credentials)