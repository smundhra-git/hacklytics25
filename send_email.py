import os
import io
import json
import base64
import logging
import asyncio
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn
import re

# Set environment variables for development (DO NOT use insecure settings in production)
os.environ["USER_AGENT"] = "my-app/1.0"
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

from dotenv import load_dotenv
load_dotenv()

# --- Imports from LangChain, Mistral (via Ollama interface), and Google APIs ---
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama  # Using Mistral via Ollama interface
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import PyPDF2
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from email.mime.text import MIMEText

# --- Persistent storage and global cache ---
PERSIST_DIR = os.path.join("data", "chroma_store")
os.makedirs(PERSIST_DIR, exist_ok=True)
vectorstore_cache = None
vector_store_ready = asyncio.Event()  # Signals when vector store build is complete

# --- Configuration ---
# Use Mistral as the base model (via our Ollama interface)
EMBEDDING_MODEL = "mistral"  # Update this with your actual mistral model identifier
# For short emails/PDFs, use a moderate chunk size.
LOCAL_CHUNK_SIZE = 512      # Recommended between 256 and 512 characters
LOCAL_CHUNK_OVERLAP = 64

# --- Initialize Models and Services ---
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=EMBEDDING_MODEL, temperature=0.1)
# Initialize OpenAI models
# embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
# llm = OpenAI(temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))

# --- Initialize Google API Clients ---
gmail_client = build("gmail", "v1")
drive_client = build("drive", "v3")

# embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
# llm = OpenAI(temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))
local_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=LOCAL_CHUNK_SIZE, chunk_overlap=LOCAL_CHUNK_OVERLAP
)

# --- Setup Logging and Warnings ---
logging.getLogger("PyPDF2").setLevel(logging.ERROR)

# --- Create FastAPI App ---
app = FastAPI()

# --- Define Base Directories ---
DATA_DIR = "data"
EMAILS_BASE_DIR = os.path.join(DATA_DIR, "emails")
DRIVE_DIR = os.path.join(DATA_DIR, "drive_files")
KNOWLEDGE_GRAPH_PATH = os.path.join(DATA_DIR, "knowledge_graph.json")  # For structured entity data
FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.json")  # To store user feedback
os.makedirs(EMAILS_BASE_DIR, exist_ok=True)
os.makedirs(DRIVE_DIR, exist_ok=True)
if not os.path.exists(KNOWLEDGE_GRAPH_PATH):
    with open(KNOWLEDGE_GRAPH_PATH, "w") as f:
        json.dump({}, f)
if not os.path.exists(FEEDBACK_PATH):
    with open(FEEDBACK_PATH, "w") as f:
        json.dump([], f)

# --- OAuth Client Configuration ---
CLIENT_CONFIG = {
    "web": {
        "client_id": "1034447378302-1sv65j51ruq47k4eqgnpkuarb6en1nba.apps.googleusercontent.com",
        "client_secret": "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",
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

# --- Helper Functions ---

def clean_html(raw_html: str) -> str:
    """Remove HTML tags and unescape HTML entities."""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

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
            logging.error(f"Error decoding email body: {e}")
    return clean_html(email_data.get("snippet", ""))

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

def split_email_chain(email_text: str) -> list:
    """
    Split a long email chain into individual messages using common delimiters.
    """
    pattern = r"(?:\n-----Original Message-----\n)|(?:\nOn .+? wrote:)"
    splits = re.split(pattern, email_text)
    return [segment.strip() for segment in splits if segment.strip()]

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


# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
def index():
    """Landing page with links to login and other endpoints."""
    return """
    <html>
      <head>
        <title>Hybrid RAG for Adam</title>
      </head>
      <body>
        <h1>Hybrid RAG System for Adam</h1>
        <p><a href="/login">Sign in with Google to ingest your email and drive data</a></p>
        <p><a href="/build_store">Build/Update Vector Store</a></p>
        <p><a href="/query?q=Your+Question+Here">Query the Hybrid System</a></p>
        <p><a href="/feedback">Submit Feedback</a></p>
      </body>
    </html>
    """

@app.get("/login")
def login():
    """Initiate the OAuth2 flow with Google."""
    flow = Flow.from_client_config(CLIENT_CONFIG, scopes=SCOPES)
    flow.redirect_uri = CLIENT_CONFIG["web"]["redirect_uris"][0]
    authorization_url, _ = flow.authorization_url(
        access_type="offline",  # ✅ Ensures a refresh token is granted
        include_granted_scopes="true",
        prompt="consent"  # ✅ Forces re-authentication to get a new refresh token
    )
    return RedirectResponse(authorization_url)

@app.get("/oauth2callback")
async def oauth2callback(request: Request):
    """
    Retrieve OAuth2 authorization response and ingest Gmail and Drive data.
    """
    flow = Flow.from_client_config(CLIENT_CONFIG, scopes=SCOPES)
    flow.redirect_uri = CLIENT_CONFIG["web"]["redirect_uris"][0]
    authorization_response = str(request.url)
    flow.fetch_token(authorization_response=authorization_response)
    credentials = flow.credentials

    # Ensure refresh_token is saved
    if not credentials.refresh_token:
        logging.error("Refresh token missing! Ensure you authorize with `access_type=offline`.")
        return {"error": "Refresh token is missing, please reauthorize."}

    # Save credentials to token.json
    with open("token.json", "w") as token_file:
        token_file.write(credentials.to_json())
    
    # Process Emails
    gmail_service = build("gmail", "v1", credentials=credentials)
    msg_list_response = gmail_service.users().messages().list(userId="me", maxResults=50).execute() #50 emails
    messages = msg_list_response.get("messages", [])
    for message in messages:
        msg_id = message["id"]
        email_data = gmail_service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        
        # Process email
        headers = email_data.get("payload", {}).get("headers", [])
        sender_info = next((h for h in headers if h.get("name") == "From"), None)
        sender_email = "unknown"
        sender_name = "unknown"
        if sender_info:
            sender_str = sender_info.get("value", "")
            match = re.match(r"^(.*)\s+<([^>]+)>$", sender_str)
            if match:
                sender_name = match.group(1).strip()
                sender_email = match.group(2)
            else:
                sender_email = sender_str.strip()
        
        email_content = extract_email_text(email_data)
        inferred_role = extract_role_from_content(email_content)
        update_knowledge_graph(sender_email, sender_name, inferred_role)
        
        # Save email data and chunks
        email_folder = os.path.join(EMAILS_BASE_DIR, msg_id)
        os.makedirs(email_folder, exist_ok=True)
        with open(os.path.join(email_folder, "email.json"), "w", encoding="utf-8") as f:
            json.dump(email_data, f, indent=2)
        combined_text = combine_email_text_and_metadata(email_data)
        chunks = local_text_splitter.split_text(combined_text)
        with open(os.path.join(email_folder, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

    # Process PDFs from Drive
    drive_service = build("drive", "v3", credentials=credentials)
    query_str = "mimeType='application/pdf'"
    collected = 0
    page_token = None
    while collected < 2:
        response = drive_service.files().list(
            q=query_str,
            spaces='drive',
            pageSize=10,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()
        for file in response.get("files", []):
            if collected >= 5: #50 PDFs
                break
            file_id = file["id"]
            file_name = file["name"]
            if not file_name.lower().endswith(".pdf"):
                file_name += ".pdf"
            request_media = drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request_media)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            drive_file_path = os.path.join(DRIVE_DIR, file_name)
            with open(drive_file_path, "wb") as f:
                f.write(fh.getvalue())
            collected += 1
        page_token = response.get("nextPageToken", None)
        if not page_token:
            break

    # Process PDFs: Extract text and split into chunks
    processed_data = {}
    for file in os.listdir(DRIVE_DIR):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(DRIVE_DIR, file)
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                full_text = "\n".join(doc.page_content for doc in docs)
                clean_text = clean_html(full_text)
                pdf_chunks = local_text_splitter.split_text(clean_text)
                processed_data[file] = pdf_chunks
            except Exception as e:
                logging.error(f"Error processing PDF {file}: {e}")
                continue
    with open(os.path.join(DATA_DIR, "processed_data.json"), "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2)
    await build_vector_store()
    return {"message": f"Data processed: {len(messages)} emails and {collected} drive files (PDFs processed into chunks)."}

@app.get("/build_store")
async def build_store_endpoint():
    """Build the vector store from ingested data."""
    response = await build_vector_store()
    return response

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
        logging.error(f"Error accessing knowledge graph: {e}")
        return "Error retrieving structured knowledge."


@app.get("/query")
async def query_endpoint(q: str):
    try:
        if not vector_store_ready.is_set():
            await build_vector_store()
        
        vectorstore_cache = Chroma(
            collection_name="documents",
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        retriever = vectorstore_cache.as_retriever(k=12) #can be changed to 10-100
        retrieved_docs = retriever.invoke(q)  # Ensure no 'await' here
        
        doc_context = "\n\n".join(
            [f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in retrieved_docs]
        )
        # Load structured knowledge from `knowledge_graph.json`
        knowledge_context = get_knowledge_graph_context()
        
        # Combine both sources into a single enriched context
        enriched_context = f"--- Knowledge Graph Data ---\n{knowledge_context}\n\n--- Retrieved Documents ---\n{doc_context}"
        
        if q.startswith("send email to"):
            recipient_query = re.sub(r"^\s*send email to\s+", "", q, flags=re.IGNORECASE).strip()
            recipient_query = re.split(r"mentioning|confirming|that|and", recipient_query, 1)[0].strip()  # Extract only name/role
            recipient_email = get_recipient_email(recipient_query)  # Look up by name or role
            if recipient_email:
                # Generate email content using LLM
                prompt = generate_email_prompt(recipient_query, q,enriched_context)
                response = llm.invoke(prompt)
                subject, body = parse_email_response(response)
                
                # Send the email
                if send_email(recipient_email, subject, body):
                    return {"query": q, "result": f"Email sent to {recipient_query} ({recipient_email})"}
                else:
                    return {"query": q, "result": "Failed to send email"}
            else:
                return {"query": q, "result": f"No email found for {({recipient_email})}"}
        
        # General query processing
        prompt = generate_query_prompt(q, enriched_context)
        response = llm.invoke(prompt)
        return {"query": q, "result": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        logging.info(f"Email sent successfully: {response}")

        return True
    except Exception as e:
        logging.error(f"Error sending email: {e}", exc_info=True)
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
        logging.info(f"Updated knowledge graph for {sender_email}: {role}")
    except Exception as e:
        logging.error(f"Error updating knowledge graph: {e}")

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


from itertools import combinations

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

        logging.info(f"Checking subqueries: {subqueries}")

        # Search for the best match in `knowledge_graph.json`
        for subquery in subqueries:
            for email, info in knowledge_graph.get("people", {}).items():
                name = info.get("name", "").lower().strip()
                role = info.get("role", "").lower().strip()

                if subquery == name or subquery == role:
                    logging.info(f"Match found: {name} ({email})")
                    return email  # Return the first best match

        logging.warning(f"No match found for: {query}")
        return None  # No match found

    except Exception as e:
        logging.error(f"Error accessing knowledge graph: {e}")
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
    from google.oauth2.credentials import Credentials
    if os.path.exists("token.json"):
        return build("gmail", "v1", credentials=Credentials.from_authorized_user_file("token.json"))
    else:
        raise Exception("No credentials found. Please log in first.")


@app.post("/feedback")
async def feedback_endpoint(request: Request):
    data = await request.json()
    try:
        with open(FEEDBACK_PATH, "r+") as f:
            feedback_data = json.load(f)
            feedback_data.append(data)
            f.seek(0)
            json.dump(feedback_data, f, indent=2)
        return {"message": "Feedback received"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Gmail API Utilities ---

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
        logging.error(f"Failed to send message: {e}")
        raise HTTPException(status_code=500, detail="Error sending email")
    
from google.oauth2.credentials import Credentials

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



if __name__ == "__main__":
    os.remove("token.json")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# THIS IS A WORKING FILE