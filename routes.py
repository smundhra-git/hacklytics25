from google_auth_oauthlib.flow import Flow
from fastapi import APIRouter
import os
from dotenv import load_dotenv
from gsuite_helper import *
from fastapi.responses import HTMLResponse, RedirectResponse
import re
import PyPDF2
import io
from googleapiclient.http import MediaIoBaseDownload
from langchain_community.document_loaders import PyPDFLoader
import os
import io
import json
import base64
import logging
from datetime import datetime
import uvicorn
import re

# --- Imports from LangChain, Mistral (via Ollama interface), and Google APIs ---
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama  # Using Mistral via Ollama interface
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings

import PyPDF2


load_dotenv()

router = APIRouter()


@router.get("/gmail/login")
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


@router.get("/gmail/oauth2callback")
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
                continue
    with open(os.path.join(DATA_DIR, "processed_data.json"), "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2)
    await build_vector_store()
    return {"message": f"Data processed: {len(messages)} emails and {collected} drive files (PDFs processed into chunks)."}

@router.get("/build_store")
async def build_store_endpoint():
    """Build the vector store from ingested data."""
    response = await build_vector_store()
    return response


@router.get("/query")
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

@router.post("/feedback")
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