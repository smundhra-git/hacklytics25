import base64
import json
import os
import jwt
import requests
from datetime import datetime, timedelta
import re

# Configuration
INTEGRATION_KEY = "4a7751f7-ad67-4887-90d5-df13303cfa43"
PRIVATE_KEY_PATH = "private.key"  # Path to your private key file
USER_ID = "249b8f1d-3f83-4eab-8be2-569b7dca8272"   # DocuSign user ID
BASE_URI = "https://demo.docusign.net/restapi"  # Use "https://www.docusign.net/restapi" for production
AUTHENTICATION_URI = f"{BASE_URI}/oauth/token"
ENVELOPE_CREATE_URI = f"{BASE_URI}/v2.1/accounts/33395373/envelopes"

def get_jwt_token():
    private_key = open(PRIVATE_KEY_PATH, 'r').read().strip()
    now = datetime.utcnow()
    claims = {
        "iss": INTEGRATION_KEY,
        "sub": USER_ID,
        "aud": "account-d.docusign.com",
        "iat": now,
        "exp": now + timedelta(seconds=3600),
        "scope": "signature impersonation",
    }
    assertion = jwt.encode(claims, private_key, algorithm="RS256")
    response = requests.post(
        "https://account-d.docusign.com/oauth/token",
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion,
        }
    )
    if response.status_code != 200:
        raise Exception(f"Authentication error: {response.status_code} - {response.json()}")
    return response.json()["access_token"]

def extract_signer_names(query, lowercase_knowledge_graph):
    query = query.lower()
    tokens = re.findall(r'\b\w+\b', query)
    ngrams = []
    n = len(tokens)
    for i in range(n):
        for j in range(i+1, n+1):
            ngram = ' '.join(tokens[i:j])
            ngrams.append(ngram)
    unique_ngrams = list(set(ngrams))
    matched_original_names = []
    for ngram in unique_ngrams:
        if ngram in lowercase_knowledge_graph:
            original_name = lowercase_knowledge_graph[ngram]["name"]
            matched_original_names.append(original_name)
    return list(set(matched_original_names))

def send_document(access_token, signers, document_path):
    with open(document_path, 'rb') as file:
        content_bytes = file.read()
        base64_doc = base64.b64encode(content_bytes).decode('utf-8')
    
    envelope = {
        "emailSubject": "Document for Signature",
        "documents": [{
            "documentId": "1",
            "name": os.path.basename(document_path),
            "documentBase64": base64_doc
        }],
        "recipients": {
            "signers": []
        },
        "status": "sent"
    }

    for idx, signer in enumerate(signers, start=1):
        envelope["recipients"]["signers"].append({
            "email": signer["email"],
            "name": signer["name"],
            "recipientId": str(idx),
            "routingOrder": str(idx),
        })

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(ENVELOPE_CREATE_URI, headers=headers, json=envelope)
    print(f"API Response: {response.status_code} - {response.json()}")
    return response.json()

def main():
    # Prompt for document path
    document_path = input("Enter document path: ").strip()
    
    # Prompt for query
    query = input("Enter your request: ").strip()

    # Load knowledge graph
    try:
        with open("knowledge_graph.json", "r") as f:
            knowledge_graph = json.load(f)
    except FileNotFoundError:
        print("Error: knowledge_graph.json not found.")
        return

    # Preprocess the knowledge graph into lowercase keys
    lowercase_knowledge_graph = {}
    for name in knowledge_graph:
        lowercase_name = name.strip().lower()
        lowercase_knowledge_graph[lowercase_name] = {
            "name": name,
            "email": knowledge_graph[name]
        }

    # Extract original names from the query
    matched_names = extract_signer_names(query, lowercase_knowledge_graph)

    if not matched_names:
        print("No valid names found in the query.")
        return

    # Get emails for matched names
    signers = []
    for original_name in matched_names:
        email = knowledge_graph.get(original_name)
        if email:
            signers.append({
                "name": original_name,
                "email": email
            })
        else:
            print(f"Email not found for {original_name}")

    if not signers:
        print("No valid signers found.")
        return

    # Get access token
    try:
        access_token = get_jwt_token()
    except Exception as e:
        print(f"Authentication Error: {e}")
        return

    # Send document to all signers
    try:
        send_document(access_token, signers, document_path)
        print(f"Document sent to {len(signers)} recipients.")
    except Exception as e:
        print(f"Failed to send document: {e}")

if __name__ == "__main__":
    main()