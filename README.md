# Legal Docs QA — EPF, TDS, Companies Act (Streamlit)

## What it does
Builds a retrieval QA demo over three sources:
- EPF Act 1952 (PDF)
- Income Tax TDS deposit page (HTML)
- Companies Act, 2013 (PDF)

It downloads sources, extracts text, chunks them, builds embeddings with `sentence-transformers` and indexes vectors with FAISS. A Streamlit UI allows querying and returns the most relevant passages.

## Run locally
1. Create venv & activate (Windows PowerShell):
   ```powershell
   python -m venv env
   .\env\Scripts\Activate.ps1
