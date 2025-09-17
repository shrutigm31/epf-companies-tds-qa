# Legal Docs QA demo — EPF, TDS, Companies Act (Streamlit)

## What this project does
This project builds a simple retrieval QA demo over three sources you provided:
- Employees' Provident Funds and Miscellaneous Provisions Act, 1952 (EPF Act) — PDF  
- Income Tax Department page on Deposit of TDS/TCS — HTML page  
- Companies Act, 2013 — PDF

It downloads the sources, extracts text, chunks the text, creates embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`), indexes the vectors with FAISS, and exposes a Streamlit UI to search queries and see the most relevant passages.

## How to run locally (recommended)
1. Create a Python 3.10+ virtual environment.
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. On first run the app downloads the three sources and builds the index (may take a few minutes).

## How to deploy
- You can deploy on Render / Railway / Fly.io / a VPS. Ensure the instance can install `faiss-cpu` and `sentence-transformers`.
- Set a process to run `streamlit run app.py --server.port $PORT --server.enableCORS false` depending on host.

## Notes & limitations
- This demo performs **retrieval** (returns passages) — it does not produce a synthesized final answer. For full QA generation, connect an LLM and prompt with the combined context.
- The app downloads the public documents at runtime. If you prefer to bundle the files, place them under `index_data/`.
- For production use, add rate-limits, better chunking, metadata handling, and an LLM with guardrails.

## Files included
- app.py — Streamlit application
- requirements.txt — Python dependencies
- README.md — this file
