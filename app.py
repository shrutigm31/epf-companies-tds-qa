# app.py
import streamlit as st
from pathlib import Path
import hashlib
import pickle
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import PyPDF2
import time

st.set_page_config(page_title="Legal Docs QA — EPF | TDS | Companies Act", layout="wide")

# --- DATA SOURCES: change if you want a different Companies Act PDF ---
DATA_SOURCES = {
    "EPF Act 1952 (PDF)": "https://www.epfindia.gov.in/site_docs/PDFs/Downloads_PDFs/EPFAct1952.pdf",
    "TDS Deposit (Income Tax site - HTML)": "https://incometaxindia.gov.in/Pages/Deposit_TDS_TCS.aspx",
    # A common public copy of Companies Act 2013 PDF — replace with official if you have another URL
    "Companies Act, 2013 (PDF)": "https://www.india.gov.in/sites/upload_files/npi/files/companies_act_2013.pdf"
}

CHUNK_SIZE = 800
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = Path("index_data")
INDEX_FAISS_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "meta.pkl"
EMB_PATH = INDEX_DIR / "embeddings.npy"

def download_file(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest

def extract_text_from_pdf(path: Path):
    texts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in range(len(reader.pages)):
            try:
                txt = reader.pages[p].extract_text() or ""
            except Exception:
                txt = ""
            texts.append(txt)
    return "\n".join(texts)

def extract_text_from_html_bytes(html_bytes):
    soup = BeautifulSoup(html_bytes, "html.parser")
    for s in soup(["script", "style", "header", "footer", "nav"]):
        s.decompose()
    return soup.get_text(separator="\n")

def chunk_text(text, size=CHUNK_SIZE):
    text = text.replace("\r", " ")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        # try to cut on newline or space
        if end < len(text):
            extra = text[end:end+200]
            cut = extra.find("\n")
            if cut == -1:
                cut = extra.find(" ")
            if cut != -1:
                end += cut
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end
    return chunks

def build_index():
    INDEX_DIR.mkdir(exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL)
    all_texts = []
    metadata = []

    st.info("Downloading and processing source documents (this may take a few minutes locally)...")
    for title, url in DATA_SOURCES.items():
        st.write("Fetching:", title)
        fname = INDEX_DIR / (hashlib.sha1(url.encode()).hexdigest() + Path(url).suffix)
        try:
            download_file(url, fname)
        except Exception as e:
            st.error(f"Failed to download {url}: {e}")
            continue
        if str(fname).lower().endswith(".pdf"):
            text = extract_text_from_pdf(fname)
        else:
            text = extract_text_from_html_bytes(open(fname, "rb").read())
        if not text:
            continue
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_texts.append(c)
            metadata.append({"source": title, "source_url": url, "chunk_id": i})

    if not all_texts:
        st.error("No text extracted from the sources. Please check internet or source URLs.")
        st.stop()

    st.write(f"Total chunks: {len(all_texts)} — creating embeddings...")
    embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, str(INDEX_FAISS_PATH))
    np.save(str(EMB_PATH), embeddings)
    with open(META_PATH, "wb") as f:
        pickle.dump({"texts": all_texts, "metadata": metadata}, f)

    st.success("Index built and saved to index_data/")
    return model, index, {"texts": all_texts, "metadata": metadata}, embeddings

@st.cache_resource
def get_model_and_index():
    # Load if exists else build
    try:
        model = SentenceTransformer(EMBED_MODEL)
        if INDEX_FAISS_PATH.exists() and EMB_PATH.exists() and META_PATH.exists():
            index = faiss.read_index(str(INDEX_FAISS_PATH))
            embeddings = np.load(str(EMB_PATH))
            meta = pickle.load(open(META_PATH, "rb"))
            return model, index, meta, embeddings
    except Exception:
        pass
    return build_index()

st.title("Legal Documents Retrieval QA — EPF | TDS | Companies Act")
st.markdown("A demo retrieval-based QA over the three provided sources. This demo **builds a semantic index** from the sources and returns the most relevant passages for your query.")

model, index, meta, embeddings = get_model_and_index()

query = st.text_input("Ask a question about EPF / TDS / Companies Act (try: 'what is contribution under EPF?' )")
top_k = st.slider("Number of passages to return", 1, 10, 3)

if st.button("Search") and query.strip():
    q_emb = model.encode([query], convert_to_numpy=True)
    # ensure float32
    q_emb = np.array(q_emb, dtype="float32")
    D, I = index.search(q_emb, top_k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        hits.append((score, idx, meta["texts"][idx], meta["metadata"][idx]))

    st.write("### Results")
    for i, (score, idx, txt, m) in enumerate(hits, start=1):
        st.write(f"**Result {i} — source:** {m['source']} (chunk {m['chunk_id']}) — score: {score:.4f}")
        st.write(txt[:2000] + ("..." if len(txt) > 2000 else ""))
    st.write("---")
    st.write("### Combined context (useful for manual answer writing or paste to an LLM)")
    combined = "\n\n---\n\n".join([h[2] for h in hits])
    st.code(combined[:3900] + ("..." if len(combined) > 3900 else ""), language="text")
    st.info("This demo returns relevant passages rather than a generated final answer. For a generated answer step, you may connect an LLM (OpenAI/Local) and prompt it with the combined context.")
