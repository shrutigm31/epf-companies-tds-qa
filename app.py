
import streamlit as st
import os, requests, tempfile, sys
from pathlib import Path
from urllib.parse import urlparse
import PyPDF2
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import hashlib
import time

st.set_page_config(page_title="Legal Docs QA — EPF | TDS | Companies Act", layout="wide")

DATA_SOURCES = {
    "EPF Act 1952 (PDF)": "https://www.epfindia.gov.in/site_docs/PDFs/Downloads_PDFs/EPFAct1952.pdf",
    "TDS Deposit (Income Tax site - HTML)": "https://incometaxindia.gov.in/Pages/Deposit_TDS_TCS.aspx",
    "Companies Act, 2013 (PDF)": "https://www.indiacode.nic.in/bitstream/123456789/2114/5/A2013-18.pdf"
}

CHUNK_SIZE = 800  # approx characters
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = Path("index_data")

def download_file(url, dest):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)

def extract_text_from_pdf(path):
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in range(len(reader.pages)):
            try:
                text.append(reader.pages[p].extract_text() or "")
            except Exception:
                continue
    return "\n".join(text)

def extract_text_from_html(html_bytes):
    soup = BeautifulSoup(html_bytes, "html.parser")
    # remove scripts/styles
    for s in soup(["script", "style"]):
        s.decompose()
    return soup.get_text(separator="\n")

def chunk_text(text, size=CHUNK_SIZE):
    text = text.replace("\r"," ")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        # try to cut at newline/space for readability
        snippet = text[start:end]
        if end < len(text):
            # extend to next space/newline to avoid splitting words
            extra = text[end:end+100]
            cut = extra.find("\n")
            if cut == -1:
                cut = extra.find(" ")
            if cut != -1:
                end += cut
                snippet = text[start:end]
        chunks.append(snippet.strip())
        start = end
    return [c for c in chunks if len(c) > 50]

def build_or_load_index():
    INDEX_DIR.mkdir(exist_ok=True)
    meta_path = INDEX_DIR / "meta.pkl"
    idx_path = INDEX_DIR / "faiss.index"
    emb_path = INDEX_DIR / "embeddings.npy"
    model = SentenceTransformer(EMBED_MODEL)
    if meta_path.exists() and idx_path.exists() and emb_path.exists():
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        index = faiss.read_index(str(idx_path))
        embeddings = np.load(str(emb_path))
        return model, index, metadata, embeddings
    # else build index
    st.info("Downloading and processing source documents (this may take a few minutes locally)...")
    all_texts = []
    metadata = []
    for title, url in DATA_SOURCES.items():
        st.write("Fetching:", title)
        fname = INDEX_DIR / (hashlib.sha1(url.encode()).hexdigest() + Path(url).suffix)
        try:
            if not fname.exists():
                download_file(url, fname)
        except Exception as e:
            st.error(f"Failed to download {url}: {e}")
            continue
        if str(fname).lower().endswith(".pdf"):
            text = extract_text_from_pdf(fname)
        else:
            text = extract_text_from_html(open(fname, "rb").read())
        if not text:
            continue
        chunks = chunk_text(text)
        for i,ch in enumerate(chunks):
            all_texts.append(ch)
            metadata.append({"source": title, "source_url": url, "chunk_id": i})
    if not all_texts:
        st.error("No texts found from sources; index cannot be built.")
        st.stop()
    st.write(f"Total chunks: {len(all_texts)} — creating embeddings...")
    embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    # persist
    faiss.write_index(index, str(idx_path))
    np.save(str(emb_path), embeddings)
    with open(meta_path, "wb") as f:
        pickle.dump({"texts": all_texts, "metadata": metadata}, f)
    st.success("Index built and saved to index_data/")
    return model, index, {"texts": all_texts, "metadata": metadata}, embeddings

@st.cache_resource
def get_model_and_index():
    return build_or_load_index()

st.title("Legal Documents Retrieval QA — EPF | TDS | Companies Act")
st.markdown("A demo retrieval-based QA over the three provided sources. This demo **builds a semantic index** from the sources and returns the most relevant passages for your query.")

model, index, meta, embeddings = get_model_and_index()

query = st.text_input("Ask a question about EPF / TDS / Companies Act (try: 'what is contribution under EPF?' )")
top_k = st.slider("Number of passages to return", 1, 10, 3)

if st.button("Search") and query.strip():
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        hits.append((score, idx, meta["texts"][idx], meta["metadata"][idx]))
    st.write("### Results")
    for i, (score, idx, txt, m) in enumerate(hits, start=1):
        st.write(f"**Result {i} — source:** {m['source']} (chunk {m['chunk_id']}) — score: {score:.4f}")
        st.write(txt[:2000] + ("..." if len(txt)>2000 else ""))
    st.write("---")
    st.write("### Combined context (useful for manual answer writing or paste to an LLM)")
    combined = "\n\n---\n\n".join([h[2] for h in hits])
    st.code(combined[:3900] + ("..." if len(combined)>3900 else ""), language="text")
    st.info("This demo returns relevant passages rather than a generated final answer. For a generated answer step, you may connect an LLM (OpenAI/Local) and prompt it with the combined context.")
