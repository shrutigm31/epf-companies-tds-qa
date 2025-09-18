# check_index.py
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

INDEX_FAISS_PATH = "index_data/index.faiss"
META_PATH = "index_data/meta.pkl"

def main():
    if not os.path.exists(META_PATH) or not os.path.exists(INDEX_FAISS_PATH):
        print("Index files missing. Run the Streamlit app once so it builds the index.")
        return

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_FAISS_PATH)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    print(f"✅ Index contains {index.ntotal} vectors")
    texts = meta.get("texts") if isinstance(meta, dict) else meta
    print(f"✅ Metadata chunks: {len(texts)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    q = "What is contribution under EPF?"
    qv = model.encode([q])
    D, I = index.search(np.array(qv, dtype="float32"), k=3)
    print("\n🔎 Query:", q)
    for rank, idx in enumerate(I[0]):
        print(f"\nResult {rank+1} (source metadata):")
        try:
            print(meta["metadata"][idx])
            print("Text excerpt:")
            print(meta["texts"][idx][:500])
        except Exception:
            print("Error reading metadata/text for idx", idx)

if __name__ == "__main__":
    main()
