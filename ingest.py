"""
ingest.py — Run this once to load your docs into the vector store.

Usage:
    1. Drop PDF or TXT files into data/docs/
    2. python ingest.py
"""

import os, uuid
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DOCS_FOLDER = "data/docs"
CHROMA_PATH = "data/chroma_db"
CHUNK_SIZE   = 1000   # characters per chunk
CHUNK_OVERLAP = 150

def read_file(path):
    if path.endswith(".pdf"):
        pages = []
        with open(path, "rb") as f:
            for page in PyPDF2.PdfReader(f).pages:
                t = page.extract_text()
                if t: pages.append(t)
        return "\n".join(pages)
    return open(path, encoding="utf-8").read()

def chunk_text(text, title, filename):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append({"text": text[start:end], "title": title, "file": filename})
        if end == len(text): break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def main():
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try: client.delete_collection("knowledge_base")
    except: pass
    col = client.create_collection("knowledge_base", metadata={"hnsw:space": "cosine"})

    all_chunks, n_docs = [], 0
    for fname in sorted(os.listdir(DOCS_FOLDER)):
        fpath = os.path.join(DOCS_FOLDER, fname)
        if not fname.lower().endswith((".pdf", ".txt")): continue
        try:
            text  = read_file(fpath)
            title = fname.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
            all_chunks += chunk_text(text, title, fname)
            n_docs += 1
            print(f"  loaded: {fname}")
        except Exception as e:
            print(f"  SKIP {fname}: {e}")

    if not all_chunks:
        print("No documents found. Add PDFs/TXTs to data/docs/ and re-run.")
        return

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    col.add(
        ids=[str(uuid.uuid4()) for _ in all_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"title": c["title"], "file": c["file"]} for c in all_chunks],
    )
    print(f"\nDone: {n_docs} docs, {len(all_chunks)} chunks stored.")

if __name__ == "__main__":
    main()
