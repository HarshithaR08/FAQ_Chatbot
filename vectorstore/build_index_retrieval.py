# Developer: Harshitha, Lam
# Description: takes rag/chunks.jsonl (each line = one chunk with text + metadata),turn the text into vectors (embeddings), 
# and store them in a persistent vector database (Chroma) so we can search later.
# Added this to build the Chroma vector index for retrieval (RAG Step 3).
 
import os, json, logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


# added for debug logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

# Configuration path for files
CHUNKS_JSONL = "rag/chunks.jsonl"          # created by your chunker
CHROMA_DIR   = "vectorstore/chroma"        # chroma persistence directory
COLLECTION   = "msu_faq_chunks"            # collection name

# Embeddings: small, fast model
# sentence-transformers/all-MiniLM-L6-v2 returns 384-d vectors
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# helper functions

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def norm_str(x):
    if x is None: return ""
    if isinstance(x, (str, int, float, bool)): return str(x)
    if isinstance(x, list): return ", ".join(map(str, x))          # <-- key fix for headings
    return str(x)

def normalize_meta(ch: dict) -> dict:
    return {
        "url":      norm_str(ch.get("source_url", "")),
        "bucket":   norm_str(ch.get("bucket", "")),
        "headings": norm_str(ch.get("headings", [])),
        "title":    norm_str(ch.get("title", "")),
        "is_table": bool(ch.get("is_table", False)),
        "table":    bool(ch.get("is_table", False)),
        "section":  norm_str(ch.get("section", "")),
        "term":     norm_str(ch.get("term", "")),
        "section_heading": norm_str(ch.get("section_heading", "")),
        "has_deadline_words": bool(ch.get("has_deadline_words", False)),
    }


def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # fresh rebuild
    try:
        client.delete_collection(name=COLLECTION)
    except Exception:
        pass

    coll = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    ids, docs, metas, seen = [], [], [], set()
    BATCH = 500
    total = 0

    for ch in load_chunks(CHUNKS_JSONL):
        cid = ch["id"]
        if cid in seen:
            continue
        seen.add(cid)

        ids.append(cid)
        docs.append(ch["text"])
        metas.append(normalize_meta(ch))

        if len(ids) >= BATCH:
            coll.add(ids=ids, documents=docs, metadatas=metas)
            total += len(ids)
            logging.info(f"Indexed {total} chunks...")
            ids, docs, metas = [], [], []

    if ids:
        coll.add(ids=ids, documents=docs, metadatas=metas)
        total += len(ids)

    logging.info(f" Done. Total chunks indexed: {total}")

if __name__ == "__main__":
    main()