# Description: takes rag/chunks.jsonl (each line = one chunk with text + metadata),turn the text into vectors (embeddings), 
# and store them in a persistent vector database (Chroma) so we can search later.
# Added this to build the Chroma vector index for retrieval (RAG Step 3).
 
import os, json, logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


# added for debug logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)

# Configuration path for files
CHUNKS_JSONL = "rag/chunks.jsonl"          # created by your chunker
CHROMA_DIR   = "vectorstore/chroma"        # chroma will persist here
COLLECTION   = "msu_faq_chunks"            # name your collection

# Embeddings: small, fast model
# sentence-transformers/all-MiniLM-L6-v2 returns 384-d vectors
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_chunks(path: str):
    """Yield chunk dicts from JSONL. Each line is one chunk with text+metadata."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    # to create (or open) a persistent Chroma client on disk
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # for fresh rebuild: drop the whole collection if it exists.
    #    NOTE: new Chroma requires delete_collection(name=...)
    try:
        client.delete_collection(name=COLLECTION)
        logging.info("Dropped existing collection for a clean rebuild.")
    except Exception:
        # If it didn't exist, that's fine.
        pass

    # Create the collection again (empty), attach our embedding function.
    coll = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  
    )

    # Read chunks and add in batches.
    #    We'll also guard against duplicate IDs *within this run* (JSONL duplicates).
    ids, texts, metadatas = [], [], []
    BATCH = 500
    total = 0
    seen_ids = set()   # protect against duplicates inside chunks.jsonl
    dup_skips = 0

    # Optional sanity counts (not required):
    total_lines = sum(1 for _ in open(CHUNKS_JSONL, "r", encoding="utf-8"))
    logging.info(f"Reading {total_lines} lines from {CHUNKS_JSONL}")

    for ch in load_chunks(CHUNKS_JSONL):
        cid = ch["id"]
        if cid in seen_ids:
            # duplicate inside the same JSONL -> skip it
            dup_skips += 1
            continue
        seen_ids.add(cid)

        ids.append(cid)                   # unique id (sha1 from the chunker)
        texts.append(ch["text"])          # the content to embed
        metadatas.append({
            "url": ch.get("source_url",""),
            "bucket": ch.get("bucket",""),
            "headings": ", ".join(ch.get("headings", [])) if isinstance(ch.get("headings"), list) else str(ch.get("headings")),
            "title": ch.get("title","")
        })

        # Write in batches to control memory and show progress
        if len(ids) >= BATCH:
            # Use add() for full rebuild. If you ever do incremental runs, prefer upsert().
            coll.add(ids=ids, documents=texts, metadatas=metadatas)
            total += len(ids)
            logging.info(f"Indexed {total} chunks so far...")
            ids, texts, metadatas = [], [], []

    # flush the tail
    if ids:
        coll.add(ids=ids, documents=texts, metadatas=metadatas)
        total += len(ids)

    logging.info(f"✅ Done. Total chunks indexed: {total} (skipped duplicates in JSONL: {dup_skips})")
    logging.info(f"Chroma path: {os.path.abspath(CHROMA_DIR)} | collection: {COLLECTION}")

if __name__ == "__main__":
    main()
