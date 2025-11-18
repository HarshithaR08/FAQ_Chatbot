# Developer: Harshitha, Lam
# Description:
#   - Takes rag/chunks.jsonl (each line = one chunk with text + metadata),
#     turns the text into vectors (embeddings), and stores them in a
#     persistent vector database (Chroma) so we can search later.
#   - Added get_retriever() so other modules (RAG CLI, tests) can query
#     the existing Chroma index and get candidates for reranking.
#
#   This file now has TWO roles:
#     1) When run as a script: build / rebuild the Chroma index.
#     2) When imported: provide get_retriever() to perform semantic retrieval.

import os
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Callable

import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# ---------------------------------------------------------------------
# Paths / Config
# ---------------------------------------------------------------------

# Created by rag/chunker.py
CHUNKS_JSONL = "rag/chunks.jsonl"

# Chroma persistence directory
CHROMA_DIR = "vectorstore/chroma"

# Chroma collection name
COLLECTION = "msu_faq_chunks"

# Embedding model (same as everywhere else)
# sentence-transformers/all-MiniLM-L6-v2 → 384-dim vectors
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Lazy global for retrieval so we don't reconnect on every query
_CHROMA_COLLECTION = None

# ---------------------------------------------------------------------
# Helpers for indexing
# ---------------------------------------------------------------------


def load_chunks(path: str):
    """Stream JSONL chunks from rag/chunks.jsonl."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def norm_str(x):
    """Normalize various metadata types into a clean string."""
    if x is None:
        return ""
    if isinstance(x, (str, int, float, bool)):
        return str(x)
    if isinstance(x, list):
        # key fix for headings: join list into a readable string
        return ", ".join(map(str, x))
    return str(x)


def normalize_meta(ch: dict) -> dict:
    """Normalize metadata fields for storage in Chroma."""
    return {
        "url": norm_str(ch.get("source_url", "")),
        "bucket": norm_str(ch.get("bucket", "")),
        "headings": norm_str(ch.get("headings", [])),
        "title": norm_str(ch.get("title", "")),
        "is_table": bool(ch.get("is_table", False)),
        "table": bool(ch.get("is_table", False)),
        "section": norm_str(ch.get("section", "")),
        "term": norm_str(ch.get("term", "")),
        "section_heading": norm_str(ch.get("section_heading", "")),
        "has_deadline_words": bool(ch.get("has_deadline_words", False)),
    }


# ---------------------------------------------------------------------
# Index building (what you already had)
# ---------------------------------------------------------------------


def main():
    """
    Build a fresh Chroma index from rag/chunks.jsonl.

    Run manually with:
        python -m vectorstore.build_index_retrieval
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh rebuild: try delete; ignore if it doesn't exist
    try:
        client.delete_collection(name=COLLECTION)
    except Exception:
        pass

    coll = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
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

    logging.info(f"Done. Total chunks indexed: {total}")


# ---------------------------------------------------------------------
# Retrieval helpers (NEW) – used by RAG CLI + regression tests
# ---------------------------------------------------------------------


def _tokenize_for_overlap(s: str) -> List[str]:
    """Simple tokenizer for lexical overlap scoring."""
    return re.findall(r"[A-Za-z0-9]+", s.lower())


def _kw_overlap_score(query: str, doc: str) -> float:
    """
    Lightweight lexical similarity score.

    Rough cosine-like measure on word counts, used as a BM25-ish
    feature alongside the embedding score.
    """
    q = _tokenize_for_overlap(query)
    d = _tokenize_for_overlap(doc[:2000])
    if not q or not d:
        return 0.0

    q_counter = Counter(q)
    d_counter = Counter(d)

    inter = sum(min(q_counter[w], d_counter.get(w, 0)) for w in q_counter)
    denom = math.sqrt(sum(q_counter.values()) * sum(d_counter.values()))
    return (inter / denom) if denom else 0.0


def _get_chroma_collection():
    """
    Lazily connect to the existing Chroma collection and cache it.

    This is used only for retrieval (NOT for building the index).
    """
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is not None:
        return _CHROMA_COLLECTION

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_collection(
        name=COLLECTION,
        embedding_function=embedding_fn,
    )
    _CHROMA_COLLECTION = coll
    return coll


def get_retriever() -> Callable[[str, int], List[Dict[str, Any]]]:
    """
    Return a callable retriever(query: str, n_results: int = 50) -> List[dict].

    Each dict is compatible with vectorstore.rerank.rerank, i.e.:

        {
            "embed_score": float,
            "bm25": float,
            "url": str,
            "text": str,
            "section_heading": str,
            "meta": dict,        # includes: bucket, table/is_table, term, etc.
            "chunk_id": int,
            "term_matches_query": bool,
        }

    This is what ui.rag_cli and vectorstore/tests/regression_testing_retrieval.py import.
    """
    coll = _get_chroma_collection()

    def _retriever(query: str, n_results: int = 50) -> List[Dict[str, Any]]:
        res = coll.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None] * len(docs)])[0]

        candidates: List[Dict[str, Any]] = []
        for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            embed_score = (1 - (dist or 0.0)) if dist is not None else 0.0
            bm25_like = _kw_overlap_score(query, doc)

            candidates.append(
                {
                    "embed_score": embed_score,
                    "bm25": bm25_like,
                    "url": meta.get("url") or meta.get("source_url", ""),
                    "text": doc,
                    "section_heading": meta.get("section_heading", ""),
                    "meta": meta,
                    "chunk_id": idx,
                    "term_matches_query": False,
                }
            )

        return candidates

    return _retriever


if __name__ == "__main__":
    main()
