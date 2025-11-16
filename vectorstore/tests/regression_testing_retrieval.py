"""
Regression tests for MSU FAQ retrieval + reranker.

This is separate from `vectorstore/test_retrieval_query.py`, which is for
manual, human-readable diagnostics.

Run via:

    cd faq_chatbot
    python -m vectorstore.tests.regression_testing_retrieval
"""

import sys
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions
import yaml

from vectorstore.rerank import rerank, detect_intent, is_program_requirements_query

# --------- shared config / paths (same style as your manual script) ---------

CHROMA_DIR = "vectorstore/chroma"
COLLECTION = "msu_faq_chunks"
CANDIDATES = 200  # how many ANN candidates to pull before reranking

# Load the same config you use everywhere else
with open("sources.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# --------- small helpers copied from manual script logic ---------


def tokenize(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


def kw_overlap_score(query: str, doc: str) -> float:
    """
    Lightweight lexical score to mimic BM25-ish behavior.
    Same idea as in your manual script.
    """
    q = tokenize(query)
    d = tokenize(doc[:2000])
    if not q or not d:
        return 0.0
    q_counter = Counter(q)
    d_counter = Counter(d)
    inter = sum(min(q_counter[w], d_counter.get(w, 0)) for w in q_counter)
    denom = math.sqrt(sum(q_counter.values()) * sum(d_counter.values()))
    return (inter / denom) if denom else 0.0


embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION, embedding_function=embedding_fn)


def build_candidates(query: str, k: int = 5, bucket_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    This is the 'pipeline' hook for regression tests.

    It reuses the SAME retrieval behavior as your manual `ask()` function,
    but returns candidates instead of printing.
    """
    coll = get_collection()

    # If this looks like “MS program requirements”, default to catalog-programs
    if bucket_filter is None and is_program_requirements_query(query):
        bucket_filter = "catalog-programs"

    where = {"bucket": bucket_filter} if bucket_filter else None

    res = coll.query(
        query_texts=[query],
        n_results=max(k, CANDIDATES),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None] * len(docs)])[0]

    candidates: List[Dict[str, Any]] = []
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        embed_score = (1 - (dist or 0)) if dist is not None else 0.0
        bm25_like = kw_overlap_score(query, doc)

        candidates.append(
            {
                "embed_score": embed_score,
                "bm25": bm25_like,
                "url": meta.get("url") or meta.get("source_url", ""),
                "text": doc,
                "section_heading": meta.get("section_heading", ""),
                "meta": meta,  # includes bucket, term, table/is_table, etc.
                "chunk_id": idx,
                "term_matches_query": False,  # reranker will update this
            }
        )

    return candidates


def _candidate_bucket(c: Dict[str, Any]) -> str:
    meta = c.get("meta", {}) or {}
    return meta.get("bucket", "")


def _candidate_url(c: Dict[str, Any]) -> str:
    return c.get("url", "") or ""


# --------- regression cases (the behaviors you just verified manually) ---------


@dataclass
class RegressionCase:
    name: str
    query: str
    expect_intent: Optional[str] = None  # "deadline", "calendar", "default", etc.
    expect_top_url_contains: Optional[str] = None
    expect_top_bucket: Optional[str] = None
    allow_top_k: int = 1  # how many top positions to search for a match


REGRESSION_CASES: List[RegressionCase] = [
    RegressionCase(
        name="withdrawal_deadline_generic",
        query="What is the course withdrawal deadline?",
        expect_intent="deadline",
        expect_top_url_contains="red-hawk-central/registrar/add-drop/",
        expect_top_bucket="registrar",
        allow_top_k=3,
    ),
    RegressionCase(
        name="withdrawal_deadline_winter_2025",
        query="what is the Last Day to Withdraw Classes for Winter 2025",
        expect_intent="deadline",
        expect_top_url_contains="red-hawk-central/registrar/add-drop/",
        expect_top_bucket="registrar",
        allow_top_k=1,
    ),
    RegressionCase(
        name="semester_end_winter_2026",
        query="what is the semester end date of Winter 2026",
        expect_intent="calendar",
        expect_top_url_contains="academics/academic-calendar/academic-calendar-2025-2026/",
        expect_top_bucket="academics-programs",
        allow_top_k=1,
    ),
    RegressionCase(
        name="ms_cs_required_courses",
        query="what are the Required Courses for MS in Computer Science",
        expect_intent=None,  # your detect_intent currently returns "default" here
        expect_top_url_contains="catalog.montclair.edu/programs/computer-science-ms/#requirementstext",
        expect_top_bucket="catalog-programs",
        allow_top_k=1,
    ),
]


# --------- core runner ---------


def run_single_case(case: RegressionCase) -> bool:
    print(f"\n[case] {case.name}")
    print(f"  query: {case.query!r}")

    # 1) Check intent (if expected)
    detected_intent = detect_intent(case.query, CFG)
    print(f"  intent_detected: {detected_intent!r}")

    if case.expect_intent is not None and detected_intent != case.expect_intent:
        print(f"  ❌ intent mismatch: expected {case.expect_intent!r}, got {detected_intent!r}")
        return False

    # 2) Build candidates and rerank with your central reranker
    try:
        candidates = build_candidates(case.query, k=max(5, case.allow_top_k))
    except Exception as e:
        print(f"  ❌ error building candidates: {e}")
        return False

    if not candidates:
        print("  ❌ no candidates returned from Chroma")
        return False

    topk, reranked_intent = rerank(candidates, case.query, CFG, top_k=max(5, case.allow_top_k))
    print(f"  reranked_intent: {reranked_intent!r}")

    ok = True
    matched_url = False
    matched_bucket = False

    for rank, c in enumerate(topk[: case.allow_top_k], start=1):
        url = _candidate_url(c)
        bucket = _candidate_bucket(c)
        print(f"    rank={rank} | bucket={bucket} | url={url}")

        if case.expect_top_url_contains and case.expect_top_url_contains in url:
            matched_url = True
        if case.expect_top_bucket and case.expect_top_bucket == bucket:
            matched_bucket = True

    if case.expect_top_url_contains and not matched_url:
        print(
            f"  ❌ none of top-{case.allow_top_k} candidates had URL containing "
            f"{case.expect_top_url_contains!r}"
        )
        ok = False

    if case.expect_top_bucket and not matched_bucket:
        print(
            f"  ❌ none of top-{case.allow_top_k} candidates had bucket "
            f"{case.expect_top_bucket!r}"
        )
        ok = False

    if ok:
        print("  ✅ PASS")
    return ok


def run_all_regression_cases() -> bool:
    print("=== MSU FAQ Retrieval Regression Suite ===")
    all_ok = True
    for case in REGRESSION_CASES:
        if not run_single_case(case):
            all_ok = False
    return all_ok


# --------- pytest-style tests (optional) ---------


def test_withdrawal_deadline_generic():
    assert run_single_case(REGRESSION_CASES[0])


def test_withdrawal_deadline_winter_2025():
    assert run_single_case(REGRESSION_CASES[1])


def test_semester_end_winter_2026():
    assert run_single_case(REGRESSION_CASES[2])


def test_ms_cs_required_courses():
    assert run_single_case(REGRESSION_CASES[3])


# --------- CLI entrypoint ---------

if __name__ == "__main__":
    ok = run_all_regression_cases()
    sys.exit(0 if ok else 1)