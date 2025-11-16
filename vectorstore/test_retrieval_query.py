# Developer: Harshitha
# Description: Added for Manual Testing
# Added this script that tests retrieval by querying the vector index (RAG Step 3 validation).
# Shows top-k chunks with URLs and light, query-aware re-ranking (no hardcoded intents).

import chromadb
import math
import re
import yaml
from .rerank import rerank, is_program_requirements_query
from collections import Counter
from chromadb.utils import embedding_functions

CHROMA_DIR = "vectorstore/chroma"
COLLECTION = "msu_faq_chunks"

# Pull more ANN candidates so reranker has room to work
CANDIDATES = 200   # was 100

# ---------------- query feature detectors ----------------

DATEY = re.compile(
    r"\b(withdraw(al)?|deadline|last\s*day|add/?drop|calendar|final\s*day|wd)\b",
    re.I
)
TERM_WORD = re.compile(r"\b(Spring|Summer|Fall|Winter)\s+20\d{2}\b", re.I)
PASSFAIL_RX = re.compile(r"\bpass[-/\s]?fail\b", re.I)
LOA_RX = re.compile(r"\bleave\s+of\s+absence\b|\bloa\b", re.I)

# load cfg once
with open("sources.yaml","r",encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

def run_one_query(question, base_candidates):
    """
    base_candidates: list of objects/dicts with fields:
        url, text, section_heading, meta, embed_score, bm25, chunk_id
      (meta MUST include 'bucket' populated during ingestion.)
    """
    # re-rank (top 5)
    top5, intent = rerank(base_candidates, question, CFG, top_k=5)

    print(f"[diagnostic] intent={intent}  candidates={len(base_candidates)}")
    print(f"Q: {question}")
    print("Top 5 results (re-ranked):\n")
    for i, c in enumerate(top5, 1):
        meta = getattr(c,'meta', c.get('meta',{}))
        is_table = "True" if meta.get('table') else "False"
        term = meta.get('term',"")
        sec  = getattr(c,'section_heading', c.get('section_heading',''))
        url  = getattr(c,'url', c.get('url',''))
        base = getattr(c,'embed_score', c.get('embed_score',0.0))
        bm25 = getattr(c,'bm25', c.get('bm25',0.0))
        final = getattr(c,'final_score', c.get('final_score',0.0))
        print(f"{i}. bucket={meta.get('bucket','')} | base~{base:.3f} | bm25~{bm25:.3f} | final~{final:.3f} | table={is_table} | sec={sec} | term={term} | url={url}")
        # optionally: print(c.text[:220], "…")
    print()

def tokenize(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())

def kw_overlap_score(query: str, doc: str) -> float:
    q = tokenize(query)
    d = tokenize(doc[:2000])
    if not q or not d:
        return 0.0
    q_counter = Counter(q)
    d_counter = Counter(d)
    inter = sum(min(q_counter[w], d_counter.get(w, 0)) for w in q_counter)
    denom = math.sqrt(sum(q_counter.values()) * sum(d_counter.values()))
    return (inter / denom) if denom else 0.0

def query_features(query: str) -> dict:
    q = query.strip()
    return {
        "is_datey": bool(DATEY.search(q) or TERM_WORD.search(q)),
        "is_passfail": bool(PASSFAIL_RX.search(q)),
        "is_loa": bool(LOA_RX.search(q)),
    }

# --------------- Chroma connection ----------------

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION, embedding_function=embedding_fn)

# --------------- small utils ----------------

def normalize_headings(h):
    if h is None:
        return []
    if isinstance(h, str):
        return [h] if h else []
    if isinstance(h, (list, tuple)):
        return list(h)
    return [str(h)]

# --------------- main ask() ----------------

def ask(query: str, k: int = 5, bucket_filter: str | None = None):
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

    docs  = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None]*len(docs)])[0]

    cand_tables = sum(1 for m in metas if bool(m.get("is_table", False) or m.get("table", False)))
    print(f"[diagnostic] candidates={len(metas)}, table_candidates={cand_tables}")

    # Build candidate objects for central reranker
    candidates = []
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        embed_score = (1 - (dist or 0)) if dist is not None else 0.0
        bm25_like   = kw_overlap_score(query, doc)  # lightweight lexical score
        candidates.append({
            "embed_score": embed_score,
            "bm25": bm25_like,
            "url": meta.get("url") or meta.get("source_url", ""),
            "text": doc,
            "section_heading": meta.get("section_heading", ""),
            "meta": meta,                 # includes bucket, table/is_table, term, etc.
            "chunk_id": idx,              # stable tie-breaker
            "term_matches_query": False,  # reranker will set this flag
        })

    # Hand off to the unified reranker
    topk, intent = rerank(candidates, query, CFG, top_k=k)

    print(f"[diagnostic] intent={intent}")
    print(f"\nQ: {query}\nTop {k} results (re-ranked):\n")
    for i, c in enumerate(topk, 1):
        meta  = c.get("meta", {})
        url   = c.get("url", "")
        base  = c.get("embed_score", 0.0)
        bm25  = c.get("bm25", 0.0)
        final = c.get("final_score", 0.0)
        sec   = c.get("section_heading","")
        bucket= meta.get("bucket","")
        is_t  = bool(meta.get("table") or meta.get("is_table"))
        term  = meta.get("term","")

        print(f"{i}. bucket={bucket} | base~{base:.3f} | bm25~{bm25:.3f} | final~{final:.3f} | table={is_t} | sec={sec} | term={term} | url={url}")

        heads = normalize_headings(meta.get("headings"))
        if heads:
            print("   headings:", heads[:2])

        # Keep TERM/SECTION markers visible in snippet
        prefix_bits = []
        for ln in (c.get("text","").splitlines()[:8]):
            if ln.startswith("**TERM:**") or ln.startswith("**SECTION:**"):
                prefix_bits.append(ln)
        prefix = (" ".join(prefix_bits) + " ") if prefix_bits else ""
        
        #snippet = c.get("text","")[:600] + ("…" if len(c.get("text","")) > 600 else "")
        #print("   text:", (prefix + snippet).strip())
        #print()
        full_text = c.get("text","")
        print("   text:", (prefix + full_text).strip())
        print()


# manual tests
if __name__ == "__main__":
    ask("What is the course withdrawal deadline?", k=5)
    ask("what is the Last Day to Withdraw Classes for Winter 2025",k=5)
    ask("what is the Final Day to Withdraw course for summer 2026",k=5)
    ask("what is the semester end date of Winter 2026", k=5)
    ask("what are the Required Courses for MS in Computer Science",k=5)
    #ask("How does pass/fail grading work?", k=5)
    #ask("How do I request a leave of absence?", k=5)
    #ask("How can I contact academic advising?", k=5)
    #ask("What are the eligibility criteria for study abroad?", k=5)
    #ask("Last day of Winter 2025 Semester",k=5)