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
# Alias used by other modules (RAG CLI, regression tests, etc.)
RERANK_CONFIG = CFG

# Reuse the same term regex as rerank.py uses (Fall 2025, Winter 2026, etc.)
TERM_RX = re.compile(r"\b(Spring|Summer|Fall|Winter)\s+20\d{2}\b", re.I)

def extract_all_terms(q: str) -> list[str]:
    """
    Extract all distinct academic terms like 'Winter 2025', 'Summer 2026' from the query.
    Returns title-cased unique terms, in order of appearance.
    """
    seen = set()
    terms = []
    for m in TERM_RX.finditer(q or ""):
        term = m.group(0).title()
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms



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

    # --- helper to run a single Chroma query and build candidates ---
    def run_one(q: str) -> tuple[list[dict], int]:
        where = {"bucket": bucket_filter} if bucket_filter else None

        res = coll.query(
            query_texts=[q],
            n_results=max(k, CANDIDATES),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs  = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None]*len(docs)])[0]

        cand_tables = sum(
            1 for m in metas if bool(m.get("is_table", False) or m.get("table", False))
        )

        candidates: list[dict] = []
        for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            embed_score = (1 - (dist or 0)) if dist is not None else 0.0
            bm25_like   = kw_overlap_score(q, doc)  # lightweight lexical score
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
        return candidates, cand_tables

    # --- detect multi-term queries like "Winter 2025 and Summer 2026" ---
    terms = extract_all_terms(query)

    # ---- SIMPLE PATH: 0 or 1 term -> original behaviour ----
    if len(terms) <= 1:
        candidates, cand_tables = run_one(query)
        print(f"[diagnostic] candidates={len(candidates)}, table_candidates={cand_tables}")

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

            print(f"{i}. bucket={bucket} | base~{base:.3f} | bm25~{bm25:.3f} | "
                  f"final~{final:.3f} | table={is_t} | sec={sec} | term={term} | url={url}")

            heads = normalize_headings(meta.get("headings"))
            if heads:
                print("   headings:", heads[:2])

            # Keep TERM/SECTION markers visible in snippet
            prefix_bits = []
            for ln in (c.get("text","").splitlines()[:8]):
                if ln.startswith("**TERM:**") or ln.startswith("**SECTION:**"):
                    prefix_bits.append(ln)
            prefix = (" ".join(prefix_bits) + " ") if prefix_bits else ""

            full_text = c.get("text","")
            print("   text:", (prefix + full_text).strip())
            print()
        return

    # ---- MULTI-TERM PATH: e.g. "Winter 2025 and Summer 2026" ----
    print(f"[diagnostic] multi-term query detected: {terms}")

    all_candidates: list[dict] = []
    total_tables = 0

    for term in terms:
        sub_query = f"{query} ({term})"
        cands_term, tables_term = run_one(sub_query)
        all_candidates.extend(cands_term)
        total_tables += tables_term

    print(f"[diagnostic] aggregated_candidates={len(all_candidates)}, "
          f"table_candidates={total_tables}")

    # de-duplicate by (url, section, first 200 chars of text)
    deduped: list[dict] = []
    seen_keys = set()
    for c in all_candidates:
        key = (
            c.get("url", ""),
            c.get("section_heading", ""),
            (c.get("text","") or "")[:200],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(c)

    print(f"[diagnostic] deduped_candidates={len(deduped)}")

    # Rerank over the merged candidate set using the *original* query
    topk, intent = rerank(deduped, query, CFG, top_k=k)

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

        print(f"{i}. bucket={bucket} | base~{base:.3f} | bm25~{bm25:.3f} | "
              f"final~{final:.3f} | table={is_t} | sec={sec} | term={term} | url={url}")

        heads = normalize_headings(meta.get("headings"))
        if heads:
            print("   headings:", heads[:2])

        prefix_bits = []
        for ln in (c.get("text","").splitlines()[:8]):
            if ln.startswith("**TERM:**") or ln.startswith("**SECTION:**"):
                prefix_bits.append(ln)
        prefix = (" ".join(prefix_bits) + " ") if prefix_bits else ""

        full_text = c.get("text","")
        print("   text:", (prefix + full_text).strip())
        print()


# manual tests
if __name__ == "__main__":
    #ask("1.What is the course withdrawal deadline?", k=5)
    #ask("2.what is the Last Day to Withdraw Classes for Winter 2025",k=5)
    ask("3.what is the Final Day to Withdraw course for summer 2026",k=5)
    #ask("4.what is the semester end date of Winter 2026", k=5)
    #ask("5.what are the Required Courses for MS in Computer Science",k=5)
    #ask("6.How does pass/fail grading work?", k=5)
    #ask("7.How do I request a leave of absence?", k=5)
    #ask("8.How can I contact academic advising?", k=5)
    #ask("9.What are the eligibility criteria for study abroad?", k=5)
    #ask("10.Last day of Winter 2025 Semester",k=5)
    #ask("11.what is the minimum English test score needed for International students",k=5)
    #ask("12.what is the last day of Session B for Fall 2025", k=5)
    #ask("13.Do we get scholarship as a graduate student", k=5)
    #ask("14.how much scholarship do we get as a graduate student",k=5) #need fix not getting the right chunk
    #ask("15.can you please give me a brief information on Academic Integrity", k=5) # ans chunk is in top 2 not 1
    #ask("16.how many graduate courses or programs are there under school of computing", k=5)
    #ask("17.what are the core research areas in soc", k=5)
    #ask("18.who is the advisor for MS in Computer Science", k=5)
    #ask("19.can you list the Masters courses i can take under SOC", k=5)
    #ask("20.what is the course withdrawl deadline for winter 2025 and Summer 2026",k=5)
    #ask("21.can u tell me more about this course CSIT 515: Software Engineering", k=5)
    #ask("22.how many credits should i complete as a graduate student to complete the degree?", k=5)
    #ask("what are the required courses i should be taking as a graduate student for Computer Science", k=5)
    #ask("i have already completed 4 required courses for MS in CS , how many credits more should i take to complete my degree and suggest some electives i can take for the same?", k=5)

    # follow up qn:
    #ask("can you list some of the elective courses for the same",k=5)
