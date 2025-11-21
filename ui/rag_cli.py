# Developers: Harshitha, Imad
"""
ui/rag_cli.py

MSU FAQ Chatbot — RAG CLI

- Retrieval + reranking from Chroma + vectorstore.rerank
- LLaMA 3 8B Instruct from infra.llama3_client

Design:
- ALWAYS uses retrieval + rerank as in vectorstore.test_retrieval_query.
- Deterministic fast-path only for:
    * Program "required courses"/"required core courses" lists
    * Withdrawal deadlines from registrar Add/Drop tables
- EVERYTHING ELSE goes through the LLM.

No course lists are hardcoded. We only parse the catalog text structure.
"""

from typing import List, Dict, Any, Tuple, Optional
import re
import math
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions

from infra.llama3_client import get_llama3_pipeline
from vectorstore.rerank import rerank, is_program_requirements_query
from vectorstore.test_retrieval_query import (
    RERANK_CONFIG as CFG,
    extract_all_terms,  # e.g., "Winter 2025", "Summer 2026"
)

# ---------- Chroma setup (MATCHES test_retrieval_query.py) ----------

CHROMA_DIR = "vectorstore/chroma"
COLLECTION = "msu_faq_chunks"
CANDIDATES = 200  # pull many so reranker has room

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION, embedding_function=embedding_fn)


# ---------- Helpers: keyword overlap ----------

def tokenize(s: str) -> List[str]:
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


# ---------- Helper: extract deadline from registrar tables ----------

def extract_deadline_from_table(text: str) -> Optional[str]:
    """
    Extract the 'Final Day to WD' / 'Final Day to Withdraw' date
    for the FULL term from a registrar Add/Drop & Withdrawal table.
    """
    if not text:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    date_rx = re.compile(r"\d{1,2}/\d{1,2}(?:/\d{2,4})?")

    # 1) locate header line
    header_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        ll = ln.lower()
        if "final day to wd" in ll or "final day to withdraw" in ll:
            header_idx = i
            break

    if header_idx is None:
        return None

    header_cols = [c.strip() for c in lines[header_idx].split("|")]

    # 2) find the column index
    col_idx: Optional[int] = None
    for i, col in enumerate(header_cols):
        cl = col.lower()
        if "final day to wd" in cl or "final day to withdraw" in cl:
            col_idx = i
            break

    if col_idx is None:
        return None

    # 3) find FULL term row
    row_line: Optional[str] = None
    for ln in lines[header_idx + 1:]:
        ll = ln.lower()
        if "full term" in ll or ll.startswith("| full ") or "| full |" in ll or ll.startswith("| full\t"):
            row_line = ln
            break

    if not row_line:
        # fallback: sometimes just "Full" appears
        for ln in lines[header_idx + 1:]:
            ll = ln.lower()
            if ll.startswith("| ---"):
                continue
            if "| full" in ll:
                row_line = ln
                break

    if not row_line:
        return None

    row_cols = [c.strip() for c in row_line.split("|")]
    if col_idx >= len(row_cols):
        return None

    cell = row_cols[col_idx]
    m = date_rx.search(cell)
    if m:
        return m.group(0)

    return None


# ---------- Helper: STRUCTURED program requirements extraction ----------

COURSE_CODE_RX = re.compile(
    r"^(CSIT|MATH|AMAT|STAT|PHYS)\s*[\u00A0 ]?\d{3}\b",  # handle normal + non-breaking spaces
    re.IGNORECASE,
)

STOP_SECTION_RX = re.compile(
    r"^(electives|elective courses|culminating experience|total credits)\b",
    re.IGNORECASE,
)


def extract_required_courses_structured(
    text: str,
) -> List[Dict[str, Optional[str]]]:
    """
    Given raw catalog text for a program (one chunk), extract a structured list
    of *required* / *required core* courses.

    Strategy:
      - Find the first "Required Courses" or "Required Core Courses" heading.
      - From there, scan line-by-line until we hit a "stop" section ("Electives",
        "Elective Courses", "Culminating Experience", "Total Credits").
      - Within that block, detect triplets:
            CODE  (e.g., CSIT 515)
            TITLE (e.g., Software Engineering)
            CREDITS (e.g., 3)
        We pair code + title, credits optional.
      - Deduplicate by course code.
    """
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines()]
    if not lines:
        return []

    # Locate required section start
    start_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        ll = ln.lower()
        if "required courses" in ll or "required core courses" in ll:
            start_idx = i + 1
            break

    if start_idx is None:
        return []

    # Collect lines until a stopping section
    block: List[str] = []
    for ln in lines[start_idx:]:
        if STOP_SECTION_RX.match(ln.strip()):
            break
        block.append(ln.strip())

    courses: List[Dict[str, Optional[str]]] = []
    seen_codes = set()
    i = 0
    n = len(block)

    while i < n:
        line = block[i]
        m = COURSE_CODE_RX.match(line)
        if not m:
            i += 1
            continue

        code = line  # e.g. "CSIT 515"

        # Look ahead for title (next non-empty line)
        title = None
        credits = None

        j = i + 1
        while j < n and not block[j]:
            j += 1
        if j < n:
            # If the next line is clearly not another course code or header, treat as title
            if not COURSE_CODE_RX.match(block[j]) and not STOP_SECTION_RX.match(block[j]):
                title = block[j].strip()
                j += 1

        # Look ahead for credits: a small integer on its own line
        k = j
        while k < n and not block[k]:
            k += 1
        if k < n:
            if re.fullmatch(r"\d{1,2}", block[k]):
                credits = block[k].strip()
                # do not necessarily need to skip k; we just move i forward enough

        norm_code = code.upper()
        if norm_code not in seen_codes:
            seen_codes.add(norm_code)
            courses.append(
                {
                    "code": code,
                    "title": title,
                    "credits": credits,
                }
            )

        # Move i forward to avoid re-processing same lines too much
        i = max(i + 1, j, k)

    return courses


# ---------- Retrieval helper (MATCHES test_retrieval_query logic) ----------

def _build_candidates_for_query(
    coll,
    q: str,
    k: int,
    bucket_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build initial Chroma candidates for a query, with optional bucket filter.
    """
    where = {"bucket": bucket_filter} if bucket_filter else None
    res = coll.query(
        query_texts=[q],
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
        bm25_like = kw_overlap_score(q, doc)
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


def retrieve_top_chunks(
    query: str,
    top_k: int = 6,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Use the SAME retrieval + rerank pipeline as vectorstore.test_retrieval_query.ask.

    Returns:
        top_chunks: list of candidate dicts
        intent: string like "deadline", "calendar", "default"
    """
    coll = get_collection()
    terms = extract_all_terms(query)

    # If this looks like an MS program requirements query,
    # restrict retrieval to catalog-programs (official catalog pages).
    bucket_filter = "catalog-programs" if is_program_requirements_query(query) else None

    # Simple path: 0 or 1 term
    if len(terms) <= 1:
        base_candidates = _build_candidates_for_query(
            coll,
            query,
            k=top_k,
            bucket_filter=bucket_filter,
        )
        top_chunks, intent = rerank(base_candidates, query, CFG, top_k=top_k)
        print(
            f"[diagnostic] candidates={len(base_candidates)}, retrieved_chunks={len(top_chunks)}"
        )
        return top_chunks, intent

    # Multi-term path: "Winter 2025 and Summer 2026"
    print(f"[diagnostic] multi-term query detected in RAG CLI: {terms}")
    all_candidates: List[Dict[str, Any]] = []
    seen_keys = set()

    for term in terms:
        sub_query = f"{query} ({term})"
        cands = _build_candidates_for_query(
            coll,
            sub_query,
            k=top_k,
            bucket_filter=bucket_filter,
        )
        for c in cands:
            key = (
                c.get("url", ""),
                c.get("section_heading", ""),
                (c.get("text", "") or "")[:200],
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_candidates.append(c)

    print(f"[diagnostic] aggregated_base_candidates={len(all_candidates)}")
    top_chunks, intent = rerank(all_candidates, query, CFG, top_k=top_k)
    print(f"[diagnostic] retrieved_chunks={len(top_chunks)} intent={intent}")
    return top_chunks, intent


# ---------- Program requirements fast path ----------

def try_program_requirements_answer(
    query: str,
    chunks: List[Dict[str, Any]],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Deterministic answer builder for questions like:
      - 'What are the required courses for MS in Computer Science?'
      - 'Give me the required core courses for MS in Data Science.'

    Uses ONLY the retrieved catalog-programs chunks and extracts course
    lines from the "Required Courses" / "Required Core Courses" block.

    Returns (answer_text, primary_source_chunk) or None.
    """
    if not chunks:
        return None

    if not is_program_requirements_query(query):
        return None

    # Prefer catalog-programs chunks
    best_chunk = None
    fallback_chunk = None

    for c in chunks:
        meta = c.get("meta", {}) or {}
        bucket = meta.get("bucket", "")
        if bucket != "catalog-programs":
            continue

        if fallback_chunk is None:
            fallback_chunk = c

        # Heuristic: prefer sections whose heading looks like a program page
        sec = (c.get("section_heading") or "").lower()
        if "data science (m.s.)" in sec or "computer science (m.s.)" in sec:
            best_chunk = c
            break

    target = best_chunk or fallback_chunk
    if not target:
        return None

    text = target.get("text", "") or ""
    courses = extract_required_courses_structured(text)

    # If we couldn't find any course-looking lines, bail out to LLM
    if len(courses) == 0:
        return None

    # Build bullet list with "CODE – Title (credits)" where available
    bullets = []
    for c in courses:
        code = c.get("code") or ""
        title = c.get("title") or ""
        credits = c.get("credits")
        if credits:
            bullets.append(f"* {code} – {title} ({credits} credits)")
        else:
            bullets.append(f"* {code} – {title}" if title else f"* {code}")

    bullet_list = "\n".join(bullets)

    answer = (
        "According to the official Montclair State University catalog [1], "
        "the Required Courses for this program are:\n\n"
        f"{bullet_list}\n\n"
        "These courses come from the 'Program Requirements' section of the catalog."
    )

    return answer, target


# ---------- Prompt building (LLM fallback) ----------

def filter_catalog_chunks_for_requirements(
    query: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For queries clearly about program/degree requirements, prefer catalog
    sections that match what the user is asking:

    - If they ask about 'required courses' / 'program requirements':
        keep 'Required Courses' / 'Program Requirements' sections.
    - If they ask about 'elective(s)':
        keep 'Elective' sections.

    This does NOT hard-code any actual course data; it just chooses which
    catalog sections to show to the LLM.
    """
    if not chunks:
        return chunks

    q_low = (query or "").lower()
    wants_electives = ("elective" in q_low or "electives" in q_low)
    wants_required = (
        "required courses" in q_low
        or "required core" in q_low
        or "program requirements" in q_low
        or "degree requirements" in q_low
    )

    # If it isn't a program-requirements-style query, leave chunks as-is
    if not (wants_required or wants_electives or is_program_requirements_query(query)):
        return chunks

    filtered: List[Dict[str, Any]] = []
    for c in chunks:
        sec = (c.get("section_heading") or "").lower()
        bucket = (c.get("meta", {}) or {}).get("bucket", "")

        # Only apply this special logic to catalog-programs chunks
        if bucket != "catalog-programs":
            filtered.append(c)
            continue

        # Required/program requirements path
        if wants_required and not wants_electives:
            if (
                "required courses" in sec
                or "required core" in sec
                or "program requirements" in sec
                or "degree requirements" in sec
            ):
                filtered.append(c)
            elif "elective" in sec:
                continue
            else:
                filtered.append(c)
            continue

        # Elective path
        if wants_electives and not wants_required:
            if "elective" in sec:
                filtered.append(c)
            else:
                filtered.append(c)
            continue

        # If both required and electives are in the question, don't filter at all.
        filtered.append(c)

    return filtered or chunks


def build_messages(
    query: str,
    top_chunks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    filtered_chunks = filter_catalog_chunks_for_requirements(query, top_chunks)

    context_blocks = []
    for i, c in enumerate(filtered_chunks, 1):
        meta = c.get("meta", {}) or {}
        url = c.get("url", "")
        text = c.get("text", "") or ""
        sec = c.get("section_heading") or ""

        snippet = text[:1200] + ("…" if len(text) > 1200 else "")
        sec_line = f"Section: {sec}\n" if sec else ""

        context_blocks.append(
            f"[{i}] Source: {url}\n"
            f"Bucket: {meta.get('bucket', '')}\n"
            f"{sec_line}"
            f"{snippet}"
        )

    context = "\n\n".join(context_blocks) if context_blocks else "(no context found)"

    system_prompt = (
        "You are the Montclair State University FAQ assistant.\n"
        "You answer questions about deadlines, academic policies, registrar rules, and School of Computing information.\n"
        "You are given short excerpts copied from official Montclair State University web pages (labeled [1], [2], etc.).\n"
        "Use ONLY this information when you answer.\n"
        "If the answer is not clearly stated in the context, say that you are not sure because it is not present "
        "in these excerpts, and suggest who the student should contact.\n"
        "Do NOT claim that Montclair State University or the catalog 'does not list' something unless the text "
        "explicitly says so.\n"
        "\n"
        "When you DO know the answer, you MUST:\n"
        "  - Extract the exact date / rule / requirement from the excerpts.\n"
        "  - For questions about 'required courses', 'required core courses', 'program requirements', or 'degree requirements', "
        "you must list the required courses exactly as they appear in the context (course codes, titles, and credit counts), "
        "without inventing or omitting courses.\n"
        "  - Distinguish clearly between 'Required Courses', 'Required Core Courses', 'Electives', and 'Culminating Experience' sections.\n"
        "  - Answer concisely in 2–4 sentences.\n"
        "  - Add a citation like [1] or [2] pointing to the excerpt index that directly supports your answer.\n"
        "\n"
        "Important style rules:\n"
        "  - Do NOT mention 'snippets', 'chunks', 'context above', or anything about how the data was retrieved.\n"
        "  - Instead, phrase answers like 'According to the Registrar's Add/Drop & Withdrawal calendar [1]...' or\n"
        "    'According to the official Montclair State University catalog [1]...'.\n"
        "  - Never guess dates or policies that are not clearly present.\n"
    )

    user_content = (
        f"Student question:\n{query}\n\n"
        f"Relevant official web-page excerpts:\n{context}\n\n"
        "Using ONLY the information above, write your answer now."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages


# ---------- LLaMA 3 call + deterministic fast-paths ----------

def answer_with_sources(query: str, top_k: int = 6) -> dict:
    """
    Convenience wrapper around generate_answer() for evaluation scripts.

    This DOES NOT change any chatbot behavior.
    It just exposes the same pipeline (including fast paths)
    in a structured form.
    """
    answer, primary_source, intent = generate_answer(query, top_k=top_k)
    return {
        "answer": answer,
        "primary_source": primary_source,
        "intent": intent,
    }


def generate_answer(
    query: str,
    top_k: int = 6,
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """
    Full RAG: retrieve -> (maybe deterministic answer) -> LLaMA 3.

    Returns:
        answer: str
        primary_source: Optional[dict]
        intent: str
    """
    chunks, intent = retrieve_top_chunks(query, top_k=top_k)
    print(f"[diagnostic] detected_intent={intent!r}   retrieved_chunks={len(chunks)}")

    # ----- PROGRAM REQUIREMENTS FAST-PATH -----
    prog_fast = try_program_requirements_answer(query, chunks)
    if prog_fast is not None:
        answer, primary_chunk = prog_fast
        meta0 = (primary_chunk.get("meta", {}) or {})
        primary_source = {
            "idx": 1,
            "url": primary_chunk.get("url", ""),
            "bucket": meta0.get("bucket", ""),
            "title": primary_chunk.get("section_heading") or meta0.get("title", ""),
        }
        print("[diagnostic] using PROGRAM-REQUIREMENTS fast path")
        return answer, primary_source, intent

    # ----- DEADLINE FAST-PATH: read registrar table(s) directly -----
    if intent == "deadline" and chunks:
        terms = extract_all_terms(query)  # e.g. ["Winter 2025", "Summer 2026"]

        if not terms:
            terms = []

        term_deadlines: List[Tuple[str, str, Dict[str, Any]]] = []

        for term in terms or [None]:
            term_lower = term.lower() if term else None
            best_chunk: Optional[Dict[str, Any]] = None

            for c in chunks:
                meta = c.get("meta", {}) or {}
                bucket = meta.get("bucket", "")
                is_table = bool(meta.get("table") or meta.get("is_table"))
                sec = (c.get("section_heading") or "").lower()
                meta_term = (meta.get("term") or "").lower()

                if bucket != "registrar" or not is_table:
                    continue

                if term_lower:
                    if term_lower not in meta_term and term_lower not in sec:
                        continue

                best_chunk = c
                break

            if not best_chunk:
                continue

            d = extract_deadline_from_table(best_chunk.get("text", "") or "")
            if not d:
                continue

            meta = best_chunk.get("meta", {}) or {}
            nice_term = (
                meta.get("term")
                or best_chunk.get("section_heading")
                or (term or "this term")
            )
            term_deadlines.append((nice_term, d, best_chunk))

        if term_deadlines:
            for t, d, _ in term_deadlines:
                print(f"[diagnostic] extracted deadline for {t}: {d}")

            if len(term_deadlines) == 1:
                t, d, ch = term_deadlines[0]
                answer = (
                    f"According to the Registrar's Add/Drop & Withdrawal calendar, "
                    f"the Final Day to Withdraw for {t} is **{d}**."
                )
                primary_chunk = ch
            else:
                parts = [f"for {t} it is **{d}**" for (t, d, _) in term_deadlines]
                answer = (
                    "According to the Registrar's Add/Drop & Withdrawal calendar, "
                    + "; ".join(parts)
                    + "."
                )
                primary_chunk = term_deadlines[0][2]

            meta0 = primary_chunk.get("meta", {}) or {}
            primary_source = {
                "idx": 1,
                "url": primary_chunk.get("url", ""),
                "bucket": meta0.get("bucket", ""),
                "title": primary_chunk.get("section_heading") or meta0.get("title", ""),
            }
            print("[diagnostic] using DEADLINE fast path")
            return answer, primary_source, intent

    # ----- FALLBACK: call LLaMA 3 with retrieved context -----
    pipe, tok = get_llama3_pipeline()
    messages = build_messages(query, chunks)
    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = pipe(
        prompt,
        max_new_tokens=320,
        do_sample=False,
        temperature=0.2,
        pad_token_id=tok.eos_token_id,
    )

    full_text = outputs[0]["generated_text"]
    answer = full_text[len(prompt):].strip()

    primary_source: Optional[Dict[str, Any]] = None
    if chunks:
        c0 = chunks[0]
        meta0 = c0.get("meta", {}) or {}
        primary_source = {
            "idx": 1,
            "url": c0.get("url", ""),
            "bucket": meta0.get("bucket", ""),
            "title": c0.get("section_heading") or meta0.get("title", ""),
        }

    return answer, primary_source, intent


# ---------- CLI ----------

def main():
    print("=== MSU FAQ Chatbot — LLaMA 3 RAG CLI ===")
    print("Type a question about Montclair State deadlines, academic policies, or School of Computing information.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            answer, primary_source, intent = generate_answer(q, top_k=6)
        except Exception as e:
            print(f"[error] {e}")
            continue

        print("\n--- Answer ---")
        print(answer)
        print("--------------")

        print("Source:")
        if primary_source:
            url = primary_source["url"] or "(no URL)"
            bucket = primary_source.get("bucket") or ""
            title = primary_source.get("title") or ""
            extra = f" | bucket={bucket}" if bucket else ""
            if title:
                extra += f" | title={title}"
            print(f"[1] {url}{extra}")
        else:
            print("(no primary source found)")
        print()


if __name__ == "__main__":
    main()
