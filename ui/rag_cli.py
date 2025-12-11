# Developers: Harshitha, Imad
"""
ui/rag_cli.py

MSU FAQ Chatbot — RAG CLI

- Retrieval + reranking from Chroma + vectorstore.rerank
- LLaMA 3 8B Instruct from infra.llama3_client

Design:
- ALWAYS uses retrieval + rerank as in vectorstore.test_retrieval_query.
- Deterministic fast-path ONLY for:
    * Withdrawal deadlines from registrar Add/Drop tables
- EVERYTHING ELSE goes through the LLM.

No course lists are hardcoded. We only bias retrieval and context for catalog pages.
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

    NOTE: This is the ONLY fast-path we keep.
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


# ---------- (Optional) STRUCTURED program requirements extraction ----------
# NOTE: helper exists but is NOT used in any fast path; all answers go through the LLM.

COURSE_CODE_RX = re.compile(
    r"^(CSIT|MATH|AMAT|STAT|PHYS)\s*[\u00A0 ]?\d{3}\b",
    re.IGNORECASE,
)

COURSE_LINE_RX = re.compile(
    r"^(?P<code>(CSIT|MATH|AMAT|STAT|PHYS)\s*[\u00A0 ]?\d{3})"
    r"\s*[-–:]?\s*"
    r"(?P<title>.*?)"
    r"\s*(?:\((?P<credits>\d+)\s+credits?\))?\s*$",
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
    NOT used in normal flow; kept only as a debugging helper.
    """
    if not text:
        return []

    raw_lines = text.splitlines()
    lines = [ln.strip() for ln in raw_lines]

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
        s = ln.strip()
        if not s:
            block.append("")
            continue
        if STOP_SECTION_RX.match(s):
            break
        block.append(s)

    courses: List[Dict[str, Optional[str]]] = []
    seen_codes = set()
    i = 0
    n = len(block)

    def _record_course(code: str, title: Optional[str], credits: Optional[str]):
        norm_code = (code or "").upper().replace("\u00A0", " ").strip()
        if not norm_code:
            return
        if norm_code in seen_codes:
            return
        seen_codes.add(norm_code)
        courses.append(
            {
                "code": norm_code,
                "title": (title or "").strip() or None,
                "credits": (credits or "").strip() or None,
            }
        )

    while i < n:
        line = block[i].strip()
        if not line:
            i += 1
            continue

        # 1) Try single-line pattern first
        m_full = COURSE_LINE_RX.match(line)
        if m_full:
            code = m_full.group("code") or ""
            title = (m_full.group("title") or "").strip() or None
            credits = m_full.group("credits")
            _record_course(code, title, credits)
            i += 1
            continue

        # 2) Fallback: code-only line with title/credits on subsequent lines
        m_code = COURSE_CODE_RX.match(line)
        if not m_code:
            i += 1
            continue

        code = m_code.group(0).strip()

        # Look ahead for title
        title = None
        credits = None

        j = i + 1
        while j < n and not block[j].strip():
            j += 1
        if j < n:
            if not COURSE_CODE_RX.match(block[j]) and not STOP_SECTION_RX.match(block[j].strip()):
                title = block[j].strip()
                j += 1

        # Look ahead for credits
        k = j
        while k < n and not block[k].strip():
            k += 1
        if k < n:
            if re.fullmatch(r"\d{1,2}", block[k].strip()):
                credits = block[k].strip()

        _record_course(code, title, credits)

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
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Use the SAME retrieval + rerank pipeline as vectorstore.test_retrieval_query.ask.

    Returns:
        top_chunks: list of candidate dicts
        intent: string like "deadline", "calendar", "default"
    """
    coll = get_collection()
    terms = extract_all_terms(query)

    # If this looks like a program requirements query,
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


# ---------- Program-related catalog filtering (NO extra fast-path) ----------

def filter_catalog_chunks_for_requirements(
    query: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For queries clearly about program/degree requirements, prefer catalog
    sections that match what the user is asking.

    Still no hardcoding of course data; we only pick which sections to show.
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

    if not (wants_required or wants_electives or is_program_requirements_query(query)):
        return chunks

    filtered: List[Dict[str, Any]] = []
    for c in chunks:
        sec = (c.get("section_heading") or "").lower()
        bucket = (c.get("meta", {}) or {}).get("bucket", "")

        if bucket != "catalog-programs":
            filtered.append(c)
            continue

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

        if wants_electives and not wants_required:
            if "elective" in sec:
                filtered.append(c)
            else:
                filtered.append(c)
            continue

        filtered.append(c)

    return filtered or chunks

def build_messages(
    query: str,
    top_chunks: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Build the chat messages for LLaMA.

    This version is intentionally lighter:
    - Shorter system prompt.
    - Smaller snippets.
    - At most 6 chunks sent to the model.
    - MUCH stricter rules against personalized credit math / hallucinated counts.
    """
    filtered_chunks = filter_catalog_chunks_for_requirements(query, top_chunks)

    # HARD CAP: only send up to 6 chunks into the LLM, no matter what.
    filtered_chunks = filtered_chunks[:6]

    # Smaller snippets to keep context size under control
    snippet_max = 1000

    context_blocks = []
    for i, c in enumerate(filtered_chunks, 1):
        meta = c.get("meta", {}) or {}
        url = c.get("url", "")
        text = c.get("text", "") or ""
        sec = c.get("section_heading") or ""

        snippet = text[:snippet_max] + ("…" if len(text) > snippet_max else "")

        # Clean headings like "H2: 2025-2026 Edition"
        sec_clean = sec
        if sec_clean.startswith("H") and ":" in sec_clean[:4]:
            sec_clean = sec_clean.split(":", 1)[1].strip()

        sec_line = f"Section: {sec_clean}\n" if sec_clean else ""

        context_blocks.append(
            f"[{i}] Source URL: {url}\n"
            f"{sec_line}"
            f"{snippet}"
        )

    context = "\n\n".join(context_blocks) if context_blocks else "(no context found)"

    # === STRICT system prompt ===
    system_prompt = (
        "You are the Montclair State University FAQ assistant.\n"
        "You answer questions about deadlines, academic policies, registrar rules, and School of Computing information.\n"
        "You are given short excerpts from official Montclair State University web pages (labeled [1], [2], etc.).\n"
        "Use ONLY the information in these excerpts.\n"
        "\n"
        "General rules:\n"
        "  - If the answer is not clearly stated in the excerpts, say you are not sure and suggest which office or page to contact.\n"
        "  - Do NOT guess or fill in missing details for dates, credit totals, course lists, or test scores.\n"
        "  - Do NOT mention internal words like 'snippet', 'chunk', 'Bucket', or 'context above'.\n"
        "  - When you refer to a fact, attach a citation like [1] or [2] pointing to the excerpt that contains it.\n"
        "\n"
        "Program requirements / required and elective courses:\n"
        "  - When the question is about required courses, program requirements, or degree requirements, copy course codes,\n"
        "    titles, and credit values exactly as they appear in the excerpts. Do not invent or rename courses.\n"
        "  - You may list elective courses that appear in the excerpts, but you must NOT invent or assume how many electives\n"
        "    the student must choose unless the exact number of courses or credits is written explicitly in the excerpts.\n"
        "    For example, only say '5 additional courses' if a sentence like 'choose 5 courses from the following' or\n"
        "    '5 additional courses' appears verbatim.\n"
        "  - If a title or credit value is cut off in the excerpt, say that it is not fully visible instead of guessing.\n"
        "\n"
        "Per-student credit / remaining-credits questions:\n"
        "  - You may state the TOTAL credits required for a degree or section only when that total appears in the excerpts\n"
        "    (for example, 'A minimum of 30 semester hours of graduate credit is required' [1]).\n"
        "  - Even if the student tells you what they have already completed (for example, 'I finished 4 required courses'),\n"
        "    you must NOT compute or state how many more credits or courses they personally need. Do NOT say things like\n"
        "    'you still need 2 more courses' or 'you need X more credits'.\n"
        "  - Instead, repeat the official totals (e.g., 'the degree requires 30 credits in total [1]') and advise them to\n"
        "    check Degree Works or speak with their academic advisor or Graduate Program Coordinator to see their exact\n"
        "    remaining credits.\n"
        "\n"
        "Different student levels:\n"
        "  - If excerpts show different rules for undergraduate, graduate, or doctoral students, clearly separate them and say\n"
        "    which rule applies to which group.\n"
        "\n"
        "English test scores:\n"
        "  - For English proficiency tables, list the exam names and minimum scores exactly as shown, and note when scores differ\n"
        "    for undergrad vs graduate applicants.\n"
        "\n"
        "Answer concisely in 2–4 sentences unless the question explicitly asks for a detailed list of courses.\n"
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


ADVISOR_WORDS = (
    "advisor",
    "advisors",
    "advising",
    "gpc",
    "graduate program coordinator",
    "graduate program chair",
)


def pick_primary_source(query: str, chunks: list[dict]) -> dict | None:
    """
    Heuristic: choose the most likely relevant chunk as primary_source.
    - For advisor / GPC questions: prefer School of Computing pages that
      mention advisors or GPC in the text.
    - Otherwise: fall back to chunks[0].
    """
    if not chunks:
        return None

    ql = (query or "").lower()

    if any(w in ql for w in ADVISOR_WORDS):
        best = None
        for c in chunks:
            url = (c.get("url") or "").lower()
            text = (c.get("text") or "").lower()

            if "school-of-computing" in url and (
                "advisor" in text
                or "advisors" in text
                or "graduate program coordinator" in text
                or "gpc" in text
            ):
                best = c
                break

        if best is None:
            best = chunks[0]
    else:
        best = chunks[0]

    meta0 = (best.get("meta", {}) or {})
    return {
        "idx": 1,
        "url": best.get("url", ""),
        "bucket": meta0.get("bucket", ""),
        "title": best.get("section_heading") or meta0.get("title", ""),
    }


# ---------- LLaMA 3 call + deterministic fast-paths ----------

def answer_with_sources(query: str, top_k: int = 5) -> dict:
    """
    Convenience wrapper around generate_answer() for evaluation scripts.
    """
    answer, primary_source, intent = generate_answer(query, top_k=top_k)
    return {
        "answer": answer,
        "primary_source": primary_source,
        "intent": intent,
    }


def generate_answer(
    query: str,
    top_k: int = 5,
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """
    Full RAG: retrieve -> (maybe deterministic answer) -> LLaMA 3.

    Fast-path is ONLY used for withdrawal/deadline questions.
    All other questions go through the LLM.
    """
    # For program requirements queries, we *do not* bump top_k too high to avoid OOM.
    chunks, intent = retrieve_top_chunks(query, top_k=top_k)
    print(f"[diagnostic] detected_intent={intent!r}   retrieved_chunks={len(chunks)}")

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

    # smaller generation length to reduce memory
    max_new_tokens = 200

    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.2,
        pad_token_id=tok.eos_token_id,
    )

    full_text = outputs[0]["generated_text"]
    answer = full_text[len(prompt):].strip()

    primary_source = pick_primary_source(query, chunks)

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
            answer, primary_source, intent = generate_answer(q, top_k=5)
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
