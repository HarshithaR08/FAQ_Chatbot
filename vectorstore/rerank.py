# Created By: Harshitha (updated with Phase 3 tuning)
# Description: central reranker with smarter intents, term alignment,
#              and better handling of deadlines, calendars, policies,
#              and MS program requirements.

import os, random, re
import numpy as np

# parse "Fall 2025" etc. from the query
TERM_RX = re.compile(r"(fall|spring|summer|winter)\s+(20\d{2})", re.I)
YEAR_RX = re.compile(r"(20\d{2})", re.I)

# NEW: full term parser for both season + year
def parse_term_parts(term: str):
    """
    Returns (season_lower, year_str) or (None, None)
    for strings like "Fall 2025", "SPRING 2026", etc.
    """
    if not term:
        return None, None
    m = TERM_RX.search(term)
    if not m:
        return None, None
    season, year = m.group(1).lower(), m.group(2)
    return season, year

# ---------- deterministic seeds ----------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ---------- intent detection ----------
def detect_intent(q: str, cfg: dict) -> str:
    ql = (q or "").lower()
    intents = cfg.get("query_intents", {}) or {}

    # 1) Config-driven intents first (if user put custom regex there)
    for intent, phrases in intents.items():
        for p in phrases:
            if re.search(p, ql):
                return intent

    # 2) Heuristic calendar vs deadline split
    has_term = bool(TERM_RX.search(ql))
    calendarish = (
        has_term
        and ("semester" in ql or "session" in ql or "term" in ql)
        and (
            "end date" in ql
            or "last day" in ql
            or "first day" in ql
            or "start date" in ql
            or "start of" in ql
            or "end of" in ql
        )
    )
    withdrawish = any(
        w in ql
        for w in [
            "withdraw", "withdrawal", "wd",
            "add/drop", "add / drop", "add or drop",
            "drop a class", "drop classes", "deadline",
        ]
    )

    # If it's clearly about when a semester starts/ends and not about withdrawing,
    # treat this as a calendar intent.
    if calendarish and not withdrawish:
        return "calendar"

    # 3) Deadline-style fallback: term + withdraw/deadline language
    if has_term and (
        "last day" in ql
        or "final day" in ql
        or "end date" in ql
        or "withdraw" in ql
        or "withdrawal" in ql
        or "deadline" in ql
    ):
        return "deadline"

    return "default"


def parse_query_term(q: str):
    m = TERM_RX.search(q or "")
    return (m.group(0).title() if m else None)

# ---------- simple detector for "MS program requirements" ----------
REQ_PAT = re.compile(r"\b(requirements?|required courses?|degree requirements?|program requirements?)\b", re.I)
MS_PAT  = re.compile(
    r"\b("
    r"ms|m\.s\.|"
    r"master'?s|"
    r"masters?\s+of\s+science|"
    r"master'?s\s+degree"
    r")\b",
    re.I,
)

def is_program_requirements_query(q: str) -> bool:
    q = q or ""
    return bool(REQ_PAT.search(q)) and bool(MS_PAT.search(q))

# ---------- helpers for feature scoring ----------
def _bucket_boost(chunk_meta: dict, intent: str, cfg: dict) -> float:
    bw_all = cfg.get("bucket_weights", {})
    table = bw_all.get(intent) or bw_all.get("default") or {}
    bucket = (chunk_meta or {}).get("bucket", "")
    return float(table.get(bucket, 1.0))

def _deadline_keyword_bonus(text: str, cfg: dict) -> float:
    kws = (cfg.get("bm25_guardrail", {}) or {}).get("deadline_keywords", [])
    tl = (text or "").lower()
    return 1.0 if any(k in tl for k in kws) else 0.0

def _term_alignment_bonus(q_term: str, chunk_term: str) -> float:
    """
    Symmetric helper used by both 'deadline' and 'calendar' intents.
    Reward exact semester+year matches, lightly reward year-only matches,
    penalize obvious mismatches.
    """
    qs, qy = parse_term_parts(q_term or "")
    ts, ty = parse_term_parts(chunk_term or "")
    if not qy or not ty:
        return 0.0

    # Exact season + year match
    if qs and ts and qs == ts and qy == ty:
        return 2.5

    # Same year but different season (Fall 2025 vs Winter 2025)
    if qy == ty and qs and ts and qs != ts:
        return -1.0

    # Same year, only one side has season text
    if qy == ty:
        return 1.0

    # Totally different years
    return -1.5


def apply_query_boosts(c, query: str, intent: str) -> float:
    """
    Extra feature-level bonus/penalty based on query words + candidate metadata.
    This returns a value that will be added into `feats` (before scaling).
    """
    q = (query or "").lower()
    meta = getattr(c, "meta", c.get("meta", {})) or {}
    url = (getattr(c, "url", c.get("url", "")) or "").lower()
    sec = (getattr(c, "section_heading", c.get("section_heading", "")) or "").lower()
    text = (getattr(c, "text", c.get("text", "")) or "").lower()
    is_table = bool(meta.get("table") or meta.get("is_table"))

    bonus = 0.0

    # 1) Withdrawal-deadline vs semester-end disambiguation
    if intent == "deadline":
        # If user is clearly asking about withdrawal
        if ("withdraw" in q or "withdrawal" in q or " wd" in q) and ("semester end" not in q and "end of term" not in q):
            # Strongly boost Add/Drop pages & withdrawal policies
            if "add-drop" in url or "add/drop" in url or "withdrawal" in url \
               or "withdrawal or leave of absence from the university" in sec:
                bonus += 0.18
            # Slightly *demote* generic academic calendar pages
            if "academic-calendar" in url:
                bonus -= 0.08

        # If user is clearly asking about semester end date using 'deadline' wording
        if "semester end" in q or "end of term" in q or "last day of semester" in q:
            if "academic-calendar" in url:
                bonus += 0.18
            if "add-drop" in url or "add/drop" in url:
                bonus -= 0.05

        # Waitlist-specific questions
        if "waitlist" in q:
            if "waitlist ends" in text or "waitlist ends" in sec:
                bonus += 0.20
        
        # Generic "course withdrawal deadline" → prefer registrar calendar table
        if "withdraw" in q and "deadline" in q:
            if "the-request-for-course-withdrawal-exception" in url:
                bonus -= 0.4
            if "red-hawk-central/registrar/add-drop" in url and is_table:
                bonus += 0.4

        # NEW: push obviously irrelevant pages down for deadline/withdrawal questions
        if "general-information/grading-standards" in url:
            bonus -= 0.4

        if "/policies/all-policies/withdrawal/" in url:
            bonus -= 0.3

    # 2) Leave of absence: prefer student policies over HR FMLA
    if "leave of absence" in q:
        # Boost student LOA/withdrawal policy + OGE LOA page
        if ("withdrawal or leave of absence from the university" in sec
                or "withdrawal-procedures-from-the-university" in url
                or "global/leave-of-absence" in url):
            bonus += 0.22

        # Demote HR employee policy page
        if "family-leave-and-medical-leaves-of-absence" in url:
            bonus -= 0.15

    return bonus

# ---------- main scoring ----------
def score_candidate(c, intent: str, cfg: dict, query: str, q_term: str):
    """
    c is a dict-like object coming from your candidate list.
    Required keys:
      - 'embed_score'  (float)
      - 'bm25'         (float)
      - 'url'          (str)
      - 'text'         (str)
      - 'section_heading' (str or None)
      - 'meta'         (dict with keys like table, bucket, term, has_deadline_words, etc.)
      - 'chunk_id'     (stable id or index)
      - 'term_matches_query' (bool)  -- set by caller
    """
    rw = cfg.get("rank_weights", {"embed": 0.55, "bm25": 0.25, "bucket": 0.10, "feats": 0.10})

    base = 0.0
    base += rw["embed"] * float(getattr(c, "embed_score", c.get("embed_score", 0.0)))
    base += rw["bm25"]  * float(getattr(c, "bm25", c.get("bm25", 0.0)))

    meta = getattr(c, "meta", c.get("meta", {})) or {}
    bucket = meta.get("bucket", "")
    tags   = meta.get("tags", []) or []

    # bucket prior (centered around 1.0)
    base += rw["bucket"] * (_bucket_boost(meta, intent, cfg) - 1.0)

    # feature bonuses
    feats = 0.0
    sec = getattr(c, "section_heading", c.get("section_heading", "")) or ""
    text = getattr(c, "text", c.get("text", "")) or ""
    url  = getattr(c, "url", c.get("url", "")) or ""

    q_lower   = (query or "").lower()
    sec_lower = sec.lower()
    text_lower = text.lower()

    is_table = bool(meta.get("table") or meta.get("is_table"))
    has_deadline_words = bool(meta.get("has_deadline_words"))
    term_match = bool(getattr(c, "term_matches_query", c.get("term_matches_query", False)))
    term_str = (meta.get("term") or "").lower()

    # ----- deadline-specific features -----
    if intent == "deadline":
        # tables & “deadline” words are generally good
        if is_table:
            feats += 0.10
        if has_deadline_words:
            feats += 0.06
        if term_match:
            feats += 0.08
        
        # Prefer dedicated table sections
        if "table" in sec_lower:
            feats += 0.05

        # Slightly downweight the huge mixed Add/Drop section when it’s not a specific Table chunk
        if "add / drop / withdrawal & adjustment calendar" in sec_lower and "table" not in sec_lower:
            feats -= 0.05

        # If user explicitly talks about withdrawing, prefer chunks that do too
        if "withdraw" in q_lower or "withdrawal" in q_lower or " wd" in q_lower:
            if ("withdraw" in text_lower) or ("withdrawal" in text_lower) or (" wd" in text_lower):
                feats += 0.18   # big bump for actual withdrawal deadlines
            else:
                feats -= 0.12   # push down generic “last day of classes” calendars

            # And specifically downweight plain "Academic Calendar" chunks in withdrawal questions
            if "academic calendar" in text_lower or "academic calendar" in sec_lower:
                feats -= 0.25

        # Use parsed semester+year to reward correct term and punish mismatches
        if q_term:
            feats += _term_alignment_bonus(q_term, term_str or sec)

            # Also reward the section heading mentioning the full "Winter 2025"
            if q_term.lower() in sec_lower:
                feats += 0.5

        # downweight “What Is Considered a Withdrawal?” explanation sections
        if "considered a withdrawal" in sec_lower:
            feats -= 0.10

        # Strongly prefer registrar Add/Drop calendar tables
        if bucket == "registrar" and is_table:
            feats += 0.10
            if "add / drop / withdrawal & adjustment calendar" in sec_lower \
               or "drop / withdrawal / adjustment calendar" in text_lower:
                feats += 0.30
            if term_match:
                feats += 0.10  # registrar rows with matching term should dominate

        # Penalize registrar pages that are about graduation apps when the user
        # is asking about course/semester deadlines.
        if ("graduation" in text_lower or "apply to graduate" in text_lower) and (
            "withdraw" in q_lower or "semester" in q_lower
        ):
            feats -= 0.40

        # Penalize admissions / OGE deadline tables for non-admissions questions
        if bucket == "oge" and "application deadline" in text_lower and not any(
            w in q_lower for w in ["apply", "application", "admission", "freshman", "transfer"]
        ):
            feats -= 0.40

        # Penalize orphan table fragments with no URL (these are the blank-url rows)
        if is_table and not url:
            feats -= 0.20

        feats += 0.02 * _deadline_keyword_bonus(text, cfg)

    # ----- calendar-specific tweaks (semester start/end) -----
    if intent == "calendar":
        # Strongly prefer academic calendar pages in general
        if bucket == "academics-programs":
            feats += 0.25
        if "academic calendar" in text_lower or "academic calendar" in sec_lower:
            feats += 0.20

        # For academic calendar pages, prefer specific year-range URLs
        # over the generic /academic-calendar/ landing page.
        if bucket == "academics-programs":
            if "academic-calendar-20" in url:
                # e.g. /academic-calendar-2025-2026/
                feats += 0.06
            elif url.rstrip("/").endswith("/academic-calendar"):
                # generic landing page, still useful but don't let it tie
                # the year-specific page with identical score
                feats -= 0.12

        # Registrar "Calendar" hub page is also helpful, but below academics/…
        if bucket == "registrar":
            if "/registrar/calendar" in url or "calendar" in sec_lower:
                feats += 0.12
            # but demote add/drop tables for pure calendar questions
            if "add/drop and withdrawal" in text_lower or "withdrawal & adjustment calendar" in text_lower:
                feats -= 0.15

        # Use semester/year alignment in the same way as deadline intent
        if q_term:
            feats += _term_alignment_bonus(q_term, term_str or sec)
            if q_term.lower() in sec_lower:
                feats += 0.4

        # demote irrelevant grading/policy/IT pages for calendar queries
        if "general-information/grading-standards" in url:
            feats -= 0.6

        if bucket == "it-policies" or "software-request-policy-instructors" in url:
            feats -= 0.5

        if "/policies/all-policies/withdrawal/" in url:
            feats -= 0.4

        # HARD NEGATIVE FILTER FOR CALENDAR QUERIES
        # These pages are never the answer for "semester end date", so push them out.

        # IT policies (software request deadlines)
        if bucket == "it-policies" or "information-technology" in url:
            feats -= 3.0   # very strong penalty

        # Any grading-related policy page (incomplete, grading standards, grade changes)
        if ("grading" in url 
             or "grade" in sec_lower
             or "grading" in sec_lower):
            feats -= 3.0

        # Generic student policies that are unrelated to calendar dates
        if bucket in ["policies", "manual"]:
            feats -= 2.0

        # Pages that contain words about deadlines but NOT semester dates
        if ("software" in text_lower or "faculty" in text_lower) and "calendar" not in text_lower:
            feats -= 2.0

    # ----- policy-specific tweaks -----
    if intent == "policy":
        # Strongly prefer actual policy pages
        if bucket == "policies":
            feats += 0.22
        if "about this policy" in sec_lower:
            feats += 0.06

        # Registrar stub that only links out should not beat the real policy
        if bucket == "registrar" and "pass-fail grading policy" in sec_lower:
            if len(text) < 500 or "available on the policies website" in text_lower:
                feats -= 2.5  # effectively pushes it below the real policy

    # ----- program requirements for MS (catalog) -----
    if is_program_requirements_query(query):
        # strongly prefer catalog-programs sections when asking about required courses for an MS
        if bucket == "catalog-programs":
            feats += 0.14
        if "catalog-programs" in tags or "catalog-section" in tags:
            feats += 0.06

        # Prefer Master's program chunks over B.S. chunks
        if "(m.s.)" in text_lower or "(m.s.)" in sec_lower or "master's project" in text_lower:
            feats += 0.22
        if "(b.s.)" in text_lower or "baccalaureate degree" in text_lower:
            feats -= 0.25

        # Reward chunks that clearly expose the required-course list
        if "required courses" in text_lower or "program requirements" in text_lower:
            feats += 0.12
        if "course list" in text_lower and "required courses" in text_lower:
            feats += 0.06

        # Big bonus for the exact MS CS requirements section
        if ("computer-science-ms" in url) and ("requirementstext" in url):
            feats += 0.20
        
        # Penalize navigation / chrome chunk on the MS CS page
        # (Go to Search, Print this page, University Twitter, etc.)
        nav_markers = [
            "go to search",
            "print this page",
            "view all results",
            "university twitter",
            "university facebook",
            "university youtube",       
        ]
        if ("computer-science-ms" in url) and ("requirementstext" not in url):
            if any(m in text_lower for m in nav_markers):
                feats -= 0.60

    # ---- query-specific boosts (withdrawal vs semester-end, LOA, waitlist) ----
    feats += apply_query_boosts(c, query, intent)

    base += rw["feats"] * feats
    return base

# ---------- diversity (MMR-lite per URL cap) ----------
def apply_diversity(sorted_candidates, cfg, intent, q_term=None, k=5):
    """
    Simple per-source diversity.

    Default:
      - cap per URL using diversity.max_per_url (e.g. 2).

    Special case:
      - For deadline/calendar queries *with a parsed term* (e.g. "Winter 2026"),
        we group by (bucket, term) and cap that group at 1.

        This avoids showing the same Winter 2026 academic calendar chunk
        twice just because it exists under two URLs.
    """
    base_max_per = (cfg.get("diversity") or {}).get("max_per_url", 2)
    keep, seen = [], {}

    for c in sorted_candidates:
        url  = getattr(c, "url", c.get("url", "")) or ""
        meta = getattr(c, "meta", c.get("meta", {})) or {}
        bucket = meta.get("bucket", "") or ""

        # ---- Default: group by URL, use config max_per
        key = url
        max_per = base_max_per

        # ---- Calendar / deadline with term: group by (bucket, term) and cap at 1
        if intent in ("deadline", "calendar") and q_term:
            term_label = (meta.get("term") or "").strip().lower()
            if not term_label:
                term_label = q_term.lower()
            key = f"{bucket}##{term_label or 'no-term'}"
            max_per = 1  # <<< IMPORTANT: only one per (bucket, term)

        count = seen.get(key, 0)
        if count >= max_per:
            continue

        seen[key] = count + 1
        keep.append(c)
        if len(keep) >= k:
            break

    return keep

# ---------- guardrail ordering (keyword-first if loss small) ----------
def apply_bm25_guardrail_ordering(candidates, intent, cfg):
    if intent != "deadline":
        return candidates
    kws = (cfg.get("bm25_guardrail", {}) or {}).get("deadline_keywords", [])
    max_loss = (cfg.get("bm25_guardrail", {}) or {}).get("max_gate_loss", 0.15)

    def has_kw(c):
        t = (getattr(c, "text", c.get("text", "")) or "").lower()
        return any(k in t for k in kws)

    with_kw = [c for c in candidates if has_kw(c)]
    loss = 1 - (len(with_kw) / max(1, len(candidates)))
    if loss <= max_loss:
        # push kw matches first, keep original order among ties
        return with_kw + [c for c in candidates if c not in with_kw]
    return candidates

# ---------- master re-ranker ----------
def rerank(candidates, query: str, cfg: dict, top_k=5):
    set_seeds(42)

    intent = detect_intent(query, cfg)
    q_term = parse_query_term(query)

    # annotate term match (use meta.term if present, otherwise fall back to section heading)
    for c in candidates:
        meta = getattr(c, "meta", c.get("meta", {})) or {}
        term = (meta.get("term") or "")
        sec  = getattr(c, "section_heading", c.get("section_heading", "")) or ""

        # prefer explicit term, fall back to heading text
        term_source = term if term else sec
        if q_term and term_source:
            c["term_matches_query"] = (q_term.lower() in str(term_source).lower())
        else:
            c["term_matches_query"] = False

    # score
    for c in candidates:
        c["final_score"] = score_candidate(c, intent, cfg, query, q_term)

    # primary sort (stable/deterministic)
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            -float(getattr(c, "final_score", c.get("final_score", 0.0))),
            -float(getattr(c, "bm25", c.get("bm25", 0.0))),
            -(1 if (
                (getattr(c, "meta", c.get("meta", {})) or {}).get("table")
                or
                (getattr(c, "meta", c.get("meta", {})) or {}).get("is_table")
            ) else 0),
            -(1 if getattr(c, "term_matches_query", c.get("term_matches_query", False)) else 0),
            getattr(c, "url", c.get("url", "")),
            getattr(c, "chunk_id", c.get("chunk_id", 0)),
        )
    )

    # guardrail ordering (small)
    sorted_candidates = apply_bm25_guardrail_ordering(sorted_candidates, intent, cfg)

    # diversity cap
    top = apply_diversity(sorted_candidates, cfg, intent, q_term=q_term, k=top_k)
    return top, intent