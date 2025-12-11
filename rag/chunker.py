# Developer Name: Harshitha
# Read Markdown (with front-matter) from data/processed/, split into small overlapping chunks,
# and write rag/chunks.jsonl for indexing.

import os, re, json, logging, hashlib, frontmatter
import yaml
from pathlib import Path
from typing import List, Dict

# split the size of each chunk into 1200 character pieces 
# the overlap between adjacent chunks keeps continuity so a fact on a chunk boundary isn’t lost.
# 1 chunk ≈ 1200 characters ≈ roughly 1200–1500 bytes per chunk 
TARGET_CHARS = 1200      # ~150–300 tokens (roughly)
OVERLAP_CHARS = 200

def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

# Added Fix to snap the start and end boundaries to a nearby whitespace or punctuation
# word/sentence-safe boundaries
_BOUNDARY = re.compile(r"[ \t\n\r\f\v\.\,\;\:\!\?\)\]]")
TERM_MARKER_RX = re.compile(r"^\*\*TERM:\*\*\s*(.+)$", re.M)

# ---------- Light de-noising for catalog program pages (old logic) ----------
COURSE_LIST_NOISE = re.compile(
    r"(Course List\s+Code\s+Title\s+Credits\s+)+",
    flags=re.IGNORECASE
)

# ---------- NEW: normalize course lines on catalog program pages ----------

COURSE_CODE_RX = re.compile(
    r"^(CSIT|MATH|AMAT|STAT|PHYS)\s*[\u00A0 ]?\d{3}\b",
    re.IGNORECASE,
)

def normalize_catalog_courses(md_text: str) -> str:
    """
    For catalog program pages, HTML->markdown often produces:

        Required Courses
        CSIT 515
        Software Engineering
        3
        CSIT 545
        Computer Architecture
        3
        ...

    This rewrites that into single-line entries:

        CSIT 515 Software Engineering (3 credits)
        CSIT 545 Computer Architecture (3 credits)

    so the LLM can safely copy exact lines.
    """
    lines = [ln.rstrip("\n") for ln in md_text.splitlines()]
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        m = COURSE_CODE_RX.match(stripped)
        if not m:
            out.append(line)
            i += 1
            continue

        code = stripped
        title = None
        credits = None

        # Look ahead for title (next non-empty line that is not another course code)
        j = i + 1
        while j < n and not lines[j].strip():
            j += 1
        if j < n and not COURSE_CODE_RX.match(lines[j].strip()):
            title = lines[j].strip()
            j += 1

        # Look ahead for credits: a small integer on its own line
        k = j
        while k < n and not lines[k].strip():
            k += 1
        if k < n and re.fullmatch(r"\d{1,2}", lines[k].strip()):
            credits = lines[k].strip()
            k += 1

        # Rebuild compact line
        if title and credits:
            out.append(f"{code} {title} ({credits} credits)")
        elif title:
            out.append(f"{code} {title}")
        else:
            out.append(code)

        i = max(i + 1, j, k)

    return "\n".join(out)

def clean_markdown_text(text: str, source_url: str) -> str:
    """
    Apply small, source-specific cleanups before chunking.
    Currently:
      - de-duplicate 'Course List / Code / Title / Credits' noise on catalog program pages
      - keep hook for other future source-specific cleanups
    """
    u = (source_url or "").lower()
    if "catalog.montclair.edu/programs/" in u:
        text = COURSE_LIST_NOISE.sub("Course List\nCode\nTitle\nCredits\n", text)
        # NOTE: deeper catalog cleanup & course normalization are applied later
        # in chunk_doc once we know the bucket.
    return text

# ---------- helpers to detect tables / labels / terms ----------
TABLE_RX_MD   = re.compile(r"\|.+\|.+\|")       # crude markdown table row
TABLE_RX_HTML = re.compile(r"<table\b", re.I)   # for any html table remnants
SEMESTER_RX   = re.compile(r"\b(spring|summer|fall|winter)\s+20\d{2}\b", re.I)

def is_table_block(text: str) -> bool:
    return bool(TABLE_RX_MD.search(text)) or bool(TABLE_RX_HTML.search(text))

def section_label(heading_path: List[str]) -> str:
    if not heading_path:
        return ""
    last = heading_path[-1].lower()
    if "calendar" in last:
        return "calendar"
    return last.split(": ", 1)[-1].strip()

def infer_term(text: str, heading_path: List[str]) -> str:
    # 0) Prefer explicit **TERM:** marker emitted by extract_tables.py
    m = TERM_MARKER_RX.search(text or "")
    if m:
        return m.group(1).strip()

    # 1) Then try headings/content like before
    joined = " — ".join(heading_path or [])
    m2 = SEMESTER_RX.search(joined) or SEMESTER_RX.search(text or "")
    return m2.group(0).title() if m2 else ""

def safe_slice(txt: str, start: int, end: int, pad: int = 40) -> str:
    start = max(0, start)
    end   = min(len(txt), end)
    if start >= end:
        return txt[start:end]
    # nudge start to next boundary
    s = max(0, start - pad)
    e = min(len(txt), start + pad)
    region = txt[s:e]
    for i in range(start - s, len(region)):
        if _BOUNDARY.match(region[i]):
            start = s + i + 1
            break
    # nudge end to previous boundary
    s2 = max(0, end - pad)
    e2 = min(len(txt), end + pad)
    region2 = txt[s2:e2]
    for i in range(end - s2 - 1, -1, -1):
        if _BOUNDARY.match(region2[i]):
            end = s2 + i + 1
            break
    if start >= end:
        return txt[max(0, start - pad):min(len(txt), end + pad)]
    return txt[start:end]

# ---------- NEW: stronger cleanup for catalog-programs sections ----------
_CATALOG_NAV_PATTERNS = [
    "University Catalog",
    "Go to Search",
    "Menu",
    "Close",
    "Print/Download Options",
    "Print this page.",
    "Send Page to Printer",
    "Download Page (PDF)",
    "Download PDF of the entire",
    "What can we help you find?",
    "View All Results",
    "University Twitter",
    "University Facebook",
    "University YouTube",
    "Search the catalog",
    "Submit",
    "Catalog A-Z Index",
    "Colleges and Schools",
    "Courses A-Z",
    "Faculty and Administration",
    "General Information",
    "Programs A-Z",
    "Undergraduate and Graduate Degree Requirements",
]

_HEADER_TOKEN_SET = {"course", "list", "code", "title", "credits"}

def _looks_like_course_header_line(s: str) -> bool:
    """
    True if a line is just combinations of 'Course/List/Code/Title/Credits'
    (possibly repeated).
    """
    if not s:
        return False
    tokens = re.split(r"\s+", s.strip())
    return all(t.lower() in _HEADER_TOKEN_SET for t in tokens)

def clean_catalog_text(text: str) -> str:
    """
    Catalog-specific cleanup:
    - remove left sidebar / chrome lines
    - collapse obvious header noise like repeated "Course List / Code / Title / Credits"
    - drop repeated section headings inside the same block
    - squash consecutive duplicate lines
    """
    lines = text.splitlines()
    cleaned = []

    SIDEBAR_NOISE_PHRASES = [
        "montclair state university",
        "university catalog",
        "search catalog",
        "catalog a-z index",
        "courses a-z",
        "programs a-z",
        "archives",
        "faculty and administration",
        "general information",
        "undergraduate and graduate degree requirements",
        "print/download options",
        "print this page",
        "download page (pdf)",
        "what can we help you find?",
        "view all results",
        "university twitter",
        "university facebook",
        "university youtube",
    ]

    ONE_PER_CHUNK_HEADINGS = {
        "program requirements",
        "required courses",
        "required core courses",
        "elective courses",
        "electives",
        "culminating experience",
        "total credits",
        "course list",
    }
    seen_headings = set()

    last_line = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        lower = line.lower()

        # 1) Kill sidebar / navigation noise
        if any(p in lower for p in SIDEBAR_NOISE_PHRASES):
            continue

        # 2) Normalize / collapse the "Course List Code Title Credits" header noise
        if "course list" in lower and "code" in lower and "title" in lower and "credits" in lower:
            line = "Course List"
            lower = line.lower()

        # 3) Only keep one copy of common section headings per chunk
        if lower in ONE_PER_CHUNK_HEADINGS:
            if lower in seen_headings:
                continue
            seen_headings.add(lower)

        # 4) Squash exact consecutive duplicates
        if last_line is not None and line == last_line:
            continue

        cleaned.append(line)
        last_line = line

    return "\n".join(cleaned)

# ---------- basic markdown parsing ----------
def list_markdown_files(processed_dir: str) -> List[Path]:
    p = Path(processed_dir)
    return [f for f in p.rglob("*.md") if not f.name.endswith("-pdf.md")]

def split_by_headings(md: str) -> List[Dict]:
    lines = md.splitlines()
    sections, current_heading_path, current_text = [], [], []
    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")

    def flush():
        if current_text:
            sections.append({
                "heading_path": current_heading_path.copy(),
                "text": "\n".join(current_text).strip()
            })

    for line in lines:
        m = heading_re.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            current_heading_path = current_heading_path[:level-1] + [f"H{level}: {title}"]
            current_text = []
        else:
            current_text.append(line)
    flush()
    return [s for s in sections if s["text"]]

def sliding_chunks(text: str, target: int, overlap: int) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= target:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + target
        chunk = safe_slice(text, start, end)
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

def chunk_doc(md_text: str, meta: dict) -> List[dict]:
    title  = meta.get("source_title", "") or ""
    url    = meta.get("source_url", "")
    bucket = meta.get("bucket", "") or ""
    tags   = meta.get("tags", [])

    # EXTRA: pre-normalize course lines for catalog program pages
    if bucket == "catalog-programs":
        # stronger structural cleanup + course normalization
        md_text = clean_catalog_text(md_text)
        md_text = normalize_catalog_courses(md_text)

    out_chunks = []
    sections = split_by_headings(md_text) or [{"heading_path": [], "text": md_text}]

    # Larger target size for catalog program pages so 'Required / Electives / Culminating'
    # blocks tend to stay together.
    base_target = TARGET_CHARS
    if bucket == "catalog-programs":
        base_target = 2200  # a bit bigger, but still manageable

    for s_idx, sec in enumerate(sections, 1):
        sec_text = sec["text"]
        heading_path = sec["heading_path"]

        # ---- if extracted-table markers exist, split per table block ----
        if "### Table " in sec_text:
            table_blocks = [b.strip() for b in sec_text.split("### Table ") if b.strip()]
            for i, block in enumerate(table_blocks, 1):
                piece = "### Table " + block
                cid = hashlib.sha1(
                    f"{url}::{';'.join(heading_path)}::table::{i}".encode("utf-8")
                ).hexdigest()
                out_chunks.append({
                    "id": f"sha1:{cid}",
                    "source_url": url,
                    "title": title,
                    "bucket": bucket,
                    "tags": tags,
                    "headings": heading_path,        # list
                    "section_heading": (heading_path[-1] if heading_path else ""),
                    "text": piece,
                    # re-rank signals (tables are tables)
                    "is_table": True,
                    "section": section_label(heading_path),
                    "term":    infer_term(piece, heading_path),
                })
            # skip normal sliding chunking for this section
            continue

        # ---- Sliding window for normal prose with short-section override ----
        local_target = base_target
        if len(sec_text) <= local_target * 2:
            # keep short/medium sections intact (no splitting)
            pieces = [sec_text]
        else:
            pieces = sliding_chunks(sec_text, local_target, OVERLAP_CHARS)
            
        for i, piece in enumerate(pieces, 1):
            cid = hashlib.sha1(
                f"{url}::{';'.join(heading_path)}::{i}".encode("utf-8")
            ).hexdigest()
            out_chunks.append({
                "id": f"sha1:{cid}",
                "source_url": url,
                "title": title,
                "bucket": bucket,
                "tags": tags,
                "headings": heading_path,                 # list
                "section_heading": (heading_path[-1] if heading_path else ""),
                "text": piece,
                # re-rank signals:
                "is_table": is_table_block(piece),
                "section": section_label(heading_path),
                "term":    infer_term(piece, heading_path),
            })
    return out_chunks

def _infer_bucket_from_url(url: str) -> str:
    u = (url or "").lower()
    if "/red-hawk-central/registrar/" in u:
        return "registrar"
    if "/policies/" in u:
        return "policies"
    if "/student-services/" in u:
        return "student-services"
    if "/global/" in u:
        return "oge"
    if "/school-of-computing/" in u:
        return "soc"
    if "/catalog.montclair.edu/" in u or "catalog.montclair.edu" in u:
        return "catalog-programs"
    return ""

def load_allowed_buckets(yaml_path: str = "sources.yaml") -> set[str]:
    """
    Reads allowed_buckets from sources.yaml. 
    Returns {'registrar'} if key/file is missing.
    Supports wildcard '*' to allow all buckets.
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ab = cfg.get("allowed_buckets")
        if not ab:
            logging.info("allowed_buckets not set in sources.yaml; defaulting to {'registrar'}.")
            return {"registrar"}
        if isinstance(ab, list):
            s = set([str(x).strip() for x in ab if str(x).strip()])
            if "*" in s:
                logging.info("allowed_buckets='*' → allowing all buckets.")
                return {"*"}  # wildcard sentinel
            return s
        logging.warning("allowed_buckets is not a list; defaulting to {'registrar'}.")
        return {"registrar"}
    except Exception as e:
        logging.warning(f"Could not read allowed_buckets from sources.yaml ({e}); defaulting to registrar.")
        return {"registrar"}

def main():
    setup_logger()
    
    global ALLOWED_BUCKETS
    ALLOWED_BUCKETS = load_allowed_buckets()
    logging.info(f"Allowed buckets for per-row table ingestion: {sorted(ALLOWED_BUCKETS)}")

    processed_dir = "data/processed"
    out_path = Path("rag/chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list_markdown_files(processed_dir)
    logging.info(f"Found {len(files)} Markdown files for chunking")

    total_main = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for idx, fpath in enumerate(files, 1):
            try:
                post = frontmatter.load(fpath)
                md_text = post.content or ""
                meta = {**post.metadata}

                # OLD: light cleanup based on source_url (still useful)
                src_url = meta.get("source_url", "")
                md_text = clean_markdown_text(md_text, src_url)

                chunks = chunk_doc(md_text, meta)
                t_cnt = sum(1 for c in chunks if c.get("is_table"))
                logging.debug(f"{fpath} → chunks={len(chunks)} (table_chunks={t_cnt})")

                total_main += len(chunks)
                for ch in chunks:
                    out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                if idx % 25 == 0:
                    logging.debug(f"Chunked {idx}/{len(files)} files; total_main_chunks={total_main}")
            except Exception as e:
                logging.warning(f"Skip {fpath}: {e}")

    logging.info(f"Main markdown chunks written: {total_main}")

    # ---- Append per-row mini-chunks (if extractor wrote them) ----
    row_files = Path("data/processed").rglob("*-table-rows.jsonl")
    row_count = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for rf in row_files:
            try:
                for line in rf.read_text(encoding="utf-8").splitlines():
                    obj  = json.loads(line)
                    text = obj.get("text", "")
                    meta = obj.get("meta", {}) or {}
                    url  = obj.get("url", "")
                    sec  = obj.get("section_heading", "Calendar")
                    bucket = meta.get("bucket") or _infer_bucket_from_url(url)

                    if "*" not in ALLOWED_BUCKETS and bucket not in ALLOWED_BUCKETS:
                        continue

                    cid = hashlib.sha1(f"{url}::row::{text}".encode("utf-8")).hexdigest()
                    out.write(json.dumps({
                        "id": f"sha1:{cid}",
                        "source_url": url,
                        "title": "Calendar Row",
                        "bucket": bucket,
                        "tags": [],
                        "headings": [sec],
                        "section_heading": sec,
                        "text": text,
                        "is_table": True,
                        "section": "calendar",
                        "term": meta.get("term",""),
                        "has_deadline_words": bool(meta.get("has_deadline_words", False)),
                    }, ensure_ascii=False) + "\n")
                    row_count += 1
            except Exception as e:
                logging.warning(f"Skip table rows from {rf}: {e}")

    logging.info(f"Per-row table chunks appended: {row_count}")
    logging.info(f"TOTAL chunks in rag/chunks.jsonl: {total_main + row_count}")

if __name__ == "__main__":
    main()
