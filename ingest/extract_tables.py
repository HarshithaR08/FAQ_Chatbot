# Developer: Harshitha
# Parse all HTML tables and save BOTH Markdown (for RAG) and JSON (for exact answers).

import os
import json
import re
import io
from pathlib import Path
from collections import Counter

from bs4 import BeautifulSoup
import pandas as pd

# -------- small helpers --------

DEADLINE_WORDS = re.compile(
    r"(withdraw|final day|last day|drop|deadline|add\/?drop|adjustment)",
    re.IGNORECASE,
)


def _normalize_html(html: str) -> str:
    """Normalize <br> and &nbsp; etc. so pandas/bs4 can parse cells better."""
    # Convert <br> to a neutral separator
    html = re.sub(r"(?i)<br\s*/?>", " / ", html)
    # Normalize non-breaking space
    html = html.replace("&nbsp;", " ")
    # Remove multiple spaces
    html = re.sub(r"[ \t]{2,}", " ", html)
    return html


def parse_catalog_course_blocks(soup: BeautifulSoup):
    """Special helper for catalog course blocks (.courseblock)."""
    out = []
    for blk in soup.select(".courseblock"):
        title = blk.select_one(".courseblocktitle")
        desc = blk.select_one(".courseblockdesc")
        prereq = None
        # Prereq sometimes sits in a <p> or as text; quick fallback:
        for p in blk.select("p"):
            txt = p.get_text(" ", strip=True)
            if txt.lower().startswith("prerequisite"):
                prereq = txt
                break
        out.append(
            {
                "code_title": title.get_text(" ", strip=True) if title else None,
                "description": desc.get_text(" ", strip=True) if desc else None,
                "prerequisites": prereq,
            }
        )
    return out


def _bs4_table_to_rows(table, page_url: str):
    """Extract rows as 'cell1 | cell2 | ...' strings using bs4 only."""
    rows_out = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        txts = []
        for c in cells:
            # text with links collapsed to text; keep inner <br> already normalized
            t = " ".join(c.get_text(" ", strip=True).split())
            txts.append(t)
        if txts:
            row_text = " | ".join(txts)
            rows_out.append(row_text)
    return rows_out


def _row_to_kv_text(headers, row_vals):
    """
    Convert a single table row into key–value markdown:

        **Start Date:** 12/23
        **End Date:** 1/15/2025

    Skips empty cells.
    """
    headers = headers or []
    row_vals = row_vals or []
    parts = []
    for h, v in zip(headers, row_vals):
        h = (h or "").strip().rstrip(":")
        v = str(v or "").strip()
        if not h or not v:
            continue
        parts.append(f"**{h}:** {v}")
    return "\n".join(parts)


def _emit_rows(rows, out_rows_path, section_heading, term_guess, bucket, page_url):
    """Append mini row-chunks to a JSONL file (currently unused)."""
    os.makedirs(os.path.dirname(out_rows_path), exist_ok=True)
    with open(out_rows_path, "a", encoding="utf-8") as rf:
        for r in rows:
            meta = {
                "bucket": bucket,
                "term": term_guess or "",
                "section_heading": section_heading or "",
                "has_deadline_words": bool(DEADLINE_WORDS.search(r)),
            }
            rf.write(
                json.dumps(
                    {
                        "url": page_url,
                        "text": r,
                        "meta": meta,
                        "section_heading": section_heading or "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def most_common_normalized(items):
    if not items:
        return None

    def norm(x):
        x = (x or "").strip()
        return x[:1].upper() + x[1:].lower() if x else x

    items = [norm(i) for i in items if i]
    return Counter(items).most_common(1)[0][0]


def _compile_many(cfg, key):
    return [re.compile(rx, re.IGNORECASE) for rx in (cfg.get(key) or [])]


TERM_FALLBACK_RE = re.compile(
    r"\b(Spring|Summer|Fall|Winter)\s+20\d{2}\b", re.IGNORECASE
)


def _guess_term_from_text(text: str):
    m = TERM_FALLBACK_RE.search(text or "")
    return m.group(0).title() if m else ""


def _scan_months_from_df(df) -> set[int]:
    months = set()
    for val in df.astype(str).values.ravel():
        m = re.search(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b", val)
        if m:
            mm = int(m.group(1))
            if 1 <= mm <= 12:
                months.add(mm)
    return months


def _infer_season_from_months(months: set[int]) -> str:
    if not months:
        return ""
    counts = {
        "Fall": len({9, 10, 11} & months),
        "Spring": len({2, 3, 4, 5} & months),
        "Summer": len({6, 7, 8} & months),
        "Winter": len({12, 1} & months),
    }
    priority = ["Fall", "Spring", "Summer", "Winter"]
    best = max(priority, key=lambda s: counts[s])
    return best if counts[best] > 0 else ""


def _nearest_h2_h3_with_term(table_tag):
    """Find nearest h2/h3 around the table whose text contains a term like 'Fall 2025'."""
    def first_match(hlist):
        for h in hlist:
            txt = h.get_text(" ", strip=True)
            if TERM_FALLBACK_RE.search(txt or ""):
                return txt
        return ""

    prev = table_tag.find_all_previous(["h2", "h3"], limit=6)
    if prev:
        t = first_match(prev)
        if t:
            return t

    nxt = table_tag.find_all_next(["h2", "h3"], limit=6)
    if nxt:
        t = first_match(nxt)
        if t:
            return t

    return ""


def _detect_term_and_session(text_segments, cfg):
    termrx = _compile_many(cfg, "term_patterns")
    sessrx = _compile_many(cfg, "session_patterns")
    terms, sessions = [], []
    for seg in text_segments:
        s = seg or ""
        for r in termrx:
            terms += r.findall(s)
        for r in sessrx:
            sessions += r.findall(s)
    return most_common_normalized(terms), most_common_normalized(sessions)


def _looks_like_deadline_table(headers: list[str], cfg) -> bool:
    hdr = " | ".join([h or "" for h in headers]).lower()
    pats = _compile_many(cfg, "deadline_header_patterns")
    # 1) If YAML patterns exist, use them
    if pats:
        for r in pats:
            if r.search(hdr):
                return True
    # 2) Fallback heuristic if config is missing or too narrow
    if "last day" in hdr and "withdraw" in hdr:
        return True
    if "add/drop" in hdr or "add or drop" in hdr:
        return True
    if "withdrawal period begins" in hdr:
        return True
    if "waitlist ends" in hdr:
        return True
    return False


def _find_column(headers, candidates):
    hdrs = [str(h or "").lower() for h in headers]
    for i, h in enumerate(hdrs):
        for c in candidates:
            if c in h:
                return i
    return None


# -------- main entry --------


def html_tables_to_markdown_and_json(
    html: str, out_md: Path, out_json: Path, cfg: dict, url: str
):
    """
    Writes:
      - out_md:   appended human-readable tables (with SECTION and TERM markers)
      - out_json: summary with normalized rows
      - <base>-table-rows.jsonl : one JSON per calendar row (mini-chunks)
    """
    import logging

    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.info("deadline_header_patterns = %s", cfg.get("deadline_header_patterns"))

    # Normalize HTML so <br> and &nbsp; don’t break cell boundaries
    html = _normalize_html(html)

    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    md_parts = []
    all_raw_rows = []
    normalized_calendar = []
    row_jsonl = []
    tables_found = 0  # count only the tables we actually handle below

    for idx, t in enumerate(tables, 1):
        # 1) nearest heading that contains a term like "Fall 2025"
        heading_text = _nearest_h2_h3_with_term(t)
        term_from_heading = _guess_term_from_text(heading_text)

        # 2) parse table: try multiple pandas parsers
        df = None
        for flavor in (["lxml"], ["bs4"], ["html5lib"]):
            try:
                dfs = pd.read_html(io.StringIO(str(t)), flavor=flavor)
                if dfs:
                    df = dfs[0]
                    break
            except Exception:
                continue

        # ---- Fallback if pandas couldn't parse ----
        if df is None:
            rows = _bs4_table_to_rows(t, page_url=url)
            if not rows:
                # truly nothing to do
                continue

            # Attempt to extract a term from either rows or the heading
            term_text = _guess_term_from_text(" ".join(rows)) or term_from_heading

            # Minimal markdown output for this table
            md_parts.append(f"\n\n### Table {idx}")
            if heading_text:
                md_parts.append(f"**SECTION:** {heading_text}")
            if term_text:
                md_parts.append(f"**TERM:** {term_text}")

            # Try to render as pipe table if the first row looks like headers
            header_cells = rows[0].split(" | ")
            looks_like_header = (
                1 <= len(header_cells) <= 12 and all(len(h) <= 80 for h in header_cells)
            )

            if looks_like_header:
                md_parts.append("| " + " | ".join(header_cells) + " |")
                md_parts.append(
                    "| " + " | ".join(["---"] * len(header_cells)) + " |"
                )
                body = rows[1:]
                headers = header_cells
            else:
                body = rows
                headers = []

            for r in body:
                if " | " in r:
                    md_parts.append(
                        "| "
                        + " | ".join([c.strip() for c in r.split("|")])
                        + " |"
                    )
                else:
                    md_parts.append(r.strip())

            # Emit per-row JSONL (mini-chunks) as key–value lists
            for r in body:
                row_cells = (
                    [c.strip() for c in r.split("|")] if " | " in r else [r.strip()]
                )
                kv_text = _row_to_kv_text(headers, row_cells) if headers else r.strip()

                # keep a tiny markdown version around if you want it later
                if headers:
                    header_line = "| " + " | ".join(headers) + " |"
                    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                    row_line = "| " + " | ".join(row_cells) + " |"
                    full_md = "\n".join([header_line, sep_line, row_line])
                else:
                    full_md = r.strip()

                row_jsonl.append(
                    {
                        "type": "table_row",
                        "url": url,
                        "section_heading": (heading_text or "Calendar"),
                        "text": kv_text,
                        "meta": {
                            "table": True,
                            "bucket": "",  # filled in by chunker via URL inference
                            "term": (term_text or _guess_term_from_text(r)),
                            "session": None,
                            "headers": headers,
                            "full_md": full_md,
                            "has_deadline_words": bool(DEADLINE_WORDS.search(r)),
                        },
                    }
                )

            tables_found += 1
            continue  # handled this table via fallback

        # ---- Pandas parsed a DataFrame; proceed as before ----
        all_raw_rows.extend(df.to_dict(orient="records"))

        # 3) infer season/year from months and any explicit years
        months = _scan_months_from_df(df)
        season_from_dates = _infer_season_from_months(months)

        term_text = term_from_heading
        year_in_table = None
        for val in df.astype(str).values.ravel():
            m = re.search(r"\b(20\d{2})\b", val)
            if m:
                year_in_table = m.group(1)
                break

        m = re.search(r"\b(20\d{2})\b", term_from_heading or "")
        year_in_heading = m.group(1) if m else ""

        if season_from_dates:
            year = year_in_table or year_in_heading or ""
            term_text = f"{season_from_dates} {year}".strip()

        # 4) header list
        headers = [str(c) for c in df.columns]

        # 5) TERM/SECTION markers + markdown
        md_parts.append(f"\n\n### Table {idx}")
        if heading_text:
            md_parts.append(f"**SECTION:** {heading_text}")
        if term_text:
            md_parts.append(f"**TERM:** {term_text}")
        md_parts.append(df.to_markdown(index=False))

        # 6) normalized calendar rows (lightweight schema)
        for row in df.to_dict(orient="records"):
            lower = {str(k).strip().lower(): str(v).strip() for k, v in row.items()}
            term_val = (
                term_text
                or lower.get("term")
                or lower.get("part of term")
                or lower.get("part of term & description")
            )
            last_add = (
                lower.get("last day to add/drop (100% adjustment)")
                or lower.get("last day to add/drop")
                or lower.get("last day to add or drop 100%")
            )
            final_withdraw = (
                lower.get("final day to withdraw")
                or lower.get("final day to withdraw classes (50% adjustment)")
                or lower.get("final day to withdraw classes  (50% adjustment)")
                or lower.get("final day to wd")
            )
            if term_val or final_withdraw:
                normalized_calendar.append(
                    {
                        "term": term_val,
                        "last_day_to_add_drop": last_add,
                        "final_day_to_withdraw": final_withdraw,
                    }
                )

        # 7) per-row mini-chunks if this *looks* like a deadline/calendar table
        if _looks_like_deadline_table(headers, cfg):
            idx_desc = _find_column(
                headers,
                [
                    "part of term",
                    "term & description",
                    "description",
                    "part of term & description",
                ],
            )
            idx_final = _find_column(
                headers,
                [
                    "final day to withdraw",
                    "final day to withdraw classes",
                    "final day to wd",
                ],
            )
            idx_wait = _find_column(headers, ["waitlist ends"])

            for row in df.astype(str).values.tolist():
                # normalize row cells as strings
                row_vals = [str(v).strip() for v in row]

                # detect row-local term/session using cfg patterns
                row_text = " | ".join(row_vals)
                row_term, row_session = _detect_term_and_session(
                    [row_text, heading_text, term_text], cfg
                )
                row_term = row_term or term_text

                # Tiny markdown version (kept in meta) + KV text for retrieval
                header_line = "| " + " | ".join(headers) + " |"
                sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                row_line = "| " + " | ".join(row_vals) + " |"
                full_md = "\n".join([header_line, sep_line, row_line])

                kv_text = _row_to_kv_text(headers, row_vals)

                row_jsonl.append(
                    {
                        "type": "table_row",
                        "url": url,
                        "section_heading": (heading_text or "Calendar"),
                        "text": kv_text,
                        "meta": {
                            "table": True,
                            "bucket": "",  # fill at ingestion if you store it
                            "term": row_term,
                            "session": row_session,
                            "final_withdraw": (
                                row_vals[idx_final]
                                if (idx_final is not None and idx_final < len(row_vals))
                                else None
                            ),
                            "waitlist_ends": (
                                row_vals[idx_wait]
                                if (idx_wait is not None and idx_wait < len(row_vals))
                                else None
                            ),
                            "desc": (
                                row_vals[idx_desc]
                                if (idx_desc is not None and idx_desc < len(row_vals))
                                else None
                            ),
                            "headers": headers,
                            "full_md": full_md,
                            "has_deadline_words": True,
                        },
                    }
                )

        # Count this pandas-handled table
        tables_found += 1

    # 8) write outputs
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n\n".join(md_parts), encoding="utf-8")

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "source": "table_extractor",
                "tables_found": int(tables_found),
                "rows": all_raw_rows,
                "normalized_calendar": normalized_calendar,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # per-row mini-chunks JSONL alongside the other outputs
    row_path = Path(str(out_json).replace("-tables.json", "-table-rows.jsonl"))
    with row_path.open("w", encoding="utf-8") as f:
        for obj in row_jsonl:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
