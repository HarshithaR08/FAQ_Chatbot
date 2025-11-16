# Developer Name: Harshitha
# Description:
# Loads config sources.yaml
# Build URL list from: sitemaps (YAML + auto-discovered) + discovery + manual
#  - For each URL:
#      * If PDF: download file + write stub Markdown with metadata
#      * If HTML: do a CONDITIONAL fetch (ETag/Last-Modified) and skip unchanged
#                 else save raw HTML, convert to Markdown, write metadata
#  - Remember per-URL state (etag/last_modified/hash) in state/fetch_state.json
#  - Re-running will SKIP pages that didn’t change (fast, polite)
#  - New sub-sites found by discovery can auto-add their sitemaps

import os, csv, logging, time, urllib.parse, yaml, requests, json, hashlib, re
from datetime import datetime
from .fetch_from_sitemap import run_from_sources   # sitemap discovery
from .discovery_links import discover_links        # deep-link discovery
from .html_to_markdown import clean_html_to_markdown

# ---- OPTIONAL: catalog sectionizer import (splits /programs/ pages by #fragment) ----
try:
    from .catalog_sectionizer import split_catalog_sections
    _HAS_SECTIONIZER = True
except Exception as _e:
    split_catalog_sections = None
    _HAS_SECTIONIZER = False

USER_AGENT = "MSU-FAQ-Bot/0.1 (contact: ramakrishnah1@montclair.edu)"
# keep False by default, set True for a one-time clean rebuild
FORCE_REWRITE = True

# Added for Incremental Update Logic
# Uses { url: {etag, last_modified, content_sha1, last_fetch} } from state/fetch_state.json to skip unchanged pages and re-fetch only updated ones.
STATE_PATH = "state/fetch_state.json"  
def load_state():
    if os.path.exists(STATE_PATH):
        try:
            return json.load(open(STATE_PATH, "r"))
        except Exception:
            return {}
    return {}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

# Compute a fingerprint of the HTML we fetched (fallback when no 304 support)
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# Below function contains code to ask the server "has this page changed?" Sends If-None-Match (ETag) and/or If-Modified-Since (Last-Modified) if we have them.
# If server replies 304 => unchanged -> return (None, "not-modified", headers)
# If server replies 200 => OK (new body) new content -> return (text, "ok", headers)
def fetch_text_conditional(url: str, state: dict, timeout=60):
    headers = {"User-Agent": USER_AGENT}
    st = state.get(url, {})
    if st.get("etag"):
        headers["If-None-Match"] = st["etag"]
    if st.get("last_modified"):
        headers["If-Modified-Since"] = st["last_modified"]
    r = requests.get(url, headers=headers, timeout=timeout)
    if r.status_code == 304:
        # Not modified since last time -> skip work
        return None, "not-modified", r.headers
    r.raise_for_status()
    return r.text, "ok", r.headers

def path_from_url(base_dir: str, url: str, replace_ext: str | None = None) -> str:
    rel = urllib.parse.urlparse(url).path.lstrip("/")
    head, tail = os.path.split(rel)
    if not tail:
        tail = "index"
    if replace_ext is not None:
        stem, _ = os.path.splitext(tail)
        tail = stem + replace_ext
    full = os.path.join(base_dir, head, tail)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full

def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

def load_cfg(path="sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# write YAML front matter at the top of the Markdown file.
def write_front_matter_markdown(md_text: str, meta: dict) -> str:
    fm_lines = ["---"]
    for k, v in meta.items():
        fm_lines.append(f"{k}: {v!r}")
    fm_lines.append("---")
    return "\n".join(fm_lines) + "\n\n" + md_text

# Input rows: (url, lastmod, bucket, source). Remove duplicate URLs, keep first occurrence.
def dedupe_rows(rows):
    seen, out = set(), []
    for u, lm, b, src in rows:
        if u in seen:
            continue
        out.append((u, lm, b, src))
        seen.add(u)
    return out

# Read any sitemaps we auto-found in discovery and saved to state/last_run.json.
def load_autodiscovered_sitemaps(cfg):
    disc = cfg.get("discovery", {})
    state_path = disc.get("last_run_db") or "state/last_run.json"
    if os.path.exists(state_path):
        try:
            state = json.load(open(state_path, "r"))
            return state.get("autodiscovered_sitemaps", [])
        except Exception:
            return []
    return []



def save_url_csv(rows, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "lastmod", "bucket", "source"])
        w.writerows(rows)
    logging.info(f"Wrote URL list → {csv_path}")

# Added Fix for Rendering JS and Extracting Tables
def _match_prefix(url: str, prefixes: list[str]) -> bool:
    return any(url.startswith(p) for p in (prefixes or []))

def needs_render(url: str, cfg) -> bool:
    return _match_prefix(url, cfg.get("render_js_prefixes", []))

def needs_table_extract(url: str, cfg) -> bool:
    # 1) prefix match
    if any(url.startswith(p) for p in (cfg.get("table_extract_prefixes", []) or [])):
        return True
    # 2) regex patterns
    pats = [re.compile(rx) for rx in (cfg.get("table_extract_patterns", []) or [])]
    return any(p.search(url) for p in pats)

# -------- catalog section writer (for #requirementstext, #fouryearplantext, etc.) --------
def save_section_markdown(proc_dir: str, sec: dict, bucket: str, source_url: str):
    """
    Writes a section to a .md file whose name includes the fragment, e.g.
    .../programs/computer-science-ms/index__requirementstext.md
    """
    url = sec["url"]
    base_url, frag = url.split("#", 1) if "#" in url else (url, None)

    # base path like .../programs/computer-science-ms/index.md
    base_out_md_path = path_from_url(proc_dir, base_url, replace_ext=".md")
    # turn it into .../index__fragment.md if there is a fragment
    if frag:
        out_md_path = os.path.splitext(base_out_md_path)[0] + f"__{frag}.md"
    else:
        out_md_path = base_out_md_path

    meta = {
        "source_url": url,
        "source_title": sec.get("title") or "",
        "last_fetched": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "lastmod": "",
        "bucket": bucket,
        "tags": [bucket, "catalog-section"],
    }
    md_body = sec.get("text") or ""
    # Prefix the human title so greps and RAG show the section header explicitly
    title_line = f"## {sec.get('title') or ''}\n\n" if sec.get('title') else ""
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(write_front_matter_markdown(title_line + md_body, meta))
    return out_md_path

# ---------- main ----------
def main():
    setup_logger()
    cfg = load_cfg()
    state = load_state()

    raw_dir  = cfg["paths"]["raw_dir"]
    proc_dir = cfg["paths"]["processed_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    sitemaps = list(dict.fromkeys(cfg["sitemaps"] + load_autodiscovered_sitemaps(cfg)))
    logging.info("Using %d sitemaps (yaml+auto)", len(sitemaps))
    for sm in sitemaps[:10]:
        logging.debug(" - %s", sm)

    sitemap_rows_raw = run_from_sources(
        sitemaps=sitemaps,
        include_prefixes=cfg["include_prefixes"],
        csv_out=cfg["paths"]["url_list_csv"],
        cfg=cfg,
    )
    sitemap_rows = [(u, lm, b, "sitemap") for (u, lm, b) in sitemap_rows_raw]

    discovered_raw = discover_links(cfg)
    discovered_rows = [(u, lm, b, "discovered") for (u, lm, b) in discovered_raw]

    manual = cfg.get("manual", []) or []
    manual_rows = [(u, "", "manual", "manual") for u in manual]

    merged = dedupe_rows(sitemap_rows + discovered_rows + manual_rows)

    # ---- Sanity guard: required key pages must be present
    REQUIRED = {
        "https://www.montclair.edu/school-of-computing/graduate/",
        "https://www.montclair.edu/school-of-computing/undergraduate-students/",
        "http://catalog.montclair.edu/coursesaz/csit/",
    }
    merged_urls = {u for (u, _, _, _) in merged}
    missing = sorted(u for u in REQUIRED if u not in merged_urls)
    if missing:
        raise SystemExit(f"❌ Missing required URLs: {missing}")

    save_url_csv(merged, cfg["paths"]["url_list_csv"])

    logging.info(
        "URL counts → sitemap=%d, discovered=%d, manual=%d, merged=%d",
        len(sitemap_rows), len(discovered_rows), len(manual_rows), len(merged)
    )
    logging.info("First 15 URLs to be fetched:")
    for u, _, b, src in merged[:15]:
        logging.info(f" - [{b}|{src}] {u}")

    success, fail = 0, 0

    for i, (url, lastmod, bucket, source) in enumerate(merged, start=1):
        try:
            # -------- PDF branch --------
            if url.lower().endswith(".pdf"):
                pdf_path = path_from_url(raw_dir, url, replace_ext=".pdf")
                with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=40, stream=True) as r:
                    r.raise_for_status()
                    with open(pdf_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                meta = {
                    "source_url": url,
                    "source_title": "",
                    "last_fetched": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "lastmod": lastmod or "",
                    "bucket": bucket,
                    "tags": [bucket, "pdf"],
                }
                out_md_path = path_from_url(proc_dir, url, replace_ext="-pdf.md")
                with open(out_md_path, "w", encoding="utf-8") as f:
                    f.write(write_front_matter_markdown("(PDF content not yet extracted)", meta))
                state[url] = {
                    "etag": None,
                    "last_modified": None,
                    "content_sha1": None,
                    "last_fetch": datetime.utcnow().isoformat(timespec="seconds") + "Z"
                }
                logging.info(f"[{i}/{len(merged)}]  [pdf|{bucket}|{source}] {url}")
                success += 1
                time.sleep(0.2)
                continue

            # -------- HTML branch --------
            html, status, hdrs = fetch_text_conditional(url, state)
            if status == "not-modified" and not FORCE_REWRITE:
                logging.info(f"[{i}/{len(merged)}] ↺ SKIP (304) [{bucket}|{source}] {url}")
                continue
            if status == "not-modified" and FORCE_REWRITE:
                html = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60).text

            # prerender if needed
            if needs_render(url, cfg):
                try:
                    from .render_fetch import fetch_rendered_html
                    html = fetch_rendered_html(url)
                except Exception as e:
                    logging.warning(f"Render fetch failed, fallback to raw HTML for {url} — {e}")
                    if html is None:
                        html = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60).text
            if html is None:
                html = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60).text

            # --- SPECIAL HANDLING: split catalog program pages into section docs ---
            if _HAS_SECTIONIZER and ("catalog.montclair.edu" in url) and ("/programs/" in url):
                try:
                    sections = split_catalog_sections(html, url)
                    # NEW RULE:
                    # Write sections whenever we have at least 2 chunks
                    # (Overview + at least one anchor like #requirementstext or #fouryearplantext).
                    if not sections or len(sections) < 2:
                        logging.info(
                            f"[sectionizer] Only {len(sections) if sections else 0} chunk(s) for {url} → "
                            "use normal processing."
                        )
                    else:
                        # save raw HTML once for completeness
                        raw_path = path_from_url(raw_dir, url, replace_ext=".html")
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(html)

                        written = []
                        for sec in sections:
                            out_md_path = save_section_markdown(
                                proc_dir, sec, bucket="catalog-programs", source_url=url
                            )
                            written.append(out_md_path)
                        
                        # record state for the base URL based on the derived files we wrote
                        md_hash = sha1("\n".join([open(p, "r", encoding="utf-8").read() for p in written]) if written else "")
                        state[url] = {
                            "etag": hdrs.get("ETag"),
                            "last_modified": hdrs.get("Last-Modified"),
                            "content_sha1": md_hash,
                            "last_fetch": datetime.utcnow().isoformat(timespec="seconds") + "Z"
                        }

                        logging.info(
                            f"[{i}/{len(merged)}] sectionized {len(sections)} sections "
                            f"[{bucket}|{source}] {url}"
                        )
                        success += 1
                        time.sleep(0.2)
                        continue  # IMPORTANT: skip the normal processing path

                except Exception as e:
                    logging.warning(f"Sectionize failed for {url} — {e} (falling back to normal processing)")

            
            # ---- extract tables FIRST (operate on raw HTML) ----
            table_md_path = None
            table_json_path = None
            tables_found = 0
            if needs_table_extract(url, cfg):
                try:
                    from .extract_tables import html_tables_to_markdown_and_json
                    base_md_path    = path_from_url(proc_dir, url, replace_ext=None)
                    table_md_path   = os.path.splitext(base_md_path)[0] + "-tables.md"
                    table_json_path = os.path.splitext(base_md_path)[0] + "-tables.json"
                    logging.info(f"[tables] extracting → {url}")
                    html_tables_to_markdown_and_json(html, table_md_path, table_json_path, cfg, url)
                    if os.path.exists(table_json_path):
                        try:
                            tables_found = int((json.load(open(table_json_path,"r")) or {}).get("tables_found", 0))
                        except Exception:
                            tables_found = 0
                except Exception as e:
                    logging.warning(f"Table extract failed for {url} — {e}")

            # ---- HTML → Markdown (after extraction) ----
            md_body = clean_html_to_markdown(html)

            # append extracted-table markdown if present
            if table_md_path and os.path.exists(table_md_path):
                md_body += "\n\n<!-- extracted tables below -->\n" + open(table_md_path, "r", encoding="utf-8").read()
                logging.info(f"[tables] appended markdown → {table_md_path}")
            else:
                logging.info(f"[tables] no table markdown written for → {url}")

            # ---- FALLBACK: if extractor found zero HTML tables, parse markdown pipe-tables ----
            if tables_found == 0:
                logging.info(f"[md-fallback] scanning markdown for pipe tables → {url}")
                import re as _re, io as _io, pandas as _pd
                heading_re = _re.compile(r"^(#{1,6})\s+(.*)$")
                lines = md_body.splitlines()
                out_lines = []
                current_heading_path = []
                i2 = 0

                def _scan_md_table(start: int):
                    if start+1 >= len(lines): return None, start
                    hdr = lines[start].strip()
                    sep = lines[start+1].strip()
                    if ("|" not in hdr) or ("|" not in sep) or (not _re.search(r"\|\s*:?-{3,}", sep)):
                        return None, start
                    rows = [hdr, sep]
                    j = start + 2
                    while j < len(lines) and ("|" in lines[j]):
                        rows.append(lines[j])
                        j += 1
                    return rows, j

                def _infer_term_from_context():
                    joined = " — ".join([h.split(": ",1)[-1] for h in current_heading_path[-3:]])
                    m = _re.search(r"\b(Spring|Summer|Fall|Winter)\s+20\d{2}\b", joined, _re.I)
                    return m.group(0).title() if m else ""

                # optional per-row JSONL like extractor
                base_md_path = path_from_url(proc_dir, url, replace_ext=None)
                row_jsonl_path = os.path.splitext(base_md_path)[0] + "-table-rows.jsonl"
                row_jsonl = []

                while i2 < len(lines):
                    m = heading_re.match(lines[i2])
                    if m:
                        level = len(m.group(1)); title = m.group(2).strip()
                        current_heading_path = current_heading_path[:level-1] + [f"H{level}: {title}"]
                        out_lines.append(lines[i2]); i2 += 1; continue

                    table_rows, j2 = _scan_md_table(i2)
                    if table_rows:
                        section = (current_heading_path[-1].split(": ",1)[-1] if current_heading_path else "")
                        term = _infer_term_from_context()
                        out_lines.append("")
                        out_lines.append("### Table (md-fallback)")
                        if section: out_lines.append(f"**SECTION:** {section}")
                        if term:    out_lines.append(f"**TERM:** {term}")
                        out_lines.extend(table_rows)
                        out_lines.append("")
                        # per-row jsonl
                        try:
                            df = _pd.read_csv(_io.StringIO("\n".join(table_rows)), sep=r"\s*\|\s*", engine="python").dropna(axis=1, how="all")
                            headers = [str(c) for c in df.columns]
                            for row in df.astype(str).values.tolist():
                                text = " | ".join(row)
                                row_jsonl.append({
                                    "type": "table_row",
                                    "url": url,
                                    "section_heading": section or "Calendar",
                                    "text": text,
                                    "meta": {
                                        "table": True,
                                        "bucket": "",
                                        "term": term,
                                        "session": None,
                                        "headers": headers,
                                        "has_deadline_words": True
                                    }
                                })
                        except Exception:
                            pass
                        i2 = j2; continue

                    out_lines.append(lines[i2]); i2 += 1

                md_body = "\n".join(out_lines)
                if row_jsonl:
                    with open(row_jsonl_path, "w", encoding="utf-8") as f:
                        for obj in row_jsonl:
                            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    logging.info(f"[md-fallback] wrote row chunks → {row_jsonl_path}")

            # hash AFTER md_body finalized
            md_hash = sha1(md_body)
            if (state.get(url, {}).get("content_sha1") == md_hash) and not FORCE_REWRITE:
                logging.info(f"[{i}/{len(merged)}] ↺ SKIP (unchanged content) [{bucket}|{source}] {url}")
                continue

            # write raw HTML + processed markdown
            raw_path = path_from_url(raw_dir, url, replace_ext=".html")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(html)

            meta = {
                "source_url": url,
                "source_title": "",
                "last_fetched": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "lastmod": lastmod or "",
                "bucket": bucket,
                "tags": [bucket],
            }
            out_md_path = path_from_url(proc_dir, url, replace_ext=".md")
            with open(out_md_path, "w", encoding="utf-8") as f:
                f.write(write_front_matter_markdown(md_body, meta))

            state[url] = {
                "etag": hdrs.get("ETag"),
                "last_modified": hdrs.get("Last-Modified"),
                "content_sha1": md_hash,
                "last_fetch": datetime.utcnow().isoformat(timespec="seconds") + "Z"
            }

            logging.info(f"[{i}/{len(merged)}] UPDATED [{bucket}|{source}] {url}")
            success += 1
            time.sleep(0.3)

        except Exception as e:
            logging.warning(f"[{i}/{len(merged)}] {url} — {e}")
            fail += 1

    save_state(state)
    logging.info(f"Done. Success: {success}, Failed: {fail}")
    logging.info(f"Raw → {raw_dir} | Markdown → {proc_dir}")
    logging.info(f"TOTAL URLs fetched and converted: {success}")


if __name__ == "__main__":
    main()