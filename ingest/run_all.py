# Developer Name: Harshitha
# Desccription: 
# Loads config sources.yaml
# Build URL list from: sitemaps (YAML + auto-discovered) + discovery + manual
#  - For each URL:
#      * If PDF: download file + write stub Markdown with metadata
#      * If HTML: do a CONDITIONAL fetch (ETag/Last-Modified) and skip unchanged
#                 else save raw HTML, convert to Markdown, write metadata
#  - Remember per-URL state (etag/last_modified/hash) in state/fetch_state.json
#  - Re-running will SKIP pages that didn’t change (fast, polite)
#  - New sub-sites found by discovery can auto-add their sitemaps


import os, csv, logging, time, urllib.parse, yaml, requests, json, hashlib
from datetime import datetime
from fetch_from_sitemap import run_from_sources   # sitemap discovery
from discovery_links import discover_links        # deep-link discovery
from html_to_markdown import clean_html_to_markdown


USER_AGENT = "MSU-FAQ-Bot/0.1 (contact: ramakrishnah1@montclair.edu)"
# by default keep FORCE_REWRITE = False, and only temporarily flip it to True when you want to force a clean rebuild (for example, right after deleting data/processed/)
FORCE_REWRITE = False

# CHANGE (Incremental Update Logic):
# Uses Last-Modified, ETag, or hash comparison from state/fetch_state.json
# to skip unchanged pages and re-fetch only updated ones.
STATE_PATH = "state/fetch_state.json"  # stores { url: {etag, last_modified, raw_sha1, last_fetch} }


# CHANGE (Incremental Update Logic):
# Load previously saved per-URL headers and content hashes so we can skip unchanged pages.
# Load/save a small JSON that remembers per-URL info (etag, last_modified, hash) if present.
def load_state():
    if os.path.exists(STATE_PATH):
        try:
            return json.load(open(STATE_PATH, "r"))
        except Exception:
            return {}
    return {}


# CHANGE (Incremental Update Logic):
# Save the current tracking information (state) about each URL (per-URL) into a file on disk.
# This function saves crawler’s progress so next time it runs, it knows which pages have already been processed
def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

# Compute a fingerprint of the HTML we fetched (fallback when no 304 support)
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# CHANGE (Incremental Update Logic):
# Conditional GET: sends If-None-Match / If-Modified-Since using saved ETag/Last-Modified.
# Below function contains code to ask the server "has this page changed?"
# Sends If-None-Match (ETag) and/or If-Modified-Since (Last-Modified) if we have them.
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


# Build a file path under base_dir that mirrors the URL path.
# Ensures parent directory exists.
# Optionally change extension via replace_ext ('.html', '.md', '-pdf.md', '.pdf').
def path_from_url(base_dir: str, url: str, replace_ext: str | None = None) -> str:
    rel = urllib.parse.urlparse(url).path.lstrip("/")  # e.g., policies/wp-content/.../file.pdf
    head, tail = os.path.split(rel)
    # If the URL ends with '/', os.path.split gives tail == ''
    # use 'index' as the filename.
    if not tail:
        tail = "index"  # <- ensure a filename
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

def main():
    setup_logger()
    cfg = load_cfg()
    # Load last-run state (per-URL etag/last_modified/hash), so we can skip unchanged.
    state = load_state()
    # Prepare output folders
    raw_dir = cfg["paths"]["raw_dir"]
    proc_dir = cfg["paths"]["processed_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    # CHANGE (Adaptability Enhancement):
    # Merge sitemaps from YAML with any auto-discovered sitemaps (saved by discovery_links.py)
    # so new sub-sites (e.g., catalog) are picked up automatically on future runs.
    sitemaps = list(cfg["sitemaps"])
    sitemaps += load_autodiscovered_sitemaps(cfg)   # include auto-found sitemaps (e.g., catalog)
    sitemaps = list(dict.fromkeys(sitemaps))        # dedupe, keep order
    logging.info("Using %d sitemaps (yaml+auto)", len(sitemaps))
    for sm in sitemaps[:10]:
        logging.debug(" - %s", sm)

    # Collect URLs with bucket rules (returns list of Sitemaps → (url,lastmod,bucket,'sitemap'))
    sitemap_rows_raw = run_from_sources(
        sitemaps=sitemaps, # using merged list here 
        include_prefixes=cfg["include_prefixes"],
        csv_out=cfg["paths"]["url_list_csv"],  # overwrite with merged CSV below
        cfg=cfg,
    )
    sitemap_rows = [(u, lm, b, "sitemap") for (u, lm, b) in sitemap_rows_raw]

    # # DISCOVERY → (url,'',bucket)
    discovered_raw = discover_links(cfg)
    discovered_rows = [(u, lm, b, "discovered") for (u, lm, b) in discovered_raw]

    # MANUAL → (url,'','manual','manual')
    manual = cfg.get("manual", []) or []
    manual_rows = [(u, "", "manual", "manual") for u in manual]

    # Merge → de-duplicate by URL → save preview CSV
    merged = dedupe_rows(sitemap_rows + discovered_rows + manual_rows)
    save_url_csv(merged, cfg["paths"]["url_list_csv"])

    logging.info(
        "URL counts → sitemap=%d, discovered=%d, manual=%d, merged=%d",
        len(sitemap_rows), len(discovered_rows), len(manual_rows), len(merged)
    )

    # Preview
    logging.info("First 15 URLs to be fetched:")
    for u, _, b, src in merged[:15]:
        logging.info(f" - [{b}|{src}] {u}")

    # FETCH LOOP (PDF branch unchanged, HTML branch now conditional + hash)
    success, fail = 0, 0
    for i, (url, lastmod, bucket, source) in enumerate(merged, start=1):
        try:
            # PDF branch (optional record headers)
            if url.lower().endswith(".pdf"):
                pdf_path = path_from_url(raw_dir, url, replace_ext=".pdf")
                with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=40, stream=True) as r:
                    r.raise_for_status()
                    with open(pdf_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                # Create a stub .md so the PDF is discoverable in RAG.
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
                # OPTIONAL: remember headers for PDFs, so a later run can skip unchanged PDF
                # CHANGE (Incremental Update Logic):
                # Remember PDF ETag/Last-Modified so we can skip re-downloading if unchanged later.
                state[url] = {
                    "etag": r.headers.get("ETag"),
                    "last_modified": r.headers.get("Last-Modified"),
                    "raw_sha1": None,  # not computing a hash for binary here
                    "last_fetch": datetime.utcnow().isoformat(timespec="seconds") + "Z"
                }
                logging.info(f"[{i}/{len(merged)}] ✅ [pdf|{bucket}|{source}] {url}")
                success += 1
                time.sleep(0.2)
                continue

            # CHANGE (Incremental Update Logic):
            # Step 1: Ask the server if the page changed since last run (ETag/Last-Modified).
            html, status, hdrs = fetch_text_conditional(url, state)

            # CHANGE (Incremental Update Logic):
            # Step 2: If 304 Not Modified, skip — unless FORCE_REWRITE is True.
            if status == "not-modified" and not FORCE_REWRITE:
                logging.info(f"[{i}/{len(merged)}] ↺ SKIP (304) [{bucket}|{source}] {url}")
                continue
            # If we forced a rewrite but got 304 (server didn’t send body),
            # do a normal GET to fetch the HTML so we can re-write files.
            if status == "not-modified" and FORCE_REWRITE:
                html = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60).text

            # CHANGE (Incremental Update Logic):
            # Step 3: Even if server returned 200, skip if the cleaned Markdown hash hasn't changed.
            md_body = clean_html_to_markdown(html)   # already in your code
            content_hash = sha1(md_body)             # hash the cleaned content
            if (state.get(url, {}).get("content_sha1") == content_hash) and not FORCE_REWRITE:
                logging.info(f"[{i}/{len(merged)}] ↺ SKIP (unchanged content) [{bucket}|{source}] {url}")
                continue

            # CHANGE (Incremental Update Logic):
            # Step 4: Save fresh headers + content hash so next run can skip if unchanged.
            raw_path = path_from_url(raw_dir, url, replace_ext=".html")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(html)

            # 5) Convert to Markdown and write front-matter.
            md_body = clean_html_to_markdown(html)
            meta = {
                "source_url": url,
                "source_title": "",  # optional: parse <title>
                "last_fetched": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "lastmod": lastmod or "",
                "bucket": bucket,
                "tags": [bucket],
            }
            out_md_path = path_from_url(proc_dir, url, replace_ext=".md")
            with open(out_md_path, "w", encoding="utf-8") as f:
                f.write(write_front_matter_markdown(md_body, meta))

            # 6) Update state so the next run can skip if unchanged.
            state[url] = {
                "etag": hdrs.get("ETag"),
                "last_modified": hdrs.get("Last-Modified"),
                "content_sha1": content_hash,
                "last_fetch": datetime.utcnow().isoformat(timespec="seconds") + "Z"
            }

            logging.info(f"[{i}/{len(merged)}] ✅ UPDATED [{bucket}|{source}] {url}")
            success += 1
            time.sleep(0.3)

        except Exception as e:
            logging.warning(f"[{i}/{len(merged)}] ❌ {url} — {e}")
            fail += 1

    # Persist the updated state file at the end of the run
    save_state(state)

    logging.info(f"Done. Success: {success}, Failed: {fail}")
    logging.info(f"Raw → {raw_dir} | Markdown → {proc_dir}")
    logging.info(f"TOTAL URLs fetched and converted: {success}")

if __name__ == "__main__":
    main()