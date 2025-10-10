# Developer Name: Harshitha
# Desccription: Loads config sources.yaml,Build URL list from sitemaps (including any manual URL's)
# For each URL:
                # if its a PDF - download file + write a stub Markdown with metadata
                # if its HTML - fetch page, save raw, convert to Markdown, add metadata (front matter)


import os, re, logging, time, urllib.parse, yaml, requests
from datetime import datetime
from fetch_from_sitemap import run_from_sources
from html_to_markdown import clean_html_to_markdown

USER_AGENT = "MSU-FAQ-Bot/0.1 (contact: ramakrishnah1@montclair.edu)"


# Build a file path under base_dir that mirrors the URL path.
# If replace_ext is given (e.g., '.pdf' or '-pdf.md'), it replaces the file's extension.
# Ensures parent directory exists.
def path_from_url(base_dir: str, url: str, replace_ext: str | None = None) -> str:
    rel = urllib.parse.urlparse(url).path.lstrip("/")  # e.g., policies/wp-content/.../file.pdf
    head, tail = os.path.split(rel)
    # If the URL ends with '/', os.path.split gives tail == ''
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

# still used by some scripts; safe to keep
def safe_filename(url: str) -> str:
    path = urllib.parse.urlparse(url).path.strip("/")
    path = re.sub(r"[^a-zA-Z0-9/_-]", "-", path)
    if not path:
        path = "index"
    if path.endswith("/"):
        path = path[:-1]
    return path or "index"

def fetch_text(url: str, timeout=60) -> str:
    logging.debug(f"GET {url}")
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.text

def write_front_matter_markdown(md_text: str, meta: dict) -> str:
    fm_lines = ["---"]
    for k, v in meta.items():
        fm_lines.append(f"{k}: {v!r}")
    fm_lines.append("---")
    return "\n".join(fm_lines) + "\n\n" + md_text

def main():
    setup_logger()
    cfg = load_cfg()
    raw_dir = cfg["paths"]["raw_dir"]
    proc_dir = cfg["paths"]["processed_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    # Collect URLs with bucket rules (returns list of (url, lastmod, bucket))
    urls = run_from_sources(
        sitemaps=cfg["sitemaps"],
        include_prefixes=cfg["include_prefixes"],   # kept for signature compatibility
        csv_out=cfg["paths"]["url_list_csv"],
        cfg=cfg,                                    # IMPORTANT: enables bucket/exclude regex
    )

    # Add manual URLs (label as 'manual')
    manual = cfg.get("manual", [])
    if manual:
        logging.info(f"Adding {len(manual)} manual URLs")
        urls.extend((u, "", "manual") for u in manual)

    # Print first 15 URLs for sanity check
    logging.info("First 15 URLs to be fetched:")
    for u, _, b in urls[:15]:
        logging.info(f" - [{b}] {u}")

    # Fetch each page and convert to Markdown (PDFs get stub markdown)
    success, fail = 0, 0
    for i, (url, lastmod, bucket) in enumerate(urls, start=1):
        try:
            # PDF branch
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

                logging.info(f"[{i}/{len(urls)}] ✅ [pdf] {url}")
                success += 1
                time.sleep(0.2)
                continue  # skip HTML branch for PDFs

            # HTML branch 
            html = fetch_text(url)

            # Save raw HTML (use mirrored URL path)
            raw_path = path_from_url(raw_dir, url, replace_ext=".html")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(html)

            md_body = clean_html_to_markdown(html)
            meta = {
                "source_url": url,
                "source_title": "",  # you can parse <title> if you want
                "last_fetched": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "lastmod": lastmod or "",
                "bucket": bucket,
                "tags": [bucket],
            }
            out_md_path = path_from_url(proc_dir, url, replace_ext=".md")
            with open(out_md_path, "w", encoding="utf-8") as f:
                f.write(write_front_matter_markdown(md_body, meta))

            logging.info(f"[{i}/{len(urls)}] ✅ [{bucket}] {url}")
            success += 1
            time.sleep(0.3)  # be polite

        except Exception as e:
            logging.warning(f"[{i}/{len(urls)}] ❌ {url} — {e}")
            fail += 1

    logging.info(f"Done. Success: {success}, Failed: {fail}")
    logging.info(f"Raw → {raw_dir} | Markdown → {proc_dir}")
    logging.info(f"TOTAL URLs fetched and converted: {success}")

if __name__ == "__main__":
    main()
