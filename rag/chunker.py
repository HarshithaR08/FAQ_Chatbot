# Developer Name: Harshitha
# Read Markdown (with front-matter) from data/processed/, split into small overlapping chunks,
# and write rag/chunks.jsonl for indexing.

import os, re, json, logging, hashlib, frontmatter
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

def list_markdown_files(processed_dir: str) -> List[Path]:
    p = Path(processed_dir)
    # Skip stub files that still end with -pdf.md (if any remain)
    return [f for f in p.rglob("*.md") if not str(f.name).endswith("-pdf.md")]

def split_by_headings(md: str) -> List[Dict]:
    # Split markdown into sections using ATX headings (#, ##, ###...).
    # Returns a list of {heading_path: [..], text: "..."} where heading_path is a list of strings.
    lines = md.splitlines()
    sections = []
    current_heading_path = []
    current_text_lines = []

    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")

    def flush():
        if current_text_lines:
            sections.append({
                "heading_path": current_heading_path.copy(),
                "text": "\n".join(current_text_lines).strip()
            })

    for line in lines:
        m = heading_re.match(line)
        if m:
            # new heading -> flush previous text
            flush()
            hashes, title = m.group(1), m.group(2).strip()
            level = len(hashes)
            # adjust heading path depth
            current_heading_path = current_heading_path[:level-1] + [f"H{level}: {title}"]
            current_text_lines = []
        else:
            current_text_lines.append(line)

    flush()
    # Drop empty sections
    sections = [s for s in sections if s["text"]]
    return sections

def sliding_chunks(text: str, target: int, overlap: int) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= target:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + target
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

def chunk_doc(md_text: str, meta: dict) -> List[dict]:
    title = meta.get("source_title", "") or ""
    url = meta.get("source_url", "")
    bucket = meta.get("bucket", "") or ""
    tags = meta.get("tags", [])

    chunks = []
    sections = split_by_headings(md_text)

    if not sections:
        # No headings—chunk whole doc
        for i, piece in enumerate(sliding_chunks(md_text, TARGET_CHARS, OVERLAP_CHARS), 1):
            cid = hashlib.sha1(f"{url}::nohead::{i}".encode("utf-8")).hexdigest()
            chunks.append({
                "id": f"sha1:{cid}",
                "source_url": url,
                "title": title,
                "bucket": bucket,
                "tags": tags,
                "headings": [],
                "text": piece
            })
        return chunks

    # Chunk by sections
    for s_idx, sec in enumerate(sections, 1):
        sec_text = sec["text"]
        heading_path = sec["heading_path"]
        pieces = sliding_chunks(sec_text, TARGET_CHARS, OVERLAP_CHARS)
        for i, piece in enumerate(pieces, 1):
            cid = hashlib.sha1(f"{url}::{';'.join(heading_path)}::{i}".encode("utf-8")).hexdigest()
            chunks.append({
                "id": f"sha1:{cid}",
                "source_url": url,
                "title": title,
                "bucket": bucket,
                "tags": tags,
                "headings": heading_path,
                "text": piece
            })
    return chunks

def main():
    setup_logger()
    processed_dir = "data/processed"
    out_path = Path("rag/chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list_markdown_files(processed_dir)
    logging.info(f"Found {len(files)} Markdown files for chunking")

    total_chunks = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for idx, fpath in enumerate(files, 1):
            try:
                post = frontmatter.load(fpath)
                md_text = post.content or ""
                meta = {**post.metadata}
                doc_chunks = chunk_doc(md_text, meta)
                total_chunks += len(doc_chunks)
                for ch in doc_chunks:
                    out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                if idx % 25 == 0:
                    logging.debug(f"Chunked {idx}/{len(files)} files; total_chunks={total_chunks}")
            except Exception as e:
                logging.warning(f"Skip {fpath}: {e}")

    logging.info(f"✅ Wrote {total_chunks} chunks → {out_path}")

if __name__ == "__main__":
    main()
