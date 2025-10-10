# Developer Name: Harshitha
# Turn PDF stubs (-pdf.md) into real Markdown by extracting text from data/raw/*.pdf

import os, re, logging, yaml, frontmatter
from pathlib import Path
from pdfminer.high_level import extract_text

def setup_logger():
    # our logs
    logging.basicConfig(
        level=logging.INFO,  # was DEBUG; INFO is enough now
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        force=True
    )
    # hush pdfminer internals
    for name in (
        "pdfminer", "pdfminer.high_level", "pdfminer.layout",
        "pdfminer.pdfinterp", "pdfminer.pdfpage", "pdfminer.psparser",
        "pdfminer.cmapdb"
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

# open sources.yaml file and parses yaml into a python dic using safe_load
def load_cfg(path="sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# finding pdf stubs
def find_pdf_stubs(processed_dir: str):
    p = Path(processed_dir)
    # recursively find Markdown files whose filename ends with -pdf.md
    stubs = list(p.rglob("*-pdf.md"))
    logging.info("Found %d PDF stubs", len(stubs))
    if stubs[:5]:
        logging.debug("Stub samples:\n  - " + "\n  - ".join(str(s) for s in stubs[:5]))
    return stubs

def stub_to_pdf_path(stub_md_path: Path, processed_dir: str, raw_dir: str) -> Path:
    # processed/foo/bar/file-pdf.md  -> raw/foo/bar/file.pdf
    rel = Path(os.path.relpath(stub_md_path, processed_dir))
    pdf_rel = Path(str(rel).replace("-pdf.md", ".pdf"))
    return Path(raw_dir) / pdf_rel

def clean_pdf_text(txt: str) -> str:
    txt = txt.replace("\r", "")
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)   # fix hyphen-linebreaks
    txt = re.sub(r"\n{3,}", "\n\n", txt)         # collapse blank lines
    txt = "\n".join(line.strip() for line in txt.splitlines())
    return txt.strip()

def to_markdown(txt: str) -> str:
    return txt  # keep paragraphs as-is for now

def process_stub(stub_path: Path, processed_dir: str, raw_dir: str) -> bool:
    post = frontmatter.load(stub_path)
    pdf_path = stub_to_pdf_path(stub_path, processed_dir, raw_dir)

    if not pdf_path.exists():
        logging.warning("PDF not found for stub: %s -> %s", stub_path, pdf_path)
        return False

    logging.info("Extracting: %s", pdf_path)
    try:
        text = extract_text(str(pdf_path)) or ""
    except Exception as e:
        logging.warning("Extraction failed: %s — %s", pdf_path, e)
        return False

    cleaned = clean_pdf_text(text)
    if not cleaned:
        logging.warning("Empty/near-empty text after extraction: %s", pdf_path)

    post.content = to_markdown(cleaned)

    out_md = Path(str(stub_path).replace("-pdf.md", ".md"))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(frontmatter.dumps(post))

    logging.info("✅ Wrote: %s", out_md)
    return True

def main():
    setup_logger()
    cfg = load_cfg()
    processed_dir = cfg["paths"]["processed_dir"]
    raw_dir = cfg["paths"]["raw_dir"]

    stubs = find_pdf_stubs(processed_dir)
    if not stubs:
        logging.info("No stubs to convert. If you expected some, re-run ingest/run_all.py and confirm PDFs were detected.")
        return

    ok = fail = 0
    for i, stub in enumerate(stubs, 1):
        if process_stub(stub, processed_dir, raw_dir):
            ok += 1
        else:
            fail += 1
        if i % 10 == 0:
            logging.debug("Progress: %d/%d (ok=%d, fail=%d)", i, len(stubs), ok, fail)

    logging.info("Done. Converted: %d, Failed: %d", ok, fail)

if __name__ == "__main__":
    main()
