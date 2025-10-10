# Build FAQ KB from seeds + rag/chunks.jsonl by simple lexical scoring.

import json, yaml, re, csv, logging, hashlib
from pathlib import Path
from typing import List, Dict

CHUNKS_PATH = Path("rag/chunks.jsonl")
OUT_JSONL = Path("kb/faq.jsonl")
OUT_CSV   = Path("kb/faq.csv")

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

def load_seeds(path="kb/seed_questions.yaml") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("seeds", [])

def load_chunks(path=CHUNKS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def score_chunk(text: str, must_any: List[str], prefer_any: List[str]) -> int:
    t = text.lower()
    # hard filter: must contain at least one must_any
    if must_any and not any(w.lower() in t for w in must_any):
        return -1
    score = 0
    # prefer signals
    for w in prefer_any:
        if w.lower() in t:
            score += 2
    # length prior (shorter chunks are often more specific)
    score += max(0, 3 - len(text) // 800)  # +3..+1
    return score

def best_chunk_for_seed(seed, chunks_iter):
    best = None
    best_score = -1
    must_any = seed.get("must_any", [])
    prefer_any = seed.get("prefer_any", [])
    for ch in chunks_iter:
        s = score_chunk(ch["text"], must_any, prefer_any)
        if s > best_score:
            best_score = s
            best = ch
    return best, best_score

def trim_answer(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text.strip()
    # cut at paragraph boundary if possible
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = ""
    for p in paras:
        if len(out) + len(p) + 2 > limit:
            break
        out += (("\n\n" if out else "") + p)
    return (out or text[:limit]).strip()

def main():
    setup_logger()
    seeds = load_seeds()
    if not CHUNKS_PATH.exists():
        logging.error("Missing chunks at %s. Run rag/chunker.py first.", CHUNKS_PATH)
        return

    # Load chunks into memory once (ok for thousands)
    chunks = list(load_chunks())
    logging.info("Loaded %d chunks", len(chunks))

    results = []
    for seed in seeds:
        # Fresh iterator each time
        best, sc = best_chunk_for_seed(seed, iter(chunks))
        if not best or sc < 0:
            logging.warning("No match for seed: %s", seed["question"])
            continue

        limit = int(seed.get("max_answer_chars", 900))
        ans = trim_answer(best["text"], limit)

        item = {
            "id": f"kb:{hashlib.sha1(seed['question'].encode()).hexdigest()[:12]}",
            "topic": seed["topic"],
            "question": seed["question"],
            "answer": ans,
            "source_url": best.get("source_url", ""),
            "bucket": best.get("bucket", ""),
            "headings": best.get("headings", []),
            "chunk_id": best.get("id", ""),
        }
        results.append(item)
        logging.info("Seed → %s | score=%d | src=%s", seed["question"], sc, item["source_url"])

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","topic","question","answer","source_url","bucket","headings","chunk_id"])
        for r in results:
            w.writerow([r["id"], r["topic"], r["question"], r["answer"], r["source_url"], r["bucket"], "; ".join(r["headings"]), r["chunk_id"]])

    logging.info("Wrote %d FAQs → %s and %s", len(results), OUT_JSONL, OUT_CSV)

if __name__ == "__main__":
    main()
