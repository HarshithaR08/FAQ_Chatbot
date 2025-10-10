# Creates a safety net: keeps policy files only if their text matches lexicon terms. 
# Off-topic files are moved to _out_of_scope/.

import re, logging, yaml, frontmatter
from pathlib import Path

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def load_cfg(path="sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def score(text: str, terms):
    return sum(1 for t in terms if re.search(rf"\b{re.escape(t)}\b", text, re.I))

def matches_topic(text: str, topic_cfg: dict) -> bool:
    any_terms = topic_cfg.get("any", [])
    must_one = topic_cfg.get("must_one_of", []) or []
    # keep if 2+ hits from "any" OR at least 1 from "must_one"
    return score(text, any_terms) >= 2 or (must_one and score(text, must_one) >= 1)

def main():
    setup_logger()
    cfg = load_cfg()
    lex = cfg.get("policy_topic_lexicon", {})

    processed = Path("data/processed")
    moved = kept = 0

    # 🔴 Expand scope: include policy PDFs as well as plain policy pages
    BUCKETS_TO_CHECK = {"policies", "policies-pdf", "uploads-pdf", "university-policies-pdf"}

    for md in processed.rglob("*.md"):
        rel = md.relative_to(processed)
        if rel.parts and rel.parts[0] == "_out_of_scope":
            # skip anything already in quarantine
            continue

        post = frontmatter.load(md)
        bucket = post.metadata.get("bucket")
        if bucket not in BUCKETS_TO_CHECK:
            continue

        text = (post.content or "").strip()

        # ✅ compute which topics matched
        hit_topics = [name for name, topic_cfg in lex.items() if matches_topic(text, topic_cfg)]
        ok = bool(hit_topics)

        if ok:
            kept += 1
            logging.info(f"KEEP {md} topics={hit_topics}")
        else:
            outdir = processed / "_out_of_scope" / md.parent.relative_to(processed)
            outdir.mkdir(parents=True, exist_ok=True)
            md.rename(outdir / md.name)
            moved += 1
            logging.info(f"MOVE {md} topics=[]")

    logging.info("Policy post-filter: kept=%d, moved_out=%d", kept, moved)

if __name__ == "__main__":
    main()
