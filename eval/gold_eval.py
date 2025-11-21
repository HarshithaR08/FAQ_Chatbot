"""
eval/gold_eval.py

Lightweight evaluation harness for the MSU FAQ Chatbot.

What it does:
- Reads a CSV of gold questions (kb/gold_qa.csv).
- For each question:
    * Calls ui.rag_cli.generate_answer()  --> uses REAL pipeline
    * Checks if the model answer contains an expected substring (after normalization).
    * Optionally:
        - Exact Match (EM) if 'expected_answer' is provided.
        - Token-level F1 if 'expected_answer' or 'expected_substring' is provided.
        - Source URL accuracy if 'expected_source_url' is provided.
- Prints overall accuracy + per-category stats + some failure examples.
- Saves per-question metrics to eval/eval_results.csv

CSV format (minimum):
    id,question,expected_substring,category

Optional extra columns:
    expected_answer,expected_source_url
"""

import csv
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from ui.rag_cli import generate_answer  # uses your real pipeline

# ---------- Normalization helpers ----------

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = "".join(" " if c.isspace() else c for c in s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    from collections import Counter
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    overlap = sum(min(pred_counts[t], gold_counts[t]) for t in gold_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


# ---------- CSV loader ----------

def load_gold(filepath: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------- Evaluate single question ----------

def eval_one_item(row: Dict[str, str]) -> Dict[str, Any]:
    qid = row.get("id") or row.get("qid") or ""
    question = row.get("question", "").strip()
    expected_sub = row.get("expected_substring", "").strip()
    expected_answer = row.get("expected_answer", "").strip()
    expected_source_url = row.get("expected_source_url", "").strip()
    category = row.get("category", "").strip() or "uncategorized"

    print(f"\n[eval] {qid or '(no-id)'}: {question}")

    answer, primary_source, intent = generate_answer(question, top_k=6)

    norm_answer = normalize_text(answer)
    norm_expected_sub = normalize_text(expected_sub) if expected_sub else ""
    norm_expected_answer = normalize_text(expected_answer) if expected_answer else ""

    substring_match = norm_expected_sub in norm_answer if norm_expected_sub else False
    em = (norm_answer == norm_expected_answer) if expected_answer else False

    if expected_answer:
        f1 = f1_score(answer, expected_answer)
    elif expected_sub:
        f1 = f1_score(answer, expected_sub)
    else:
        f1 = 0.0

    if expected_source_url:
        if primary_source and primary_source.get("url"):
            src_correct = expected_source_url in primary_source["url"]
        else:
            src_correct = False
    else:
        src_correct = None

    print(f"  -> substring_match={substring_match}, EM={em}, F1={f1:.3f}, intent={intent}")
    if expected_source_url:
        print(f"  -> source_correct={src_correct} (expected contains: {expected_source_url})")

    return {
        "id": qid,
        "category": category,
        "question": question,
        "expected_substring": expected_sub,
        "expected_answer": expected_answer,
        "expected_source_url": expected_source_url,
        "intent": intent,
        "answer": answer,
        "substring_match": substring_match,
        "em": em,
        "f1": f1,
        "source_correct": src_correct,
        "primary_source": primary_source,
    }


# ---------- SAVE CSV of all results ----------

def save_eval_csv(results: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "category", "substring_match", "EM",
                    "F1", "intent", "source_correct", "primary_url"])
        for r in results:
            primary_url = r["primary_source"]["url"] if r["primary_source"] else ""
            w.writerow([
                r["id"],
                r["question"],
                r["category"],
                int(r["substring_match"]),
                int(r["em"]),
                f"{r['f1']:.3f}",
                r["intent"],
                ("" if r["source_correct"] is None else int(r["source_correct"])),
                primary_url,
            ])
    print(f"\nSaved eval results to {out_path}\n")


# ---------- Main ----------

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m eval.gold_eval kb/gold_qa.csv")
        sys.exit(1)

    gold_path = sys.argv[1]
    print(f"Loading gold questions from: {gold_path}")
    gold_rows = load_gold(gold_path)
    print(f"Loaded {len(gold_rows)} gold items.")

    results: List[Dict[str, Any]] = []
    for row in gold_rows:
        try:
            res = eval_one_item(row)
        except Exception as e:
            print(f"[error] failed on question {row.get('id')}: {e}")
            continue
        results.append(res)

    # Aggregate metrics
    total = len(results)
    correct_sub = sum(1 for r in results if r["substring_match"])
    avg_f1 = sum(r["f1"] for r in results) / total

    # EM
    em_questions = [r for r in results if r["expected_answer"]]
    em_acc = (
        sum(1 for r in em_questions if r["em"]) / len(em_questions)
        if em_questions else 0.0
    )

    # Source accuracy
    src_rows = [r for r in results if r["source_correct"] is not None]
    src_acc = (
        sum(1 for r in src_rows if r["source_correct"]) / len(src_rows)
        if src_rows else 0.0
    )

    print("\n========== OVERALL ==========")
    print(f"Total questions: {total}")
    print(f"Substring Accuracy: {correct_sub / total:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    if em_questions:
        print(f"EM Accuracy: {em_acc:.3f}")
    if src_rows:
        print(f"Source URL Accuracy: {src_acc:.3f}")

    # Per-category
    cat_stats = defaultdict(lambda: {"n": 0, "correct": 0, "f1": 0.0})
    for r in results:
        c = cat_stats[r["category"]]
        c["n"] += 1
        if r["substring_match"]:
            c["correct"] += 1
        c["f1"] += r["f1"]

    print("\n====== BY CATEGORY (substring) ======")
    for cat, s in sorted(cat_stats.items()):
        acc = s["correct"] / s["n"]
        print(f"{cat:20s} {s['correct']:2d}/{s['n']:3d} ({acc:.3f}) | avg F1={s['f1']/s['n']:.3f}")

    # Sample failures
    fails = [r for r in results if not r["substring_match"]]
    if fails:
        print("\n====== SAMPLE FAILURES ======")
        for r in fails[:5]:
            print(f"\n--- {r['id']} | category={r['category']} ---")
            print(f"Q: {r['question']}")
            print(f"Expected: {r['expected_substring']}")
            print(f"Answer: {r['answer']}")
            if r["primary_source"]:
                print("Primary Source:", r["primary_source"].get("url"))
            print("----------------------------------------")

    # Save CSV
    save_eval_csv(results, Path("eval/eval_results.csv"))


if __name__ == "__main__":
    main()
