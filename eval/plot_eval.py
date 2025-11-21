# eval/plot_eval.py
#
# Simple plotting script for gold_eval results.
# Reads eval/eval_results.csv and produces a bar chart of
# substring accuracy per category.

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_results():
    # Resolve CSV path relative to this file (eval/ directory)
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "eval_results.csv")

    if not os.path.exists(csv_path):
        print(f"[error] CSV not found at: {csv_path}")
        print("Run:  python -m eval.gold_eval kb/gold_qa.csv  first.")
        return

    print(f"Loading eval results from: {csv_path}")
    df = pd.read_csv(csv_path)

    if "category" not in df.columns or "substring_match" not in df.columns:
        print("[error] eval_results.csv is missing 'category' or 'substring_match' columns.")
        return

    # Convert substring_match to numeric (True/False -> 1/0 if needed)
    df["substring_match"] = df["substring_match"].astype(int)

    # Compute accuracy per category
    acc_by_cat = (
        df.groupby("category")["substring_match"]
        .mean()
        .reset_index()
        .sort_values("substring_match", ascending=False)
    )

    print("\nPer-category substring accuracy:")
    for _, row in acc_by_cat.iterrows():
        print(f"  {row['category']}: {row['substring_match']:.3f}")

    # Plot
    plt.figure(figsize=(7, 4))
    plt.bar(acc_by_cat["category"], acc_by_cat["substring_match"])
    plt.ylim(0, 1.0)
    plt.xlabel("Category")
    plt.ylabel("Substring Accuracy")
    plt.title("MSU FAQ Chatbot – Accuracy by Category")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    # Save figure next to the CSV (eval/accuracy_by_category.png)
    out_path = os.path.join(base_dir, "accuracy_by_category.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved plot to: {out_path}")

    # Optional: show window if you run locally
    plt.show()


if __name__ == "__main__":
    plot_results()
