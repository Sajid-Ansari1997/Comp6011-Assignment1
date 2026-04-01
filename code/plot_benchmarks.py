from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Plot benchmark comparison")
    parser.add_argument("--metrics_dir", type=str, required=True, help="Folder with metric JSON files")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    outdir = ensure_dir("outputs/figures")

    rows = []
    for path in metrics_dir.glob("*_metrics.json"):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    if not rows:
        raise FileNotFoundError("No metric JSON files found. Run benchmark.py first.")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "combined_metrics.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(df["fps_estimate"], df["mAP50"], s=90)
    for _, row in df.iterrows():
        plt.annotate(row["candidate_label"], (row["fps_estimate"], row["mAP50"]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Estimated FPS")
    plt.ylabel("mAP@0.5")
    plt.title("Accuracy vs Speed Across Benchmark Candidates")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "accuracy_speed_tradeoff.png", dpi=300)
    plt.close()

    summary = df[["candidate_label", "dataset_name", "mAP50", "mAP50_95", "precision", "recall", "fps_estimate"]]
    summary.to_csv(outdir / "benchmark_summary_table.csv", index=False)

    print("Saved figure and summary table")
    print(summary)


if __name__ == "__main__":
    main()
