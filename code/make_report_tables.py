from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report-ready tables")
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    outdir = ensure_dir("outputs/tables")

    rows = []
    for path in metrics_dir.glob("*_metrics.json"):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    if not rows:
        raise FileNotFoundError("No metric JSON files found.")

    df = pd.DataFrame(rows)
    benchmark_table = df[[
        "candidate_label",
        "dataset_name",
        "mAP50",
        "mAP50_95",
        "precision",
        "recall",
        "fps_estimate",
    ]]
    benchmark_table.columns = [
        "Model Candidate",
        "Dataset",
        "mAP@0.5",
        "mAP@0.5:0.95",
        "Precision",
        "Recall",
        "Estimated FPS",
    ]
    benchmark_table.to_csv(outdir / "benchmark_table.csv", index=False)

    interpretation = pd.DataFrame([
        ["Higher mAP@0.5", "Better coarse detection quality"],
        ["Higher mAP@0.5:0.95", "Better localization across IoU thresholds"],
        ["Higher Precision", "Fewer false positives"],
        ["Higher Recall", "Fewer missed detections"],
        ["Higher FPS", "Better real-time suitability"],
    ], columns=["Metric", "Interpretation"])
    interpretation.to_csv(outdir / "metric_interpretation_table.csv", index=False)

    print("Report tables saved")
    print(benchmark_table)


if __name__ == "__main__":
    main()
