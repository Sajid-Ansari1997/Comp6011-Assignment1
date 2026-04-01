from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from utils import ensure_dir, load_yaml, save_csv, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a trained model")
    parser.add_argument("--config", type=str, required=True, help="Experiment config YAML")
    return parser.parse_args()


def resolve_weights_path(cfg: dict) -> Path:
    # Option 1:
    # If YAML gives a direct weights path, use it
    if "weights_path" in cfg and cfg["weights_path"]:
        return Path(cfg["weights_path"])

    # Option 2:
    # Build path from project_dir + experiment_name
    if "project_dir" in cfg and "experiment_name" in cfg:
        return Path(cfg["project_dir"]) / cfg["experiment_name"] / "weights" / "best.pt"

    raise ValueError(
        "Config must contain either 'weights_path' or both 'project_dir' and 'experiment_name'."
    )


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    weights = resolve_weights_path(cfg)
    metrics_dir = ensure_dir("outputs/metrics")

    print(f"Looking for weights at: {weights.resolve()}")

    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights.resolve()}")

    model = YOLO(str(weights))

    start = time.time()
    metrics = model.val(
        data=cfg["dataset_yaml"],
        imgsz=int(cfg["imgsz"]),
        device=str(cfg["device"]),
        split=cfg.get("split", "val"),
        plots=True,
    )
    elapsed = time.time() - start

    result = {
        "experiment_name": cfg.get("experiment_name", weights.stem),
        "candidate_label": cfg.get("candidate_label", "unknown"),
        "dataset_name": cfg.get("dataset_name", "unknown"),
        "base_model": cfg.get("base_model", "unknown"),
        "weights_path": str(weights),
        "dataset_yaml": cfg["dataset_yaml"],
        "split": cfg.get("split", "val"),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "eval_time_seconds": round(elapsed, 2),
    }

    result["fps_estimate"] = round(
        1.0 / max(result["eval_time_seconds"] / 100.0, 1e-6), 2
    )

    save_json(result, metrics_dir / f"{result['experiment_name']}_metrics.json")
    save_csv(pd.DataFrame([result]), metrics_dir / f"{result['experiment_name']}_metrics.csv")

    print("Benchmark complete")
    print(result)


if __name__ == "__main__":
    main()