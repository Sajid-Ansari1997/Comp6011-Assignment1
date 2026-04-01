from __future__ import annotations

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

from utils import device_name, ensure_dir, load_yaml, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a perception candidate model")
    parser.add_argument("--config", type=str, required=True, help="Experiment config YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(42)

    project_dir = cfg["project_dir"]
    exp_name = cfg["experiment_name"]
    ensure_dir(project_dir)

    model = YOLO(cfg["base_model"])
    start = time.time()

    model.train(
        data=cfg["dataset_yaml"],
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        device=str(cfg["device"]),
        project=project_dir,
        name=exp_name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        verbose=True,
    )

    elapsed = time.time() - start
    run_dir = Path(project_dir) / exp_name

    training_meta = {
        "experiment_name": exp_name,
        "dataset_name": cfg["dataset_name"],
        "candidate_label": cfg["candidate_label"],
        "base_model": cfg["base_model"],
        "epochs": int(cfg["epochs"]),
        "imgsz": int(cfg["imgsz"]),
        "batch": int(cfg["batch"]),
        "device": str(cfg["device"]),
        "device_name": device_name(),
        "training_time_seconds": round(elapsed, 2),
        "best_weights": str((run_dir / "weights" / "best.pt").resolve()),
    }
    save_json(training_meta, run_dir / "training_metadata.json")
    print("Training complete")
    print(training_meta)


if __name__ == "__main__":
    main()
