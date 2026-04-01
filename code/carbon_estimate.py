from __future__ import annotations

import argparse
from pathlib import Path

from codecarbon import EmissionsTracker
from ultralytics import YOLO

from utils import ensure_dir, load_yaml, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate carbon footprint for one validation run")
    parser.add_argument("--config", type=str, required=True, help="Experiment config YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    run_dir = Path(cfg["project_dir"]) / cfg["experiment_name"]
    weights = run_dir / "weights" / "best.pt"
    outdir = ensure_dir("outputs/metrics")

    tracker = EmissionsTracker(project_name=cfg["experiment_name"])
    tracker.start()

    model = YOLO(str(weights))
    model.val(
        data=cfg["dataset_yaml"],
        imgsz=int(cfg["imgsz"]),
        device=str(cfg["device"]),
        split="val",
        plots=False,
    )

    emissions = tracker.stop()
    result = {
        "experiment_name": cfg["experiment_name"],
        "estimated_emissions_kg_co2": emissions,
        "method": "CodeCarbon validation-run estimate"
    }
    save_json(result, outdir / f"{cfg['experiment_name']}_carbon.json")
    print("Carbon estimate saved")
    print(result)


if __name__ == "__main__":
    main()
