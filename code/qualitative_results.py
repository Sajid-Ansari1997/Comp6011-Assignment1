from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Save qualitative prediction images")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--source", type=str, required=True, help="Image folder or video")
    parser.add_argument("--outdir", type=str, required=True, help="Output folder")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        project=args.outdir,
        name="predictions",
        exist_ok=True,
    )
    print(f"Qualitative outputs saved in {Path(args.outdir) / 'predictions'}")


if __name__ == "__main__":
    main()
