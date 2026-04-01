from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained model for deployment")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript"])
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    exported = model.export(format=args.format, imgsz=args.imgsz)
    print("Export complete")
    print(exported)


if __name__ == "__main__":
    main()
