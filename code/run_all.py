from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--qual_source", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    py = sys.executable

    run([py, "code/train.py", "--config", args.config])
    run([py, "code/benchmark.py", "--config", args.config])
    run([py, "code/carbon_estimate.py", "--config", args.config])

    if args.qual_source:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        weights = f"{cfg['project_dir']}/{cfg['experiment_name']}/weights/best.pt"
        outdir = f"outputs/qualitative/{cfg['experiment_name']}"
        run([py, "code/qualitative_results.py", "--weights", weights, "--source", args.qual_source, "--outdir", outdir])

    print("Pipeline complete")


if __name__ == "__main__":
    main()
