"""Merge partial eval results from parallel GPU runs and plot the final figure."""

import json
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from evaluation.eval_generated_masks import plot_results  # noqa: E402


def load_partial_results(path_pt: Path, path_json: Path) -> dict:
    """Load one worker result, preferring the tensor artifact."""
    if path_pt.exists():
        return torch.load(path_pt, weights_only=True)
    if path_json.exists():
        return json.loads(path_json.read_text())
    raise FileNotFoundError(f"Missing evaluation result: {path_pt} or {path_json}")


def main():
    parser = argparse.ArgumentParser(description="Merge one evaluation run.")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--out_suffix", type=str, default="")
    args = parser.parse_args()
    out_dir = config.EVAL_DIR
    merged = {}
    for gpu_id in range(args.num_gpus):
        stem = out_dir / f"eval_results_gpu{gpu_id}{args.out_suffix}"
        merged.update(load_partial_results(stem.with_suffix(".pt"),
                                           stem.with_suffix(".json")))

    out_pt = out_dir / f"eval_results{args.out_suffix}.pt"
    out_json = out_dir / f"eval_results{args.out_suffix}.json"
    torch.save(merged, out_pt)
    with open(out_json, "w") as fh:
        json.dump(merged, fh, indent=2)
    print(f"[merge] {len(merged)} entries -> {out_json}")

    config.ensure_plot_dirs()
    plot_results(merged, args.out_suffix)


if __name__ == "__main__":
    main()
