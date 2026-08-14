"""Merge partial eval results from parallel GPU runs and plot the final figure."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from evaluation.eval_generated_masks import plot_results  # noqa: E402


def main():
    out_dir = config.EVAL_DIR
    merged = {}
    for f in sorted(out_dir.glob("eval_results_gpu*.pt")):
        d = torch.load(f, weights_only=True)
        merged.update(d)
    for f in sorted(out_dir.glob("eval_results_gpu*.json")):
        with open(f) as fh:
            d = json.load(fh)
        merged.update({k: v for k, v in d.items() if k not in merged})

    torch.save(merged, out_dir / "eval_results.pt")
    with open(out_dir / "eval_results.json", "w") as fh:
        json.dump(merged, fh, indent=2)
    print(f"[merge] {len(merged)} entries -> {out_dir / 'eval_results.json'}")

    config.ensure_plot_dirs()
    plot_results(merged)


if __name__ == "__main__":
    main()
