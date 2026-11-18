"""Run aSRR through the manuscript's existing numerical fitting workers."""

import argparse
import json
import shutil

import numpy as np
import torch

from experiments.paper_recovery_extension_v1 import workers

from .prepare import OUTPUT, ROOT


def run(dataset, phase, codes, device):
    torch.set_num_threads(2)
    root, output = (ROOT / dataset).resolve(), (OUTPUT / dataset).resolve()
    if phase == "refit":
        folder = root / "source/asrr"
        metadata = json.loads((folder / "candidates.json").read_text())[
            "candidates"
        ]
        with np.load(folder / "candidates.npz") as arrays:
            proposals = [
                (r["name"], arrays[r["name"]], r["anchor"])
                for r in metadata
                if r["codes"] == codes
            ]
        workers.refit(root, output, "asrr", codes, device, proposals=proposals)
        shutil.copytree(
            root / f"fits/asrr_k{codes}", output / f"source_fit/k{codes}"
        )
    elif phase == "tune":
        workers.tune(root, output, "asrr", device)
        shutil.copytree(root / "validation_fit/asrr", output / "development")
    else:
        workers.target(root, output, f"asrr_k{codes}", device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--phase", choices=("refit", "tune", "target"), required=True
    )
    parser.add_argument("--codes", type=int)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run(args.dataset, args.phase, args.codes, torch.device(args.device))
