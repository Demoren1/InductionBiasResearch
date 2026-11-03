"""Collect one source family for the structure-only generative pilot."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from experiments.adaptive_tanh_v1.source import fit_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = json.loads((root / "protocol.json").read_text())
    if not 0 <= args.family < plan["families"]:
        parser.error("family is outside the declared source library")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    torch.set_num_threads(4)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    family_output = output / f"source_{args.family}"
    fit_source(
        plan,
        plan["seed"] + 1000 * args.family,
        family_output,
        torch.device("cuda:0"),
    )
    shutil.copy2(root / "protocol.json", family_output / "protocol.json")


if __name__ == "__main__":
    main()
