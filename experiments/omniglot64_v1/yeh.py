"""Fit the fixed Yeh soft configuration after the initial paper build."""

import argparse
import time
from pathlib import Path

import torch

from experiments.paper_structure_benchmark_v1.methods.real_competitors import run
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .source import BANKS, CODES, OUTPUT, ROOT, read_json


def fit(device):
    paper = read_json(OUTPUT / "paper_before_yeh.json")
    if (
        digest(Path("paper/main.pdf")) != paper["paper_sha256"]
        or digest(OUTPUT / "results_initial.json") != paper["results_sha256"]
    ):
        raise ValueError("The initial paper build and result snapshot must match")
    settings = read_json(OUTPUT / "registration.json")
    basis = BANKS / "yeh.npz"
    if basis.exists():
        raise FileExistsError(basis)
    started = time.perf_counter()
    run(
        ROOT,
        "yeh",
        CODES,
        device,
        settings["yeh_outer_steps"],
        settings=[settings["yeh_settings"]],
        basis_path=basis,
        profile=False,
    )
    elapsed = time.perf_counter() - started
    metrics = read_json(ROOT / "source" / f"yeh_k{CODES}/metrics.json")
    save_json(
        OUTPUT / "yeh_source.json",
        {
            "candidates": [{"name": "yeh", "model_sha256": digest(basis)}],
            "seconds": elapsed,
            "native_fit_and_export_seconds": metrics["wall_seconds"],
            "cpu_peak_rss_bytes": metrics["cpu_peak_rss_bytes"],
            "cuda_peak_allocated_bytes": metrics["cuda_peak_allocated_bytes"],
            "cuda_peak_reserved_bytes": metrics["cuda_peak_reserved_bytes"],
            "device": torch.cuda.get_device_name(device),
            "configuration": settings["yeh_settings"],
            "outer_steps": settings["yeh_outer_steps"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:7")
    fit(torch.device(parser.parse_args().device))
