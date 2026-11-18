"""Measure a fixed LUT-Q configuration on the width-64 Omniglot study."""

import argparse
import resource
import time

import numpy as np
import torch

from experiments.asrr_tuning_v1.study import run as run_fits
from experiments.paper_compact_v1.transfer import save_model
from experiments.paper_structure_benchmark_v1.fitting.real_bank_fit import candidates
from experiments.paper_structure_benchmark_v1.methods.real_competitors import run
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .source import BANKS, CODES, OUTPUT, PREVIOUS, ROOT, read_json
from .transfer import PANELS, PLAN, job, score, select_development, write_target_queues

ARCHIVE = "compact_experiments_2026-11-18"
REGISTRATION = OUTPUT / "lutq_registration.json"
DEVELOPMENT_JOBS = OUTPUT / "lutq_development_jobs.json"


def fit_source(device):
    compact = OUTPUT.parent / ARCHIVE
    previous_choice = read_json(compact / "omniglot10/choices_frozen.json")["lutq"]
    original = read_json(
        ROOT.parent / ARCHIVE / "omniglot_source/source/lutq_k1024/registered.json"
    )
    scenario = int(previous_choice["name"].split("_")[1].removeprefix("r"))
    setting = original["settings"][scenario]
    restarts = [row for row in original["settings"] if row == setting]
    width64 = read_json(OUTPUT / "registration.json")
    for panel, expected in width64["source_panel_sha256"].items():
        if digest(ROOT / "public" / f"{panel}.npz") != expected:
            raise ValueError(f"Changed Omniglot source panel: {panel}")
    config = {
        "previous_choice": previous_choice["name"],
        "source_tasks": width64["source_tasks"],
        "codes": CODES,
        "setting": setting,
        "source_restarts": len(restarts),
        "source_steps": original["steps"],
        "numeric_rule": width64["numeric_rule"],
        "target_plan_sha256": digest(PLAN),
        "candidate_screen": "Up to 8 distinct assignments from the four source tasks with lowest numerical-validation loss",
    }
    save_json(REGISTRATION, config)
    started = time.perf_counter()
    run(
        ROOT,
        "lutq",
        CODES,
        device,
        config["source_steps"],
        settings=restarts,
        profile=False,
    )
    rows = []
    for name, index, task in candidates(ROOT, "lutq", CODES):
        supported = np.count_nonzero(index)
        if not supported:
            continue
        scale = np.sqrt(index.size / supported)
        row = save_model(BANKS, f"lutq_{name}", CODES, index=index, scale=scale)
        row.update(
            source_task=config["source_tasks"][task],
            used_codes=len(np.unique(np.abs(index[index != 0]))),
            active_connections=int(supported),
        )
        rows.append(row)
    if not rows:
        raise ValueError("LUT-Q produced no nonzero assignments")
    metrics = read_json(ROOT / "source" / f"lutq_k{CODES}/metrics.json")
    save_json(
        OUTPUT / "lutq_source.json",
        {
            "candidates": rows,
            "seconds": time.perf_counter() - started,
            "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "cuda_peak_allocated_bytes": metrics["cuda_peak_allocated_bytes"],
            "cuda_peak_reserved_bytes": metrics["cuda_peak_reserved_bytes"],
            "device": torch.cuda.get_device_name(device),
            "configuration": config,
        },
    )
    dev = read_json(PREVIOUS / "development_panel.json")
    if digest(PANELS / "development.npz") != dev["sha256"]:
        raise ValueError("Changed Omniglot development panel")
    save_json(
        DEVELOPMENT_JOBS,
        [job(row, "development", dev["sha256"], seed=7611172201) for row in rows],
    )


def freeze():
    config = read_json(REGISTRATION)
    if digest(PLAN) != config["target_plan_sha256"]:
        raise ValueError("Changed Omniglot target plan")
    selected = select_development(read_json(DEVELOPMENT_JOBS), config["numeric_rule"])
    choice = {key: selected[key] for key in ("name", "model_sha256", "accuracy", "ce")}
    save_json(OUTPUT / "choices_lutq.json", choice)
    write_target_queues({"lutq": choice}, read_json(PLAN), prefix="lutq_")
    print(f"Selected {choice['name']} on development ({choice['accuracy']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("source", "development", "freeze", "benchmark", "score")
    )
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--gpus", nargs="+", type=int, default=[7])
    args = parser.parse_args()
    if args.action == "source":
        fit_source(torch.device(args.device))
    elif args.action == "development":
        run_fits("lutq_development", args.gpus, root=ROOT, output=OUTPUT)
    elif args.action == "freeze":
        freeze()
    elif args.action == "benchmark":
        run_fits(
            "lutq_target",
            args.gpus,
            root=ROOT,
            output=OUTPUT,
            timing_first=("lutq_timing", args.gpus[-1]),
        )
    else:
        score(
            added_methods=("yeh", "asrr_lloyd", "lutq"),
            result_file="results_with_lutq.json",
        )
