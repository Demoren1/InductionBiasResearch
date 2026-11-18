"""Explicit staged development search over budgets, families and optimizers."""

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .data import SETTINGS, episodes
from .replicate import OUTPUT, ROOT


def prepare(phase, settings):
    path = OUTPUT / f"{phase}_jobs.json"
    if path.exists():
        raise RuntimeError(f"Already prepared {path}")
    if (
        phase == "d2"
        and (OUTPUT / "refine_jobs.json").exists()
        and not (OUTPUT / "refinements_ready.json").exists()
    ):
        raise RuntimeError("Collect source refinements before preparing D2")
    jobs = []
    for setting in settings:
        dataset, _shots, validation, _outputs = SETTINGS[setting]
        root = ROOT / "study" / setting
        (root / "panels").mkdir(parents=True, exist_ok=True)
        if phase == "d1":
            models = json.loads((ROOT / "banks" / dataset / "models.json").read_text())
        else:
            candidates = []
            for metadata in (root / "fits/d1").glob("*.json"):
                result = json.loads(metadata.read_text())
                model = next(
                    row
                    for row in json.loads(
                        (ROOT / "banks" / dataset / "models.json").read_text()
                    )
                    if row["name"] == result["job"]["name"]
                )
                best = min(
                    result["metrics"].values(),
                    key=lambda r: (-r["accuracy"], r["ce"]),
                )
                candidates.append({**model, **best})
            families = {
                family
                for row in candidates
                for family in [row["family"], *row.get("family_aliases", [])]
            }
            advanced = {}
            for family in families:
                options = [
                    row
                    for row in candidates
                    if family in [row["family"], *row.get("family_aliases", [])]
                ]
                options.sort(key=lambda r: (-r["accuracy"], r["ce"], r["codes"]))
                for row in options[:3]:
                    advanced[row["name"]] = row
            # Always retain the original successful indexator as a fixed reference.
            reference = {
                "mnist": "asrr_input_m1024",
                "omniglot": "asrr_local_m2048",
            }.get(dataset)
            if reference:
                row = next(
                    r
                    for r in candidates
                    if reference in [r["name"], *r.get("name_aliases", [])]
                )
                advanced[row["name"]] = row
            models = list(advanced.values())
            registry = json.loads(
                (ROOT / "banks" / dataset / "models.json").read_text()
            )
            models.extend(r for r in registry if r.get("setting") == setting)
            save_json(
                OUTPUT / f"{setting}_d2_candidates.json",
                {
                    "candidates": models,
                    "screening_results_sha256": {
                        p.name: digest(p) for p in (root / "fits/d1").glob("*.json")
                    },
                },
            )
        seed = (
            7611172300 + list(SETTINGS).index(setting) + (100 if phase == "d2" else 0)
        )
        panel, provenance = episodes(
            setting, "development", 48 if phase == "d1" else 96, seed
        )
        np.savez_compressed(root / f"panels/{phase}.npz", **panel)
        save_json(root / f"panels/{phase}_provenance.json", provenance)
        for model in models:
            variants = (
                ("legacy6_200",)
                if phase == "d1"
                else (
                    (
                        "legacy6_200",
                        "legacy24_200",
                        "wide24_200",
                        "wide24_500",
                        "wide24_1000",
                    )
                    if validation
                    else ("support",)
                )
            )
            for variant in variants:
                jobs.append(
                    {
                        "setting": setting,
                        "name": model["name"],
                        "phase": phase,
                        "variant": variant,
                    }
                )
    save_json(path, jobs)
    print(
        json.dumps({"phase": phase, "jobs": len(jobs), "settings": settings}),
        flush=True,
    )


def run(phase, gpus, *, root=ROOT, output=OUTPUT, timing_first=None):
    """Run a shared queue, with optional timing jobs on one designated GPU first."""
    if timing_first is not None:
        timing_phase, timing_gpu = timing_first
        if timing_phase == phase or timing_gpu not in gpus:
            raise ValueError("Invalid timing phase or GPU")
    jobs = json.loads((output / f"{phase}_jobs.json").read_text())
    receipts = output / f"{phase}_jobs"
    receipts.mkdir(exist_ok=True)
    pending = queue.Queue()
    for number, job in enumerate(jobs):
        receipt = receipts / f"{number:04d}.json"
        if receipt.exists():
            if json.loads(receipt.read_text())["exit_code"] == 0:
                continue
            raise RuntimeError(f"Inspect failed job before resume: {receipt}")
        pending.put((number, job))

    def worker(gpu):
        if timing_first is not None and gpu == timing_gpu:
            run(timing_phase, [gpu], root=root, output=output)
        while True:
            try:
                number, job = pending.get_nowait()
            except queue.Empty:
                return
            module = (
                "experiments.asrr_tuning_v1.refine"
                if phase == "refine"
                else "experiments.asrr_tuning_v1.fit"
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                module,
                "--queue",
                phase,
                "--job",
                str(number),
                "--device",
                f"cuda:{gpu}",
            ]
            if phase != "refine":
                command.extend(("--root", str(root), "--output", str(output)))
            started = time.perf_counter()
            with (receipts / f"{number:04d}.log").open("w") as stream:
                result = subprocess.run(
                    command,
                    check=False,
                    stdout=stream,
                    stderr=stream,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                    },
                )
            record = {
                "number": number,
                **job,
                "gpu": gpu,
                "exit_code": result.returncode,
                "seconds": time.perf_counter() - started,
            }
            save_json(receipts / f"{number:04d}.json", record)
            print(json.dumps(record), flush=True)
            if result.returncode:
                raise RuntimeError(f"Failed job {record}")

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument(
        "--phase", choices=("d1", "refine", "d2", "target"), required=True
    )
    parser.add_argument("--settings", nargs="+")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.phase, args.settings)
    else:
        run(args.phase, args.gpus)
