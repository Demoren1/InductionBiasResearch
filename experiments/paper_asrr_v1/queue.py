"""Run one independent fitting subprocess per available GPU, with receipts."""

import argparse
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from experiments.sharing_structure_research_v1.blind_contract import save_json

from .prepare import GRID, OUTPUT, ROOT


def run(datasets, phase, gpus):
    for dataset in datasets:
        if phase == "refit":
            required = [OUTPUT / dataset / "extraction_access.json"]
        elif phase == "tune":
            required = [
                ROOT / dataset / f"fits/asrr_k{k}/selected.json" for k in GRID
            ]
        else:
            required = [OUTPUT / dataset / "observations_frozen.json"]
        for path in required:
            if not path.exists():
                raise RuntimeError(f"Preceding stage is incomplete: {path}")
    pending = queue.Queue()
    for dataset in datasets:
        for codes in (None,) if phase == "tune" else GRID:
            pending.put((dataset, codes))

    def worker(gpu):
        while True:
            try:
                dataset, codes = pending.get_nowait()
            except queue.Empty:
                return
            name = f"{phase}_{dataset}" + (
                f"_k{codes}" if codes is not None else ""
            )
            receipt = OUTPUT / "jobs" / f"{name}.json"
            if receipt.exists():
                prior = json.loads(receipt.read_text())
                if prior["exit_code"] == 0:
                    continue
                raise RuntimeError(
                    f"Inspect failed job before restarting: {receipt}"
                )
            command = [
                sys.executable,
                "-u",
                "-m",
                "experiments.paper_asrr_v1.worker",
                "--dataset",
                dataset,
                "--phase",
                phase,
                "--device",
                f"cuda:{gpu}",
            ]
            if codes is not None:
                command += ["--codes", str(codes)]
            started = time.perf_counter()
            environment = {
                **os.environ,
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
            }
            with (OUTPUT / "jobs" / f"{name}.log").open("w") as log:
                result = subprocess.run(
                    command, stdout=log, stderr=log, env=environment
                )
            record = {
                "job": name,
                "gpu": gpu,
                "command": command,
                "exit_code": result.returncode,
                "seconds": time.perf_counter() - started,
            }
            save_json(receipt, record)
            print(json.dumps(record), flush=True)
            if result.returncode:
                raise RuntimeError(f"Job failed: {name}")

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        list(executor.map(worker, gpus))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--phase", choices=("refit", "tune", "target"), required=True
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()
    run(args.datasets, args.phase, args.gpus)
