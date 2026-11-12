"""Run identical fresh adapters and freeze every prediction before scoring."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def execute(output, gpus):
    plan = json.loads((output / "plan.json").read_text())
    root = Path(plan["root"])
    registered = json.loads((output / "registration.json").read_text())
    for path, expected in registered["code_sha256"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Registered source changed: {path}")
    inputs = json.loads((output / "inputs_frozen.json").read_text())
    for path, expected in inputs["public_sha256"].items():
        if digest(root / path) != expected:
            raise RuntimeError(f"Registered input changed: {path}")
    (output / "logs").mkdir()
    queue, active, finished = list(plan["jobs"]), {}, []
    started = time.monotonic()
    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            job = queue.pop(0)
            stream = (output / "logs" / f"{job['id']}.log").open("w")
            command = [
                sys.executable,
                "-u",
                "-m",
                "experiments.paper_structure_benchmark_v1.fitting.target_fit",
                "--root",
                str(root),
                "--case",
                job["case"],
                "--condition",
                str(job["condition"]),
                "--method",
                job["method"],
                "--device",
                "cuda:0",
            ]
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "2",
            }
            process = subprocess.Popen(
                command, env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
            active[gpu] = (process, stream, job)
            print(json.dumps({"started": job["id"], "gpu": gpu}), flush=True)
        for gpu, (process, stream, job) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            stream.close()
            del active[gpu]
            finished.append({"job": job["id"], "exit_code": status})
            print(
                json.dumps(
                    {
                        "finished": job["id"],
                        "exit_code": status,
                        "done": len(finished),
                        "total": len(plan["jobs"]),
                    }
                ),
                flush=True,
            )
        save_json(
            output / "execution_state.json",
            {
                "done": finished,
                "live": {gpu: job["id"] for gpu, (_, _, job) in active.items()},
                "pending": len(queue),
                "seconds": time.monotonic() - started,
            },
        )
        if active:
            time.sleep(3)
    failed = [item for item in finished if item["exit_code"]]
    if failed:
        raise RuntimeError(f"Numerical adaptation failed: {failed}")
    predictions = {}
    for job in plan["jobs"]:
        folder = root / "targets" / f"{job['case']}_c{job['condition']}" / job["method"]
        receipt = json.loads((folder / "access_receipt.json").read_text())
        if receipt["status"] != "complete" or receipt["violations"]:
            raise RuntimeError(f"Invalid access receipt: {job['id']}")
        predictions[job["id"]] = {
            "path": str(folder / "frozen_predictions.npz"),
            "sha256": digest(folder / "frozen_predictions.npz"),
            "receipt_sha256": digest(folder / "access_receipt.json"),
        }
    save_json(
        output / "prediction_freeze.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "predictions": predictions,
            "plan_sha256": digest(output / "plan.json"),
            "inputs_frozen_sha256": digest(output / "inputs_frozen.json"),
            "target_answers_opened": False,
            "wall_seconds": time.monotonic() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    args = parser.parse_args()
    execute(args.output.resolve(), args.gpus)


if __name__ == "__main__":
    main()
