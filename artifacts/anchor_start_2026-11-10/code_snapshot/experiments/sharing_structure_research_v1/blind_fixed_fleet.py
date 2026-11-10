"""Schedule independent learner processes on explicitly allowed GPU slots."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(root, devices, stage="full"):
    if any(device not in (0, 1, 2, 3) for device in devices):
        raise ValueError("Only user-allowed GPU0-3 may be used")
    protocol = json.loads((root / "protocol.json").read_text())
    module, owner, marker = {
        "full": ("blind_fixed_job", "fits", "job_complete.json"),
        "dense": ("blind_dense_job", "dense", "prepared.json"),
    }[stage]
    pending = [
        (case["name"], condition)
        for condition in reversed(protocol["conditions"])
        for case in protocol["cases"]
        if not (root / owner / f"{case['name']}_c{condition}" / marker).exists()
    ]
    active, completed, failed = {}, [], []
    started, last_update = time.monotonic(), 0.0
    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            case, condition = pending.pop(0)
            path = root / "logs" / f"{case}_c{condition}.log"
            stream = path.open("w")
            environment = os.environ.copy()
            environment.update(
                CUDA_VISIBLE_DEVICES=str(device),
                OPENBLAS_NUM_THREADS="2",
                OMP_NUM_THREADS="2",
            )
            command = [
                sys.executable,
                "-m",
                f"experiments.sharing_structure_research_v1.{module}",
                "--root",
                str(root),
                "--case",
                case,
                "--condition",
                str(condition),
                "--device",
                "cuda:0",
            ]
            process = subprocess.Popen(
                command, env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
            active[device] = {
                "process": process,
                "stream": stream,
                "case": case,
                "condition": condition,
                "started": time.monotonic(),
                "log": str(path),
            }
            print(
                json.dumps(
                    {
                        "started": case,
                        "condition": condition,
                        "gpu": device,
                        "pid": process.pid,
                    }
                ),
                flush=True,
            )
        for device, job in list(active.items()):
            code = job["process"].poll()
            if code is None:
                continue
            job["stream"].close()
            row = {key: job[key] for key in ("case", "condition", "log")}
            row.update(
                gpu=device, exit_code=code, seconds=time.monotonic() - job["started"]
            )
            (completed if code == 0 else failed).append(row)
            print(json.dumps({"finished": row}), flush=True)
            del active[device]
        if time.monotonic() - last_update >= 20:
            state = {
                "seconds": time.monotonic() - started,
                "pending": len(pending),
                "active": [
                    {
                        "case": j["case"],
                        "condition": j["condition"],
                        "gpu": device,
                        "pid": j["process"].pid,
                        "seconds": time.monotonic() - j["started"],
                    }
                    for device, j in active.items()
                ],
                "completed": completed,
                "failed": failed,
            }
            (root / "fleet_status.json").write_text(json.dumps(state, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "live": state["active"],
                        "pending": len(pending),
                        "completed": len(completed),
                        "failed": len(failed),
                    }
                ),
                flush=True,
            )
            last_update = time.monotonic()
        time.sleep(2)
    (root / "fleet_complete.json").write_text(
        json.dumps(
            {
                "seconds": time.monotonic() - started,
                "completed": completed,
                "failed": failed,
            },
            indent=2,
        )
        + "\n"
    )
    if failed:
        raise SystemExit(f"{len(failed)} scientific jobs failed; inspect their logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--stage", choices=("full", "dense"), default="full")
    args = parser.parse_args()
    run(args.root.resolve(), args.devices, args.stage)


if __name__ == "__main__":
    main()
