"""Keep CPU structure proposals separate from packed numerical GPU fitting."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(root, devices, cpu_workers):
    if any(device not in (0, 1, 2, 3) for device in devices):
        raise ValueError("Only allowed GPU0-3 may be used")
    protocol = json.loads((root / "protocol.json").read_text())
    jobs = [
        (case["name"], c)
        for c in reversed(protocol["conditions"])
        for case in protocol["cases"]
    ]
    cpu, gpu, failures = {}, {}, []
    failed = set()
    started, last_update = time.monotonic(), 0.0

    def complete(job):
        case, condition = job
        return (
            root / "fits" / f"{case}_c{condition}" / "joint_resume_finish.json"
        ).exists()

    def ready(job):
        case, condition = job
        return (root / "fits" / f"{case}_c{condition}" / "joint_bank.json").exists()

    def launch(job, phase, device):
        case, condition = job
        log = root / "logs" / f"joint_{phase}_{case}_c{condition}.log"
        stream = log.open("w")
        environment = os.environ.copy()
        environment.update(
            OPENBLAS_NUM_THREADS="2",
            OMP_NUM_THREADS="2",
            CUDA_VISIBLE_DEVICES="" if device is None else str(device),
        )
        command = [
            sys.executable,
            "-m",
            "experiments.sharing_structure_research_v1.blind_joint_resume",
            "--root",
            str(root),
            "--case",
            case,
            "--condition",
            str(condition),
            "--phase",
            phase,
            "--device",
            "cpu" if device is None else "cuda:0",
        ]
        process = subprocess.Popen(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
        print(
            json.dumps(
                {
                    "started": case,
                    "condition": condition,
                    "phase": phase,
                    "gpu": device,
                    "pid": process.pid,
                }
            ),
            flush=True,
        )
        return {
            "process": process,
            "stream": stream,
            "job": job,
            "phase": phase,
            "started": time.monotonic(),
            "log": str(log),
            "gpu": device,
        }

    while True:
        for pool in (cpu, gpu):
            for slot, entry in list(pool.items()):
                code = entry["process"].poll()
                if code is None:
                    continue
                entry["stream"].close()
                case, condition = entry["job"]
                row = {
                    "case": case,
                    "condition": condition,
                    "phase": entry["phase"],
                    "exit_code": code,
                    "seconds": time.monotonic() - entry["started"],
                    "gpu": entry["gpu"],
                    "log": entry["log"],
                }
                print(json.dumps({"finished": row}), flush=True)
                if code:
                    failures.append(row)
                    failed.add(entry["job"])
                del pool[slot]
        active = {a["job"] for pool in (cpu, gpu) for a in pool.values()}
        unfinished = [job for job in jobs if job not in failed and not complete(job)]
        for device in devices:
            if device in gpu:
                continue
            available = [job for job in unfinished if job not in active and ready(job)]
            if available:
                job = available[0]
                gpu[device] = launch(job, "finish", device)
                active.add(job)
        for worker in range(cpu_workers):
            if worker in cpu:
                continue
            available = [
                job for job in unfinished if job not in active and not ready(job)
            ]
            if available:
                job = available[0]
                cpu[worker] = launch(job, "prepare", None)
                active.add(job)
        if not unfinished and not cpu and not gpu:
            break
        if time.monotonic() - last_update >= 20:
            live = [
                {
                    "case": a["job"][0],
                    "condition": a["job"][1],
                    "phase": a["phase"],
                    "gpu": a["gpu"],
                    "pid": a["process"].pid,
                    "seconds": time.monotonic() - a["started"],
                }
                for pool in (cpu, gpu)
                for a in pool.values()
            ]
            state = {
                "seconds": time.monotonic() - started,
                "completed": sum(complete(j) for j in jobs),
                "total": len(jobs),
                "active": live,
                "failures": failures,
            }
            (root / "joint_fleet_status.json").write_text(
                json.dumps(state, indent=2) + "\n"
            )
            print(json.dumps(state), flush=True)
            last_update = time.monotonic()
        time.sleep(2)
    (root / "joint_fleet_complete.json").write_text(
        json.dumps(
            {
                "seconds": time.monotonic() - started,
                "failures": failures,
                "completed": sum(complete(j) for j in jobs),
            },
            indent=2,
        )
        + "\n"
    )
    if failures:
        raise SystemExit(f"{len(failures)} scientific jobs failed; inspect logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--cpu-workers", type=int, default=3)
    args = parser.parse_args()
    run(args.root.resolve(), args.devices, args.cpu_workers)


if __name__ == "__main__":
    main()
