"""Schedule measured GPU fitting separately from CPU structure proposals."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.sharing_structure_research_v1.blind_contract import save_json

PHASES = ("source", "seeds", "proposals", "fit", "select")


def run(root, allowed_devices):
    if not allowed_devices or any(device not in range(6) for device in allowed_devices):
        raise ValueError("Only nonempty subsets of GPU0-5 are allowed")
    protocol = json.loads((root / "protocol.json").read_text())
    jobs = [
        (case["name"], condition)
        for case in protocol["cases"]
        for condition in protocol["conditions"]
    ]
    active = {}
    failures = []
    failed_jobs = set()
    started = time.monotonic()
    last_update = 0.0
    devices = {
        row.split(",")[1].strip(): int(row.split(",")[0])
        for row in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
        ).splitlines()
    }

    def pending(job):
        case, condition = job
        for phase in PHASES:
            path = root / "metrics" / f"{case}_c{condition}_{phase}.json"
            if not path.exists():
                return phase
            if json.loads(path.read_text())["status"] != "complete":
                raise RuntimeError(f"Explicit recovery required for {path}")
        return None

    def launch(job, phase, slot):
        case, condition = job
        gpu = slot if isinstance(slot, int) else None
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
            OPENBLAS_NUM_THREADS="2",
            OMP_NUM_THREADS="2",
        )
        log = root / "logs" / f"{case}_c{condition}_{phase}.log"
        stream = log.open("w")
        command = [
            sys.executable,
            "-m",
            "experiments.paper_structure_benchmark_v1.current_method",
            "--root",
            str(root),
            "--case",
            case,
            "--condition",
            str(condition),
            "--phase",
            phase,
            "--device",
            "cpu" if gpu is None else "cuda:0",
        ]
        process = subprocess.Popen(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
        entry = {
            "job": job,
            "phase": phase,
            "gpu": gpu,
            "pid": process.pid,
            "started": time.monotonic(),
            "process": process,
            "stream": stream,
        }
        print(
            json.dumps(
                {k: v for k, v in entry.items() if k not in ("process", "stream")}
            ),
            flush=True,
        )
        return entry

    while True:
        for slot, entry in list(active.items()):
            code = entry["process"].poll()
            if code is None:
                continue
            entry["stream"].close()
            if code:
                failed_jobs.add(entry["job"])
                failures.append(
                    {
                        "job": entry["job"],
                        "phase": entry["phase"],
                        "exit_code": code,
                        "pid": entry["pid"],
                    }
                )
            del active[slot]
        unfinished = {
            job: pending(job)
            for job in jobs
            if job not in failed_jobs and pending(job) is not None
        }
        live_jobs = {entry["job"] for entry in active.values()}
        occupied = set()
        for row in subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines():
            fields = row.split(",")
            if len(fields) == 2 and fields[1].strip() in devices:
                occupied.add(devices[fields[1].strip()])
        for gpu in allowed_devices:
            if gpu in active or gpu in occupied:
                continue
            ready = [
                job
                for job, phase in unfinished.items()
                if job not in live_jobs and phase in ("source", "seeds", "fit")
            ]
            if ready:
                job = ready[0]
                active[gpu] = launch(job, unfinished[job], gpu)
                live_jobs.add(job)
        for worker in range(3):
            slot = f"cpu_{worker}"
            if slot in active:
                continue
            ready = [
                job
                for job, phase in unfinished.items()
                if job not in live_jobs and phase in ("proposals", "select")
            ]
            if ready:
                job = ready[0]
                active[slot] = launch(job, unfinished[job], slot)
                live_jobs.add(job)
        if time.monotonic() - last_update >= 15:
            save_json(
                root / "schedule_state.json",
                {
                    "pid": os.getpid(),
                    "seconds": time.monotonic() - started,
                    "completed": len(jobs) - len(unfinished) - len(failed_jobs),
                    "total": len(jobs),
                    "active": [
                        {
                            k: v
                            for k, v in entry.items()
                            if k not in ("process", "stream")
                        }
                        for entry in active.values()
                    ],
                    "failures": failures,
                },
            )
            last_update = time.monotonic()
        if not unfinished and not active:
            break
        time.sleep(2)
    save_json(
        root / "schedule_finished.json",
        {
            "seconds": time.monotonic() - started,
            "failures": failures,
        },
    )
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; inspect phase logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=list(range(6)))
    args = parser.parse_args()
    run(args.root.resolve(), args.devices)


if __name__ == "__main__":
    main()
