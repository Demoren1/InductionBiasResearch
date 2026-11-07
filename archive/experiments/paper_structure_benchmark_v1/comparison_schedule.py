"""Resume native competitor jobs while respecting live processes and GPU owners."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.sharing_structure_research_v1.blind_contract import save_json


def run(root):
    protocol = json.loads((root / "protocol.json").read_text())
    jobs = []
    for case in protocol["cases"]:
        for condition in protocol["conditions"]:
            for method in ("lutq", "yeh", "msr_maml"):
                phases = (
                    ("train", "hard")
                    if method == "lutq"
                    else (
                        ("train", "hard", "continuous")
                        if method == "yeh"
                        else ("train", "continuous")
                    )
                )
                for phase in phases:
                    jobs.append((case["name"], condition, method, phase))
    uuid_to_gpu = {
        line.split(",")[1].strip(): int(line.split(",")[0])
        for line in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
        ).splitlines()
    }
    children, done, failures = {}, set(), {}
    started, last_update = time.monotonic(), 0.0

    def paths(job):
        case, condition, method, phase = job
        folder = root / "competitors" / f"{case}_c{condition}" / method
        if phase == "train":
            return folder / "metrics.json", folder / "registered.json"
        if phase == "hard":
            return folder / "selected.json", folder / "refit_registered.json"
        return (
            folder / "selected_continuous.json",
            folder / "continuous_refit_registered.json",
        )

    def command(job):
        case, condition, method, phase = job
        if phase == "train":
            module = {"lutq": "lutq", "yeh": "bilevel", "msr_maml": "msr_maml"}[method]
            extra = ["--method", method] if method == "yeh" else []
        else:
            module = "refit" if phase == "hard" else "continuous_refit"
            extra = ["--method", method]
        return [
            sys.executable,
            "-m",
            f"experiments.paper_structure_benchmark_v1.{module}",
            "--root",
            str(root),
            "--case",
            case,
            "--condition",
            str(condition),
            "--device",
            "cuda:0",
            *extra,
        ]

    def discover():
        occupied, live_jobs = set(), set()
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()
        for row in rows:
            fields = row.split(",")
            if len(fields) != 2 or fields[1].strip() not in uuid_to_gpu:
                continue
            occupied.add(uuid_to_gpu[fields[1].strip()])
            try:
                args = (
                    (Path("/proc") / fields[0].strip() / "cmdline")
                    .read_bytes()
                    .decode()
                    .split("\0")
                )
            except (FileNotFoundError, PermissionError):
                continue
            if str(root) not in args or "--case" not in args or "-m" not in args:
                continue
            module = args[args.index("-m") + 1].rsplit(".", 1)[-1]
            case = args[args.index("--case") + 1]
            condition = (
                int(args[args.index("--condition") + 1]) if "--condition" in args else 0
            )
            if module == "lutq":
                method, phase = "lutq", "train"
            elif module == "msr_maml":
                method, phase = "msr_maml", "train"
            elif module in ("bilevel", "refit", "continuous_refit"):
                method = args[args.index("--method") + 1]
                phase = {
                    "bilevel": "train",
                    "refit": "hard",
                    "continuous_refit": "continuous",
                }[module]
            else:
                continue
            live_jobs.add((case, condition, method, phase))
        return occupied, live_jobs

    while True:
        for gpu, entry in list(children.items()):
            code = entry["process"].poll()
            if code is None:
                continue
            entry["stream"].close()
            if code:
                failures[entry["job"]] = {"exit_code": code, "log": entry["log"]}
            del children[gpu]
        occupied, discovered = discover()
        live_jobs = discovered | {entry["job"] for entry in children.values()}
        for job in jobs:
            if job in done or job in failures:
                continue
            metric, registered = paths(job)
            if metric.exists() and job not in live_jobs:
                try:
                    metadata = json.loads(metric.read_text())
                except json.JSONDecodeError:
                    continue
                if metadata.get("status", "complete") == "complete":
                    done.add(job)
                else:
                    failures[job] = {"status": "failed", "metrics": str(metric)}
            elif registered.exists() and job not in live_jobs:
                failures[job] = {
                    "status": "partial_without_live_process",
                    "registration": str(registered),
                }
        for gpu in range(6):
            if gpu in occupied or gpu in children:
                continue
            ready = []
            for job in jobs:
                if job in done or job in failures or job in live_jobs:
                    continue
                case, condition, method, phase = job
                if phase != "train" and (case, condition, method, "train") not in done:
                    continue
                source = root / "metrics" / f"{case}_c{condition}_source.json"
                if source.exists():
                    ready.append(job)
            if not ready:
                continue
            job = ready[0]
            case, condition, method, phase = job
            path = root / "logs" / f"queue_{method}_{case}_c{condition}_{phase}.log"
            stream = path.open("w")
            environment = os.environ.copy()
            environment.update(
                CUDA_VISIBLE_DEVICES=str(gpu),
                OPENBLAS_NUM_THREADS="2",
                OMP_NUM_THREADS="2",
            )
            process = subprocess.Popen(
                command(job), env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
            children[gpu] = {
                "job": job,
                "process": process,
                "stream": stream,
                "pid": process.pid,
                "log": str(path),
            }
            live_jobs.add(job)
            print(
                json.dumps({"launched": job, "gpu": gpu, "pid": process.pid}),
                flush=True,
            )
        if time.monotonic() - last_update >= 15:
            save_json(
                root / "comparison_schedule_state.json",
                {
                    "pid": os.getpid(),
                    "seconds": time.monotonic() - started,
                    "completed": len(done),
                    "total": len(jobs),
                    "live_jobs": sorted(live_jobs),
                    "owned_children": [
                        {"gpu": gpu, "job": entry["job"], "pid": entry["pid"]}
                        for gpu, entry in children.items()
                    ],
                    "failures": [
                        {"job": job, **details} for job, details in failures.items()
                    ],
                },
            )
            last_update = time.monotonic()
        remaining = [job for job in jobs if job not in done and job not in failures]
        if not remaining and not children and not (discovered & set(jobs)):
            break
        time.sleep(5)
    save_json(
        root / "comparison_schedule_finished.json",
        {
            "seconds": time.monotonic() - started,
            "completed": len(done),
            "total": len(jobs),
            "failures": [{"job": job, **details} for job, details in failures.items()],
        },
    )
    if failures:
        raise SystemExit("Some native competitor stages require explicit recovery")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
