"""Execute a concrete dependency plan and share small jobs on allowed GPUs."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.sharing_structure_research_v1.blind_contract import save_json


# Preserve job identities from immutable pre-refactor execution plans.
MODULE_MOVES = {
    "experiments.paper_structure_benchmark_v1.prepare": "experiments.paper_structure_benchmark_v1.protocol.prepare",
    "experiments.paper_structure_benchmark_v1.prepare_digit_sum": "experiments.paper_structure_benchmark_v1.protocol.prepare_digit_sum",
    "experiments.paper_structure_benchmark_v1.prepare_omniglot": "experiments.paper_structure_benchmark_v1.protocol.prepare_omniglot",
    "experiments.paper_structure_benchmark_v1.real_boundary": "experiments.paper_structure_benchmark_v1.protocol.real_boundary",
    "experiments.paper_structure_benchmark_v1.target_export": "experiments.paper_structure_benchmark_v1.protocol.target_export",
    "experiments.paper_structure_benchmark_v1.real_target_export": "experiments.paper_structure_benchmark_v1.protocol.real_target_export",
    "experiments.paper_structure_benchmark_v1.variant_export": "experiments.paper_structure_benchmark_v1.protocol.variant_export",
    "experiments.paper_structure_benchmark_v1.native_msr_prepare": "experiments.paper_structure_benchmark_v1.protocol.native_msr_prepare",
    "experiments.paper_structure_benchmark_v1.adaptation": "experiments.paper_structure_benchmark_v1.fitting.adaptation",
    "experiments.paper_structure_benchmark_v1.numerical": "experiments.paper_structure_benchmark_v1.fitting.numerical",
    "experiments.paper_structure_benchmark_v1.target_fit": "experiments.paper_structure_benchmark_v1.fitting.target_fit",
    "experiments.paper_structure_benchmark_v1.real_target": "experiments.paper_structure_benchmark_v1.fitting.real_target",
    "experiments.paper_structure_benchmark_v1.real_validation": "experiments.paper_structure_benchmark_v1.fitting.real_validation",
    "experiments.paper_structure_benchmark_v1.real_bank_fit": "experiments.paper_structure_benchmark_v1.fitting.real_bank_fit",
    "experiments.paper_structure_benchmark_v1.refit": "experiments.paper_structure_benchmark_v1.fitting.refit",
    "experiments.paper_structure_benchmark_v1.continuous_refit": "experiments.paper_structure_benchmark_v1.fitting.continuous_refit",
    "experiments.paper_structure_benchmark_v1.binary_control": "experiments.paper_structure_benchmark_v1.fitting.binary_control",
    "experiments.paper_structure_benchmark_v1.real_dense": "experiments.paper_structure_benchmark_v1.fitting.real_dense",
    "experiments.paper_structure_benchmark_v1.native_msr_target": "experiments.paper_structure_benchmark_v1.fitting.native_msr_target",
    "experiments.paper_structure_benchmark_v1.cost_replay": "experiments.paper_structure_benchmark_v1.fitting.cost_replay",
    "experiments.paper_structure_benchmark_v1.bilevel": "experiments.paper_structure_benchmark_v1.methods.bilevel",
    "experiments.paper_structure_benchmark_v1.lutq": "experiments.paper_structure_benchmark_v1.methods.lutq",
    "experiments.paper_structure_benchmark_v1.msr_maml": "experiments.paper_structure_benchmark_v1.methods.msr_maml",
    "experiments.paper_structure_benchmark_v1.real_competitors": "experiments.paper_structure_benchmark_v1.methods.real_competitors",
    "experiments.paper_structure_benchmark_v1.scalable_projection": "experiments.paper_structure_benchmark_v1.methods.scalable_projection",
    "experiments.paper_structure_benchmark_v1.real_project": "experiments.paper_structure_benchmark_v1.methods.real_project",
    "experiments.paper_structure_benchmark_v1.functional_refine": "experiments.paper_structure_benchmark_v1.methods.functional_refine",
    "experiments.paper_structure_benchmark_v1.fleet": "experiments.paper_structure_benchmark_v1.execution.fleet",
    "experiments.paper_structure_benchmark_v1.export_watch": "experiments.paper_structure_benchmark_v1.execution.export_watch",
    "experiments.paper_structure_benchmark_v1.evaluate": "experiments.paper_structure_benchmark_v1.reporting.evaluate",
    "experiments.paper_structure_benchmark_v1.source_report": "experiments.paper_structure_benchmark_v1.reporting.source_report",
    "experiments.paper_structure_benchmark_v1.render_report": "experiments.paper_structure_benchmark_v1.reporting.render_report",
    "experiments.paper_structure_benchmark_v1.real_source_diagnosis": "experiments.paper_structure_benchmark_v1.reporting.real_source_diagnosis",
}


def identity(module, arguments):
    module = MODULE_MOVES.get(module, module)
    fields = {}
    fields["timing_only"] = "--timing-only" in arguments
    for key in ("root", "case", "condition", "method", "codes"):
        flag = f"--{key}"
        fields[key] = (
            arguments[arguments.index(flag) + 1]
            if flag in arguments
            else "0"
            if key == "condition"
            else ""
        )
    return module, tuple(fields.items())


def run(plan, per_gpu):
    configuration = json.loads(plan.read_text())
    jobs = configuration["jobs"]
    by_id = {row["id"]: row for row in jobs}
    keys = {identity(row["module"], row["args"]): row["id"] for row in jobs}
    if len(keys) != len(jobs):
        raise ValueError("Plan contains duplicate job identities")
    gpu_for_uuid = {
        row.split(",")[1].strip(): int(row.split(",")[0])
        for row in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
        ).splitlines()
    }
    children, done, failed = {}, set(), {}
    started, updated = time.monotonic(), 0.0
    while True:
        jobs = json.loads(plan.read_text())["jobs"]
        by_id = {row["id"]: row for row in jobs}
        keys = {identity(row["module"], row["args"]): row["id"] for row in jobs}
        if len(keys) != len(jobs):
            raise ValueError("Updated plan contains duplicate job identities")
        for key, entry in list(children.items()):
            status = entry["process"].poll()
            if status is None:
                continue
            entry["stream"].close()
            if status:
                failed[key] = {"exit_code": status, "log": entry["log"]}
            del children[key]
        occupied, live, foreign = {gpu: 0 for gpu in range(6)}, set(children), set()
        observed_pids = set()
        for row in subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines():
            fields = [item.strip() for item in row.split(",")]
            if len(fields) != 2 or fields[1] not in gpu_for_uuid:
                continue
            gpu = gpu_for_uuid[fields[1]]
            observed_pids.add(int(fields[0]))
            if gpu not in occupied:
                continue
            occupied[gpu] += 1
            try:
                args = (
                    (Path("/proc") / fields[0] / "cmdline")
                    .read_bytes()
                    .decode()
                    .split("\0")
                )
            except (FileNotFoundError, PermissionError):
                foreign.add(gpu)
                continue
            if "-m" not in args or not args[args.index("-m") + 1].startswith(
                "experiments.paper_structure_benchmark_v1."
            ):
                foreign.add(gpu)
                continue
            key = identity(args[args.index("-m") + 1], args)
            if key in keys:
                live.add(keys[key])
        for entry in children.values():
            if entry["pid"] not in observed_pids:
                occupied[entry["gpu"]] += 1
        for job in jobs:
            key = job["id"]
            if key in done or key in failed or key in live:
                continue
            metric = Path(job["metric"])
            if metric.exists():
                try:
                    status = json.loads(metric.read_text()).get("status", "complete")
                except json.JSONDecodeError:
                    continue
                if status == "complete":
                    done.add(key)
                else:
                    failed[key] = {"status": status, "metric": str(metric)}
            elif Path(job["registration"]).exists():
                failed[key] = {
                    "status": "partial_without_live_process",
                    "registration": job["registration"],
                }
        for gpu in range(6):
            if gpu in foreign or occupied[gpu] >= per_gpu:
                continue
            ready = [
                row
                for row in jobs
                if row["id"] not in done | set(failed) | live
                and all(Path(path).exists() for path in row.get("dependencies", []))
            ]
            if not ready:
                continue
            job = ready[0]
            key = job["id"]
            log = Path(job["log"])
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("w")
            environment = os.environ.copy()
            environment.update(
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                MPLCONFIGDIR="/tmp/ibr-mpl-cache",
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    MODULE_MOVES.get(job["module"], job["module"]),
                    *job["args"],
                ],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            children[key] = {
                "process": process,
                "stream": stream,
                "gpu": gpu,
                "pid": process.pid,
                "log": str(log),
            }
            live.add(key)
            occupied[gpu] += 1
            print(
                json.dumps({"launched": key, "gpu": gpu, "pid": process.pid}),
                flush=True,
            )
        if time.monotonic() - updated > 15:
            save_json(
                plan.with_suffix(".state.json"),
                {
                    "pid": os.getpid(),
                    "seconds": time.monotonic() - started,
                    "completed": len(done),
                    "total": len(jobs),
                    "live": sorted(live),
                    "children": {
                        key: {
                            field: value
                            for field, value in entry.items()
                            if field not in ("process", "stream")
                        }
                        for key, entry in children.items()
                    },
                    "failures": failed,
                    "per_gpu_limit": per_gpu,
                    "allowed_gpus": list(range(6)),
                },
            )
            updated = time.monotonic()
        remaining = set(by_id) - done - set(failed)
        if not remaining and not children:
            break
        if (
            remaining
            and not children
            and not live
            and not any(
                all(Path(path).exists() for path in by_id[key].get("dependencies", []))
                for key in remaining
            )
        ):
            save_json(
                plan.with_suffix(".blocked.json"),
                {"waiting_for_dependencies": sorted(remaining), "failures": failed},
            )
        time.sleep(3)
    save_json(
        plan.with_suffix(".finished.json"),
        {
            "seconds": time.monotonic() - started,
            "completed": len(done),
            "total": len(jobs),
            "failures": failed,
        },
    )
    if failed:
        raise SystemExit("Some jobs need explicit recovery; see retained logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--per-gpu", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run(args.plan.resolve(), args.per_gpu)


if __name__ == "__main__":
    main()
