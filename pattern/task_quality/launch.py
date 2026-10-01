"""Run independent source-bank or meta fits only on the two approved GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


GPU_UUIDS = (
    "GPU-6784dc4e-6ec9-2266-5d23-85bf1b1c2af3",
    "GPU-f8501f2d-53bc-9087-6041-64ee69876325",
)
SEEDS = (8100, 8101, 8102, 8103)
WORKSPACE = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("bank", "meta", "tune", "test"), required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.extend and args.max_steps is None:
        raise SystemExit("An extension requires an explicit larger step cap.")
    if args.extend and args.stage in ("tune", "test"):
        raise SystemExit("Final child fits extend automatically from convergence status.")
    phase_name = args.stage + (f"_extend_{args.max_steps}" if args.extend else "")
    stage_dir = root / "orchestration" / phase_name
    if (root / "STOPPED_BY_USER.json").exists():
        raise SystemExit("User-stop marker present; resumption needs a new explicit request.")
    stage_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    snapshot = stage_dir / ("source_" + timestamp)
    snapshot.mkdir()
    hashes = {}
    for source in sorted(Path(__file__).parent.glob("*.py")):
        data = source.read_bytes()
        (snapshot / source.name).write_bytes(data)
        hashes[str(source.relative_to(WORKSPACE))] = hashlib.sha256(data).hexdigest()
    for relative in ("meta_pattern/data.py", "pattern/length_interp/mlp.py"):
        source = WORKSPACE / relative
        data = source.read_bytes()
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()

    jobs = []
    for index, seed in enumerate(SEEDS):
        seed_root = root / f"seed_{seed}"
        if args.stage == "bank":
            jobs.append({"name": f"bank_{seed}", "gpu_uuid": GPU_UUIDS[index % 2],
                         "command": [sys.executable, "-m", "pattern.task_quality.bank",
                                     "--out", str(seed_root / "bank"),
                                     "--seed", str(seed), "--device", "cuda"]})
        elif args.stage == "meta":
            bank_path = seed_root / "bank" / "bank.pt"
            if not bank_path.is_file():
                raise SystemExit(f"Missing completed source bank: {bank_path}")
            for method in ("transformer_mask", "free_mask"):
                jobs.append({"name": f"{method}_{seed}", "gpu_uuid": GPU_UUIDS[index % 2],
                             "command": [sys.executable, "-m", "pattern.task_quality.run", "meta",
                                         "--bank", str(bank_path), "--out", str(seed_root / method),
                                         "--seed", str(seed), "--method", method,
                                         "--device", "cuda"]})
        else:
            jobs.append({"name": f"{args.stage}_{seed}", "gpu_uuid": GPU_UUIDS[index % 2],
                         "command": [sys.executable, "-m", "pattern.task_quality.eval_run",
                                     args.stage, "--root", str(root), "--seed", str(seed),
                                     "--device", "cuda"]})
    for job in jobs:
        if args.max_steps is not None:
            job["command"] += ["--max-steps", str(args.max_steps)]
        if args.extend:
            job["command"].append("--extend")
    allocation = {"stage": args.stage, "started_utc": timestamp,
                  "approved_gpu_indices": [1, 2], "memory_considered_for_selection": False,
                  "source_sha256": hashes, "source_snapshot": str(snapshot), "jobs": jobs,
                  "utilization_note": "nvidia-smi measures all activity, including other users."}
    write_json(stage_dir / "allocation.json", allocation)
    processes: list[tuple[subprocess.Popen, object, dict]] = []

    def stop_children(signum: int, _frame: object) -> None:
        # Only explicitly created children belong to this experiment.
        for process, _, _ in processes:
            if process.poll() is None:
                process.terminate()
        write_json(stage_dir / "interrupted.json", {"signal": signum,
                   "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "recovery": "Latest atomic periodic checkpoints remain on disk."})
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)
    try:
        for job in jobs:
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=job["gpu_uuid"], OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            stream = (stage_dir / (job["name"] + ".log")).open("a", encoding="utf-8")
            process = subprocess.Popen(job["command"], cwd=WORKSPACE, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            processes.append((process, stream, job))
            print(f"Started {job['name']} pid={process.pid} on {job['gpu_uuid']}", flush=True)
        with (stage_dir / "gpu_activity.jsonl").open("a", encoding="utf-8") as activity:
            while any(process.poll() is None for process, _, _ in processes):
                observed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
                observation = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "gpu_all_users": observed.stdout.splitlines(),
                               "returncode": observed.returncode}
                activity.write(json.dumps(observation) + "\n")
                activity.flush()
                statuses = [{"name": job["name"], "pid": process.pid,
                             "returncode": process.poll()} for process, _, job in processes]
                write_json(stage_dir / "progress.json", statuses)
                if any(item["returncode"] not in (None, 0) for item in statuses):
                    for process, _, _ in processes:
                        if process.poll() is None:
                            process.terminate()
                    raise RuntimeError("A worker failed; sibling workers stopped with periodic checkpoints retained.")
                time.sleep(20)
        statuses = [{"name": job["name"], "pid": process.pid, "returncode": process.wait()}
                    for process, _, job in processes]
        write_json(stage_dir / "progress.json", statuses)
        if any(item["returncode"] != 0 for item in statuses):
            raise RuntimeError("Stage failed; inspect worker logs.")
        write_json(stage_dir / "completed.json", {"statuses": statuses,
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(f"{args.stage}: all workers completed", flush=True)
    finally:
        for process, stream, _ in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
            stream.close()


if __name__ == "__main__":
    main()
