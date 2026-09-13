from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .common import load_protocol, prepare_task_data, protocol_payload, write_json
from .config import Config


ROOT = Path(__file__).resolve().parents[2]


def _signal_exit(signum, _frame):
    raise SystemExit(128 + signum)


def _prepare(out: Path, config: Config, smoke: bool) -> None:
    payload = protocol_payload(config, smoke)
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        current = json.loads(protocol_path.read_text())
        if current != payload:
            raise FileExistsError(f"{protocol_path} exists with a different protocol")
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "logs").mkdir()
        write_json(protocol_path, payload)
    prepare_task_data(out, config, payload["task_split"]["test_patterns"])


def _run_jobs(out: Path, stage: str, jobs: list[tuple[str, list[str]]], gpus: list[int]) -> None:
    """Run one serial queue per GPU and fail after stopping every owned worker."""
    queues = [jobs[index::len(gpus)] for index in range(len(gpus))]
    running: dict[int, tuple[str, subprocess.Popen, Any]] = {}
    positions = [0] * len(gpus)

    def start(slot: int) -> None:
        if positions[slot] >= len(queues[slot]):
            return
        name, arguments = queues[slot][positions[slot]]
        positions[slot] += 1
        log_path = out / "logs" / f"{stage}_{name}.log"
        log = log_path.open("a")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpus[slot]),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONUNBUFFERED": "1",
        }
        process = subprocess.Popen(
            [sys.executable, *arguments], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
        running[slot] = name, process, log
        print(f"START stage={stage} job={name} gpu={gpus[slot]} pid={process.pid}", flush=True)

    try:
        for slot in range(len(gpus)):
            start(slot)
        last_status = time.monotonic()
        while running:
            for slot, (name, process, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[slot]
                print(f"EXIT stage={stage} job={name} code={code}", flush=True)
                if code:
                    raise RuntimeError(f"stage {stage}, job {name} failed; inspect {out / 'logs'}")
                start(slot)
            if time.monotonic() - last_status >= 30:
                complete = sum(positions) - len(running)
                print(f"STATUS stage={stage} complete={complete}/{len(jobs)} running={len(running)}", flush=True)
                last_status = time.monotonic()
            if running:
                time.sleep(1)
    except BaseException:
        for _, process, _ in running.values():
            process.terminate()
        for _, process, log in running.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
        raise
    write_json(out / f"{stage}_done.json", {"jobs": [name for name, _ in jobs], "completed": True})


def _module(module: str, out: Path, *arguments: str) -> list[str]:
    return ["-m", module, "--out", str(out), *arguments]


def main() -> None:
    signal.signal(signal.SIGTERM, _signal_exit)
    signal.signal(signal.SIGINT, _signal_exit)
    parser = argparse.ArgumentParser(description="Run the fixed-k5 overnight experiment")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--vae-pairs", type=int,
        help="override the number of VAE pairs when creating a new non-smoke protocol",
    )
    parser.add_argument("--stage", choices=(
        "prepare", "bank", "vae", "agreement", "individual", "evaluate", "aggregate", "all"),
        default="all")
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain distinct GPU IDs")
    out = args.out.resolve()
    if args.vae_pairs is not None and (args.smoke or args.vae_pairs < 1):
        parser.error("--vae-pairs must be positive and cannot be combined with --smoke")
    config = Config.smoke() if args.smoke else Config()
    if args.vae_pairs is not None:
        config = replace(config, vae_pairs=args.vae_pairs)
    _prepare(out, config, args.smoke)
    config, protocol = load_protocol(out)
    pairs = config.pairs
    patterns = protocol["task_split"]["test_patterns"]
    selected = args.stage
    began = time.monotonic()

    if selected in ("bank", "all"):
        jobs = [(str(shard), _module(
            "pattern.fixed_k5_agreement.train", out, "--device", args.device, "bank",
            "--shard", str(shard), "--shards", str(len(args.gpus))))
                for shard in range(len(args.gpus))]
        _run_jobs(out, "bank", jobs, args.gpus)
    if selected in ("vae", "all"):
        jobs = [(str(seed), _module(
            "pattern.fixed_k5_agreement.train", out, "--device", args.device, "vae", "--seed", str(seed)))
                for seed in config.vae_seeds]
        _run_jobs(out, "vae", jobs, args.gpus)
    if selected in ("agreement", "all"):
        jobs = [(f"{a}_{b}", _module(
            "pattern.fixed_k5_agreement.search", out, "--device", args.device, "--pair", str(a), str(b),
            "agreement")) for a, b in pairs]
        _run_jobs(out, "agreement", jobs, args.gpus)
    if selected in ("individual", "all"):
        jobs = [(f"{a}_{b}_{pattern}", _module(
            "pattern.fixed_k5_agreement.search", out, "--device", args.device, "--pair", str(a), str(b),
            "--pattern", pattern, "task-z")) for a, b in pairs for pattern in patterns]
        _run_jobs(out, "individual", jobs, args.gpus)
    if selected in ("evaluate", "all"):
        jobs = [(f"{a}_{b}_{pattern}", _module(
            "pattern.fixed_k5_agreement.search", out, "--device", args.device, "--pair", str(a), str(b),
            "--pattern", pattern, "evaluate")) for a, b in pairs for pattern in patterns]
        _run_jobs(out, "evaluate", jobs, args.gpus)
    if selected in ("aggregate", "all"):
        for pair in pairs:
            subprocess.run([sys.executable, *_module(
                "pattern.fixed_k5_agreement.search", out, "--device", "cpu", "--pair",
                str(pair[0]), str(pair[1]), "summarize")], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "pattern.fixed_k5_agreement.report", "--out", str(out)],
                       cwd=ROOT, check=True)
        write_json(out / "done.json", {"completed": True, "elapsed_seconds_this_invocation": time.monotonic() - began,
                                        "result": "RESULTS.md", "summary": "summary.json"})


if __name__ == "__main__":
    main()
