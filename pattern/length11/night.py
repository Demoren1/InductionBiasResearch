"""Resume-safe GPU scheduler for the nested 1–10 VAE experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys

import torch
from tqdm import tqdm


PATTERNS = ("1100", "1101", "1110", "1111", "0100", "1011",
            "0101", "1010", "0110", "1001")
SIZES = tuple(range(1, 11))
REPLICATES = tuple(range(4))
BANK_MLPS = 16384
SEARCH_STARTS = 16
SEARCH_MAX_STEPS = 30000
COEFFICIENT = .5


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    output: Path


def idle_gpus(requested: str) -> tuple[int, ...]:
    wanted = tuple(dict.fromkeys(int(value) for value in re.findall(r"\d+", requested)))
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    utilization = {}
    for line in raw.splitlines():
        gpu, value = line.split(",")
        utilization[int(gpu.strip())] = int(value.strip())
    absent = [gpu for gpu in wanted if gpu not in utilization]
    if absent:
        raise ValueError(f"unknown GPUs: {absent}")
    chosen = tuple(gpu for gpu in wanted if utilization[gpu] == 0)
    ignored = tuple(gpu for gpu in wanted if utilization[gpu] != 0)
    print(f"Idle GPUs: {chosen}; skipped busy GPUs: {ignored}", flush=True)
    if not chosen:
        raise RuntimeError("no requested GPU has utilization 0%")
    return chosen


def protocol(out: Path, reuse_pair: Path) -> None:
    value = {
        "patterns": PATTERNS, "sizes": SIZES, "replicates": REPLICATES,
        "bank_mlps": BANK_MLPS, "bank_steps": 4000, "bank_top_fraction": .1,
        "vae_beta": .1, "vae_early_stop_patience": 30,
        "search_starts": SEARCH_STARTS, "search_max_steps": SEARCH_MAX_STEPS,
        "search_coefficient": COEFFICIENT,
        "search_objective": "per-VAE hard top-32 mask BCE on every task + raw-logit MSE",
        "evaluation_mlp_repeats": 4, "evaluation_max_steps": 20000,
        "random_masks_per_run": 16,
        "positive_control": "direct gold target, evaluation only",
        "reuse_pair_root": str(reuse_pair.resolve()),
    }
    path = out / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text())
        current = json.loads(json.dumps(value))
        if previous != current:
            original = dict(current, sizes=list(range(3, 11)))
            if previous != original:
                raise ValueError(f"experiment protocol changed: {path}")
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            temp.replace(path)
            print("extended existing protocol to sizes 1–10", flush=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        temp.replace(path)


def reuse_pair_artifacts(out: Path, source: Path) -> None:
    """Reuse the already validated two-pattern bank and four VAE seeds."""
    for pattern in PATTERNS[:2]:
        for folder, names in (
            ("bank", [f"pattern_{pattern}.pt"]),
            ("vae", [f"pattern_{pattern}_rep{rep}.pt" for rep in REPLICATES]),
            ("positive_control", [f"pattern_{pattern}_rep{rep}.pt" for rep in REPLICATES]),
        ):
            for name in names:
                origin = source / folder / name
                destination = out / folder / name
                if destination.exists() or not origin.exists():
                    continue
                if folder == "bank":
                    payload = torch.load(origin, map_location="cpu", weights_only=True)
                    if (payload["pattern"] != pattern or
                            payload["protocol"]["bank_mlps"] != BANK_MLPS):
                        raise ValueError(f"incompatible existing bank: {origin}")
                elif folder == "vae":
                    payload = torch.load(origin, map_location="cpu", weights_only=True)
                    expected_rep = int(name.rsplit("rep", 1)[1].split(".", 1)[0])
                    if payload["pattern"] != pattern or payload["replicate"] != expected_rep:
                        raise ValueError(f"incompatible existing VAE: {origin}")
                else:
                    payload = torch.load(origin, map_location="cpu", weights_only=True)
                    if payload["pattern"] != pattern:
                        raise ValueError(f"incompatible positive control: {origin}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                shutil.copy2(origin, temporary)
                temporary.replace(destination)
                print(f"reused {destination}", flush=True)


def bank_jobs(out: Path) -> list[Job]:
    return [Job(f"bank_{pattern}", (
        sys.executable, "-m", "pattern.length11.bank", "--pattern", pattern,
        "--bank-mlps", str(BANK_MLPS), "--out", str(out)),
        out / "bank" / f"pattern_{pattern}.pt") for pattern in PATTERNS]


def vae_jobs(out: Path) -> list[Job]:
    return [Job(f"vae_{pattern}_rep{rep}", (
        sys.executable, "-m", "pattern.length11.vae", "--pattern", pattern,
        "--replicate", str(rep), "--out", str(out)),
        out / "vae" / f"pattern_{pattern}_rep{rep}.pt")
        for pattern in PATTERNS for rep in REPLICATES]


def control_jobs(out: Path) -> list[Job]:
    return [Job(f"control_{pattern}_rep{rep}", (
        sys.executable, "-m", "pattern.length11.positive_control",
        "--pattern", pattern, "--replicate", str(rep), "--out", str(out)),
        out / "positive_control" / f"pattern_{pattern}_rep{rep}.pt")
        for pattern in PATTERNS for rep in REPLICATES]


def search_jobs(out: Path, reuse_pair: Path, size: int) -> list[Job]:
    patterns = PATTERNS[:size]
    jobs = []
    for rep in REPLICATES:
        if size == 1:
            warm = None
        elif size == 3:
            warm = reuse_pair / "joint_long" / f"pair02_rep{rep}.pt"
        else:
            warm = out / "joint_multi" / f"n{size - 1}" / f"rep{rep}.pt"
        command = (
            sys.executable, "-m", "pattern.length11.multi_joint",
            "--patterns", *patterns, "--replicate", str(rep),
            "--out", str(out), "--starts", str(SEARCH_STARTS),
            "--max-steps", str(SEARCH_MAX_STEPS),
            "--coefficient", str(COEFFICIENT),
        )
        if warm is not None:
            command += ("--warm-start", str(warm))
        jobs.append(Job(f"joint_n{size}_rep{rep}", command,
                        out / "joint_multi" / f"n{size}" / f"rep{rep}.pt"))
        plain = (sys.executable, "-m", "pattern.length11.agreement",
                 "--patterns", *patterns, "--replicate", str(rep),
                 "--out", str(out))
        jobs.append(Job(f"plain_n{size}_rep{rep}", plain,
                        out / "agreement" / "_".join(patterns) / f"rep{rep}.pt"))
    return jobs


def evaluation_jobs(out: Path, size: int) -> list[Job]:
    patterns = PATTERNS[:size]
    return [Job(f"eval_n{size}_rep{rep}", (
        sys.executable, "-m", "pattern.length11.multi_evaluate",
        "--patterns", *patterns, "--replicate", str(rep), "--out", str(out)),
        out / "evaluation_multi" / f"n{size}" / f"rep{rep}.pt")
        for rep in REPLICATES]


def run_jobs(jobs: list[Job], gpus: tuple[int, ...], out: Path, label: str) -> None:
    available: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        available.put(gpu)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def worker(job: Job) -> tuple[Job, int, int, Path]:
        gpu = available.get()
        log = log_dir / f"{job.name}.log"
        try:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["OMP_NUM_THREADS"] = "2"
            env["PYTHONUNBUFFERED"] = "1"
            with log.open("w") as stream:
                code = subprocess.run(job.command, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT).returncode
            if code == 0 and not job.output.exists():
                code = 99
            return job, gpu, code, log
        finally:
            available.put(gpu)

    pending = [job for job in jobs if not job.output.exists()]
    with tqdm(total=len(jobs), desc=label, unit="job") as progress:
        progress.update(len(jobs) - len(pending))
        if not pending:
            return
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, job) for job in pending]
            for future in as_completed(futures):
                job, gpu, code, log = future.result()
                if code:
                    tail = "\n".join(log.read_text(errors="replace").splitlines()[-35:])
                    raise RuntimeError(f"{job.name} failed on GPU {gpu}, exit {code}; "
                                       f"log: {log}\n{tail}")
                progress.update(1)
                progress.set_postfix_str(job.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-pair", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0 1 2 3 4 5 6 7")
    parser.add_argument("--stage", choices=("all", "bank", "vae", "controls", "agreement",
                                            "evaluate", "report"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    reuse_pair = args.reuse_pair.resolve()
    gpus = () if args.stage == "report" or args.dry_run else idle_gpus(args.gpu_ids)
    if args.dry_run:
        print(f"Output: {out}\nRequested GPUs: {args.gpu_ids}\n"
              f"Patterns: {PATTERNS}\nSizes: {SIZES}\n"
              f"Bank jobs: {len(bank_jobs(out))}; VAE jobs: {len(vae_jobs(out))}; "
              f"positive controls: {len(control_jobs(out))}; "
              f"agreement jobs: {sum(len(search_jobs(out, reuse_pair, n)) for n in SIZES)}; "
              f"evaluation jobs: {sum(len(evaluation_jobs(out, n)) for n in SIZES)}")
        return
    protocol(out, reuse_pair)
    reuse_pair_artifacts(out, reuse_pair)
    if args.stage in ("all", "bank"):
        run_jobs(bank_jobs(out), gpus, out, "bank")
    if args.stage in ("all", "vae"):
        run_jobs(vae_jobs(out), gpus, out, "vae")
    if args.stage in ("all", "controls"):
        run_jobs(control_jobs(out), gpus, out, "gold controls")
    if args.stage == "all":
        for size in SIZES:
            jobs = search_jobs(out, reuse_pair, size)
            if size > SIZES[0]:
                jobs += evaluation_jobs(out, size - 1)
            run_jobs(jobs, gpus, out, f"n{size} search + eval")
        run_jobs(evaluation_jobs(out, SIZES[-1]), gpus, out, "n10 eval")
    elif args.stage == "agreement":
        for size in SIZES:
            run_jobs(search_jobs(out, reuse_pair, size), gpus, out,
                     f"n{size} agreement")
    elif args.stage == "evaluate":
        run_jobs([job for size in SIZES for job in evaluation_jobs(out, size)],
                 gpus, out, "evaluation")
    if args.stage in ("all", "report"):
        subprocess.run([sys.executable, "-m", "pattern.length11.multi_report",
                        "--out", str(out)], check=True)


if __name__ == "__main__":
    main()
