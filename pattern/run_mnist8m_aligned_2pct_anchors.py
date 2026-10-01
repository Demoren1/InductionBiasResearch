"""Four common-reference controls for the 2% MNIST8m digit-pair experiment."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm


BASE = Path("pattern/outputs/mnist8m_raw_mlp_bce")
SOURCE = BASE / "width64_2pct"


@dataclass(frozen=True)
class Job:
    folder: Path
    label: str
    output: Path
    arguments: tuple[str, ...]


def selected_anchors() -> list[tuple[int, int]]:
    result = []
    for task in (0, 1):
        bank = torch.load(SOURCE / f"bank_task{task}.pt", map_location="cpu",
                          weights_only=True)
        count = len(bank["importance"])
        order = torch.randperm(count, generator=torch.Generator().manual_seed(3130 + task))
        n_val = max(32, round(.15 * count))
        result.extend((task, int(index)) for index in order[n_val:n_val + 2])
    return result


def gpu_idle(index: int) -> bool:
    output = subprocess.check_output(("nvidia-smi", "--query-gpu=index,utilization.gpu",
                                      "--format=csv,noheader"), text=True)
    utilization = {int(row.split(",")[0]): int(row.split(",")[1].strip().split()[0])
                   for row in output.splitlines()}
    return utilization[index] == 0


def run_jobs(jobs: list[Job], gpus: list[int], stage: str) -> None:
    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures = []
    lock = threading.Lock()
    progress = tqdm(total=len(jobs), desc=stage, unit="job")

    def worker(gpu: int) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            if job.output.exists():
                with lock:
                    progress.update()
                    progress.set_postfix_str(f"reuse {job.label}")
                pending.task_done()
                continue
            while not gpu_idle(gpu):
                time.sleep(5)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = job.folder / f"{job.label}.log"
            with log.open("w") as handle:
                code = subprocess.run((sys.executable, "-m", "pattern.mnist8m_raw_mlp_bce",
                                       "--out", str(job.folder), "--hidden", "64",
                                       "--density", ".02", "--stage", stage,
                                       *job.arguments), env=environment,
                                      stdout=handle, stderr=subprocess.STDOUT,
                                      check=False).returncode
            with lock:
                if code or not job.output.exists():
                    failures.append((str(log), code))
                progress.update()
                progress.set_postfix_str(f"GPU {gpu}: {job.label}")
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    progress.close()
    if failures:
        raise RuntimeError(f"{stage}: failed jobs {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--stage", choices=("all", "align", "vae", "search", "evaluate"),
                        default="all")
    parser.add_argument("--search-steps", type=int, default=24000)
    parser.add_argument("--evaluation-steps", type=int, default=30000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    anchors = selected_anchors()
    folders = []
    for task, index in anchors:
        folder = BASE / f"width64_2pct_anchor_t{task}_i{index}"
        folders.append(folder)
        if args.stage not in ("all", "align"):
            continue
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / "align.log"
        with log.open("w") as handle:
            subprocess.run((sys.executable, "-m", "pattern.align_mnist8m_importance_bank",
                            "--source", str(SOURCE), "--out", str(folder),
                            "--reference-task", str(task), "--reference-index", str(index)),
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        print(f"aligned to task {task} bank map {index}: {folder}", flush=True)
    if args.stage == "align":
        return
    if args.stage in ("all", "vae"):
        jobs = [Job(folder, f"vae_task{task}", folder / f"vae_task{task}.pt",
                    ("--task-index", str(task)))
                for folder in folders for task in (0, 1)]
        run_jobs(jobs, args.gpus, "vae")
    if args.stage == "vae":
        return
    if args.stage in ("all", "search"):
        jobs = [Job(folder, f"search_{objective}_{coefficient}",
                    folder / f"{'search_shared' if objective == 'shared' else 'search'}_lambda{coefficient}.pt",
                    ("--objective", objective, "--coefficient", coefficient,
                     "--search-steps", str(args.search_steps)))
                for folder in folders for objective in ("own", "shared")
                for coefficient in ("0", "1", "10")]
        run_jobs(jobs, args.gpus, "search")
    if args.stage == "search":
        return
    if args.stage in ("all", "evaluate"):
        jobs = [Job(folder, "evaluate", folder / "evaluation.pt",
                    ("--evaluation-steps", str(args.evaluation_steps)))
                for folder in folders]
        run_jobs(jobs, args.gpus, "evaluate")


if __name__ == "__main__":
    main()
