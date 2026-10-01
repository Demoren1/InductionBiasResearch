"""Run MNIST8m importance-map VAE experiments for narrower MLPs."""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


BASE = Path("pattern/outputs/mnist8m_raw_mlp_bce")
@dataclass(frozen=True)
class Job:
    width: int
    density: float
    stage: str
    extra: tuple[str, ...] = ()

    @property
    def folder(self) -> Path:
        return BASE / f"width{self.width}_{round(100 * self.density)}pct"

    @property
    def label(self) -> str:
        return "_".join((f"width{self.width}", f"density{self.density:g}",
                         self.stage, *self.extra))


def run_jobs(jobs: list[Job], gpus: list[int], *, epochs: int,
             search_steps: int, evaluation_steps: int) -> None:
    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures: list[tuple[str, int]] = []
    lock = threading.Lock()
    progress = tqdm(total=len(jobs), desc=jobs[0].stage, unit="job")

    def worker(gpu: int) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            args = [sys.executable, "-m", "pattern.mnist8m_raw_mlp_bce",
                    "--out", str(job.folder), "--hidden", str(job.width),
                    "--density", str(job.density), "--stage", job.stage,
                    "--device", f"cuda:{gpu}"]
            if job.stage == "vae":
                args.extend(("--vae-epochs", str(epochs), "--task-index",
                             job.extra[0]))
            elif job.stage == "search":
                args.extend(("--search-steps", str(search_steps),
                             "--objective", job.extra[0],
                             "--coefficient", job.extra[1]))
            elif job.stage == "evaluate":
                needed_steps = (max(evaluation_steps, 60000)
                                if job.width == 16 and job.density == .02
                                else evaluation_steps)
                args.extend(("--evaluation-steps", str(needed_steps)))
            log = job.folder / f"{job.label}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as handle:
                code = subprocess.run(args, stdout=handle, stderr=subprocess.STDOUT,
                                      check=False).returncode
            with lock:
                if code:
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
        raise RuntimeError(f"{len(failures)} jobs failed: {failures}")


def prepare(density: float, widths: list[int]) -> None:
    for width in widths:
        source = BASE / ("pair38" if width == 64 else f"width{width}_bank20")
        target = BASE / f"width{width}_{round(100 * density)}pct"
        subprocess.run((sys.executable, "-m", "pattern.prepare_mnist8m_5pct",
                        "--source", str(source), "--out", str(target),
                        "--density", str(density)), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=float, choices=(.05, .02), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 7])
    parser.add_argument("--widths", type=int, nargs="+", choices=(16, 32, 64),
                        default=[32, 16])
    parser.add_argument("--vae-epochs", type=int, default=200)
    parser.add_argument("--search-steps", type=int, default=24000)
    parser.add_argument("--evaluation-steps", type=int)
    args = parser.parse_args()
    if args.evaluation_steps is None:
        args.evaluation_steps = 30000 if args.density == .02 else 15000
    prepare(args.density, args.widths)
    vaes = [Job(width, args.density, "vae", (str(task),))
            for width in args.widths for task in (0, 1)]
    run_jobs(vaes, args.gpus, epochs=args.vae_epochs,
             search_steps=args.search_steps, evaluation_steps=args.evaluation_steps)
    searches = [Job(width, args.density, "search", (objective, str(coefficient)))
                for width in args.widths for objective in ("own", "shared")
                for coefficient in (0, 1, 10)]
    run_jobs(searches, args.gpus, epochs=args.vae_epochs,
             search_steps=args.search_steps, evaluation_steps=args.evaluation_steps)
    evaluations = [Job(width, args.density, "evaluate") for width in args.widths]
    run_jobs(evaluations, args.gpus, epochs=args.vae_epochs,
             search_steps=args.search_steps, evaluation_steps=args.evaluation_steps)


if __name__ == "__main__":
    main()
