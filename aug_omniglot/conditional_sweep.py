"""Run matched static, mean-pool and transformer U experiments on idle GPUs.

The terminal shows a live step progress bar; each run also has a detailed log.
Completed runs are skipped and interrupted runs resume from their checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

from .conditional_run import DEFAULT_ROOT


ARMS = ("static", "mean", "transformer")


def parse_csv(text: str, convert):
    items = [convert(part.strip()) for part in text.split(",") if part.strip()]
    if not items or len(set(items)) != len(items):
        raise ValueError("Need at least one distinct value")
    return items


def idle_devices() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    devices = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 2 and fields[1] != "[Not Supported]" and int(fields[1]) == 0:
            devices.append(int(fields[0]))
    return devices


def run_dir(root: Path, arm: str, seed: int) -> Path:
    return root / f"{arm}_seed{seed}"


def budget(arm: str, max_steps: int, test_tasks: int) -> int:
    return max_steps + test_tasks * (1 if arm == "static" else 2)


def progress(run: Path) -> dict:
    path = run / "progress.json"
    if not path.exists():
        return {"step": 0, "stage": "starting"}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"step": 0, "stage": "starting"}


def print_summary(root: Path, jobs: list[tuple[str, int]]) -> None:
    rows = []
    for arm, seed in jobs:
        path = run_dir(root, arm, seed) / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            rows.append(result)
            wrong = result["test_wrong_context"]
            wrong_text = (f"{wrong['accuracy']:.2%}/{wrong['loss']:.3f}"
                          if wrong is not None else "—")
            print(f"{arm:11s} seed={seed} best={result['best_step']:5d} "
                  f"test={result['test']['accuracy']:.2%}/"
                  f"{result['test']['loss']:.3f} "
                  f"wrong_support={wrong_text} (accuracy/loss)")
    if not rows:
        return
    summary = {"runs": rows, "by_arm": {}}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        accuracies = [row["test"]["accuracy"] for row in arm_rows]
        mean = sum(accuracies) / len(accuracies)
        spread = (sum((x - mean) ** 2 for x in accuracies) / (len(accuracies) - 1)) ** .5 \
            if len(accuracies) > 1 else None
        summary["by_arm"][arm] = {"seeds": [row["seed"] for row in arm_rows],
                                  "test_accuracy_mean": mean,
                                  "test_accuracy_sd": spread}
        if arm != "static":
            gains = [row["test"]["accuracy"] -
                     row["test_wrong_context"]["accuracy"] for row in arm_rows]
            summary["by_arm"][arm]["conditioning_gain_mean"] = \
                sum(gains) / len(gains)
            loss_gains = [row["test_wrong_context"]["loss"] -
                          row["test"]["loss"] for row in arm_rows]
            summary["by_arm"][arm]["conditioning_loss_gain_mean"] = \
                sum(loss_gains) / len(loss_gains)
        print(f"{arm:11s} mean={mean:.2%}" +
              (f" ± {spread:.2%} across seeds" if spread is not None else ""))
        if arm != "static":
            print(f"             matched minus wrong support: "
                  f"{summary['by_arm'][arm]['conditioning_gain_mean']:+.2%} accuracy; "
                  f"{summary['by_arm'][arm]['conditioning_loss_gain_mean']:+.3f} "
                  "loss reduction")
        capped = [row["seed"] for row in arm_rows
                  if row.get("stop_reason") == "max_steps"]
        if capped:
            print(f"             hit max-steps on seeds {capped}; "
                  "extend the cap to check convergence")
    (root / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="auto",
                        help="auto: all GPUs with zero compute utilization at launch")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--min-steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-tasks", type=int, default=100)
    parser.add_argument("--test-tasks", type=int, default=1000)
    parser.add_argument("--bases", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        arms = parse_csv(args.arms, str)
        seeds = parse_csv(args.seeds, int)
        devices = idle_devices() if args.devices == "auto" else parse_csv(args.devices, int)
    except (ValueError, subprocess.CalledProcessError, FileNotFoundError) as error:
        parser.error(str(error))
    if any(arm not in ARMS for arm in arms):
        parser.error(f"arms must be drawn from {ARMS}")
    if not devices:
        parser.error("No idle GPU found; pass --devices 0,1,... to choose explicitly")
    if args.max_steps < args.eval_every or args.min_steps > args.max_steps:
        parser.error("Require max-steps >= eval-every and min-steps <= max-steps")
    root = args.out.resolve()
    jobs = [(arm, seed) for seed in seeds for arm in arms]
    pending = []
    for arm, seed in jobs:
        run = run_dir(root, arm, seed)
        config_path = run / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())["config"]
            expected = {"arm": arm, "seed": seed, "max_steps": args.max_steps,
                        "min_steps": args.min_steps, "eval_every": args.eval_every,
                        "patience": args.patience, "val_tasks": args.val_tasks,
                        "test_tasks": args.test_tasks, "bases": args.bases,
                        "rank": args.rank, "scale": args.scale,
                        "output": str(run)}
            mismatched = [key for key, value in expected.items()
                          if (config.get(key) != value and
                              not (key == "max_steps" and value > config.get(key, 0)))]
            if mismatched:
                parser.error(f"Existing {run} has different {mismatched}")
        result_path = run / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        extend_capped = (result is not None and
                         result.get("stop_reason") == "max_steps" and
                         args.max_steps > result["last_step"])
        if result is None or extend_capped:
            pending.append((arm, seed))
    print(f"GPUs: {devices}; runs: {len(jobs)}; completed: "
          f"{len(jobs)-len(pending)}; results: {root}", flush=True)
    if args.dry_run:
        print("Pending: " + ", ".join(f"{arm}/{seed}" for arm, seed in pending))
        return 0
    if not pending:
        print_summary(root, jobs)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    active: dict[int, tuple[str, int, subprocess.Popen, object]] = {}
    interrupted = False
    failed = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    tracked = {job: budget(job[0], args.max_steps, args.test_tasks)
               for job in jobs if job not in pending}
    bar = tqdm(total=sum(budget(arm, args.max_steps, args.test_tasks)
                         for arm, _seed in jobs),
               initial=sum(tracked.values()), desc="Train + test", unit="unit",
               dynamic_ncols=True, file=sys.stdout)
    try:
        while (pending or active) and not interrupted and not failed:
            for gpu in devices:
                if not pending or gpu in active:
                    continue
                arm, seed = pending.pop(0)
                run = run_dir(root, arm, seed)
                run.mkdir(parents=True, exist_ok=True)
                (run / "progress.json").unlink(missing_ok=True)
                log_path = root / "logs" / f"{arm}_seed{seed}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("a")
                command = [sys.executable, "-m", "aug_omniglot.conditional_run",
                           "--arm", arm, "--seed", str(seed),
                           "--device", f"cuda:{gpu}", "--output", str(run),
                           "--max-steps", str(args.max_steps),
                           "--min-steps", str(args.min_steps),
                           "--eval-every", str(args.eval_every),
                           "--patience", str(args.patience),
                           "--val-tasks", str(args.val_tasks),
                           "--test-tasks", str(args.test_tasks),
                           "--bases", str(args.bases), "--rank", str(args.rank),
                           "--scale", str(args.scale)]
                process = subprocess.Popen(command, stdout=log,
                                           stderr=subprocess.STDOUT,
                                           start_new_session=True)
                active[gpu] = (arm, seed, process, log)
                tracked[(arm, seed)] = 0
                tqdm.write(f"START {arm} seed={seed} GPU {gpu} "
                           f"pid={process.pid} log={log_path}", file=sys.stdout)
            status_text = []
            for gpu, (arm, seed, process, log) in list(active.items()):
                run = run_dir(root, arm, seed)
                state = progress(run)
                stage = state.get("stage", "starting")
                if stage == "test":
                    tracked[(arm, seed)] = args.max_steps + min(
                        int(state.get("test_done", 0)),
                        budget(arm, args.max_steps, args.test_tasks) - args.max_steps)
                elif stage == "done":
                    tracked[(arm, seed)] = budget(arm, args.max_steps, args.test_tasks)
                else:
                    tracked[(arm, seed)] = min(int(state.get("step", 0)), args.max_steps)
                if stage == "test":
                    detail = (f"test {int(state.get('test_done', 0))}/"
                              f"{int(state.get('test_total', 0))}")
                elif stage == "done":
                    detail = "done"
                else:
                    detail = f"step {tracked[(arm, seed)]}/{args.max_steps} {stage}"
                status_text.append(f"{arm[:4]}{seed}@{gpu}: {detail}")
                exit_code = process.poll()
                if exit_code is None:
                    continue
                log.close()
                del active[gpu]
                tqdm.write(f"END {arm} seed={seed} GPU {gpu} "
                           f"exit={exit_code}", file=sys.stdout)
                if exit_code != 0:
                    failed = True
                    tqdm.write(f"See {root / 'logs' / f'{arm}_seed{seed}.log'}",
                               file=sys.stdout)
                else:
                    tracked[(arm, seed)] = budget(arm, args.max_steps,
                                                  args.test_tasks)
            target = sum(tracked.values())
            if target > bar.n:
                bar.update(target - bar.n)
            bar.set_postfix_str(" | ".join(status_text), refresh=True)
            time.sleep(1)
    finally:
        bar.close()
        if active:
            for _arm, _seed, process, _log in active.values():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 20
            for _arm, _seed, process, log in active.values():
                try:
                    process.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                log.close()
    if interrupted or failed:
        return 130 if interrupted else 1
    print_summary(root, jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
