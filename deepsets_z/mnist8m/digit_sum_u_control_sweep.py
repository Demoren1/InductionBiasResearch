"""Train matched Kronecker/fixed-U controls, fine-tune, and update the plot.

The generated-U runs already present in the repository outputs are reused.
This launcher never starts work on a GPU whose utilization is nonzero when the
launcher begins. Interrupted jobs resume from their latest checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tqdm import tqdm


ARMS = ("kronecker", "random", "convolution")
ROOT = Path(__file__).resolve().parents[2]
PRETRAIN = ROOT / "deepsets_z" / "mnist8m" / "outputs" / "meta_u_first_v_moe_u_controls"
FINETUNE = ROOT / "deepsets_z" / "mnist8m" / "outputs" / "v_moe_digit_sum_finetune_raw"
FINETUNE_MODES = {
    "kronecker": (False, True),
    "random": (True,),
    "convolution": (True,),
}
TOTAL_JOBS = len(ARMS) + sum(map(len, FINETUNE_MODES.values()))


def idle_devices() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    return [int(parts[0]) for line in output.splitlines()
            if len(parts := [item.strip() for item in line.split(",")]) == 2
            and parts[1] != "[Not Supported]" and int(parts[1]) == 0]


def pretrain_stem(arm: str) -> str:
    return f"moe_k96_ortho0p2_seed42_u_{arm}"


def finetune_stem(arm: str, unfreeze: bool) -> str:
    stem = f"conv_router_v_head_seed42_u_{arm}"
    return stem + ("_unfrozen_middle" if unfreeze else "")


def job_label(stage: str, arm: str, unfreeze: bool) -> str:
    if stage == "pretrain":
        return f"pretrain_{arm}"
    mode = "only_u_frozen" if unfreeze else "u_and_middle_frozen"
    return f"finetune_{arm}_{mode}"


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def pretrain_done(arm: str) -> bool:
    result = read_json(PRETRAIN / f"{pretrain_stem(arm)}.json")
    return result is not None and "test" in result and "stopping" in result


def finetune_done(arm: str, unfreeze: bool) -> bool:
    result = read_json(FINETUNE / f"{finetune_stem(arm, unfreeze)}.json")
    return result is not None and "test" in result and "stopping" in result


def _bar(fraction: float, width: int = 16) -> str:
    filled = min(width, max(0, round(width * fraction)))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return ((f"{hours}h{minutes:02d}m") if hours else
            (f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"))


def pretrain_progress(arm: str) -> str:
    result = read_json(PRETRAIN / f"{pretrain_stem(arm)}.json")
    label = f"pretrain {arm:11s}"
    if result is None or not result.get("history"):
        return f"{label} waiting"
    row = result["history"][-1]
    step = int(row["step"])
    elapsed = float(row.get("elapsed_seconds", 0.0))
    config = result.get("config", {})
    minimum = int(config.get("min_steps", 12000))
    cap = int(config.get("steps", 100000))
    rate = step / elapsed if elapsed > 0 else 0.0
    eta_min = max(0, minimum - step) / rate if rate > 0 else None
    delta = float(config.get("early_stop_min_delta", 0.002))
    eval_every = int(config.get("eval_every", 250))
    patience = int(config.get("early_stop_patience", 16))
    material_best = float("inf")
    last_material_step = 0
    for item in result["history"]:
        value = float(item["validation_score"])
        if value < material_best - delta:
            material_best = value
            last_material_step = int(item["step"])
    plateau = max(0, (step - last_material_step) // eval_every)
    best = float(result["best_validation_score"])
    if pretrain_done(arm):
        stop = result["stopping"].get("stop_step", step)
        return (f"{label} DONE step={stop} best={best:.4f} "
                f"time={_duration(elapsed)}")
    if step < minimum:
        progress = (f"{_bar(step / minimum)} {step}/{minimum} minimum "
                    f"ETA≥{_duration(eta_min)}")
    else:
        progress = f"minimum reached; step={step}/{cap}"
    return (f"{label} {progress} best={best:.4f} "
            f"plateau={plateau}/{patience} rate={rate:.1f} step/s")


def finetune_progress(arm: str, unfreeze: bool) -> str:
    mode = "only-U" if unfreeze else "U+middle"
    label = f"finetune {arm:11s} {mode:8s}"
    result = read_json(FINETUNE / f"{finetune_stem(arm, unfreeze)}.json")
    if result is None or not result.get("history"):
        state = "queued" if pretrain_done(arm) else "waiting for pretrain"
        return f"{label} {state}"
    row = result["history"][-1]
    epoch = int(row["epoch"])
    elapsed = float(row.get("elapsed_seconds", 0.0))
    config = result.get("config", {})
    cap = int(config.get("epochs", 100))
    patience = int(config.get("patience", 12))
    best_epoch = int(result["best_epoch"])
    best = float(result["best_validation_mae"])
    rate = epoch / elapsed if elapsed > 0 else 0.0
    eta_cap = max(0, cap - epoch) / rate if rate > 0 else None
    if finetune_done(arm, unfreeze):
        return (f"{label} DONE epoch={result['stopping']['epoch']} "
                f"best={best:.4f}@{best_epoch} time={_duration(elapsed)}")
    return (f"{label} {_bar(epoch / cap)} {epoch}/{cap} best={best:.4f}@{best_epoch} "
            f"plateau={epoch - best_epoch}/{patience} ETA≤{_duration(eta_cap)}")


def progress_lines() -> list[str]:
    lines = [pretrain_progress(arm) for arm in ARMS]
    lines.extend(finetune_progress(arm, unfreeze)
                 for arm in ARMS for unfreeze in FINETUNE_MODES[arm])
    return lines


def monitor_progress(refresh: float) -> int:
    interactive = sys.stdout.isatty()
    try:
        while True:
            done = completed_jobs()
            if interactive:
                print("\033[2J\033[H", end="")
            print(f"U controls: {done}/{TOTAL_JOBS} stages complete · "
                  f"refresh {refresh:g}s")
            print("\n".join(progress_lines()), flush=True)
            if done == TOTAL_JOBS:
                return 0
            time.sleep(refresh)
    except KeyboardInterrupt:
        return 130


def pretrain_command(arm: str, gpu: int) -> list[str]:
    command = [
        sys.executable, "-m", "deepsets_z.mnist8m.meta_u_first_v_moe",
        "--experts", "96", "--router", "conv", "--u-arm", arm,
        "--ortho-weight", "0.2", "--device", f"cuda:{gpu}", "--seed", "42",
        "--out", str(PRETRAIN), "--steps", "100000", "--inner-steps", "5",
        "--lr-coeff", "0.1", "--lr-readout", "0.03", "--outer-lr", "0.0002",
        "--tasks-per-step", "2", "--support", "32", "--query", "32",
        "--train-images-per-digit", "3000", "--eval-images-per-digit", "1000",
        "--eval-every", "250", "--val-tasks", "16", "--test-tasks", "64",
        "--early-stop-patience", "16", "--early-stop-min-delta", "0.002",
        "--min-steps", "12000",
    ]
    if (PRETRAIN / f"{pretrain_stem(arm)}_latest.pt").exists():
        command.append("--resume")
    return command


def finetune_command(arm: str, unfreeze: bool, gpu: int) -> list[str]:
    command = [
        sys.executable, "-m", "deepsets_z.mnist8m.finetune_v_moe_digit_sum",
        "--router", "conv", "--u-arm", arm,
        "--trainable", "router_v_head", "--device", f"cuda:{gpu}",
        "--seed", "42", "--checkpoint",
        str(PRETRAIN / f"{pretrain_stem(arm)}_best.pt"),
        "--out", str(FINETUNE), "--epochs", "100", "--patience", "12",
        "--batch-size", "128", "--lr", "0.001", "--adam-eps", "0.001",
        "--target-center", "0", "--target-scale", "1", "--probe-sets", "1000",
    ]
    if unfreeze:
        command.append("--unfreeze-middle")
    if (FINETUNE / f"{finetune_stem(arm, unfreeze)}_latest.pt").exists():
        command.append("--resume")
    return command


def jobs_ready() -> list[tuple[str, str, bool]]:
    jobs: list[tuple[str, str, bool]] = []
    for arm in ARMS:
        if not pretrain_done(arm):
            jobs.append(("pretrain", arm, False))
            continue
        # Kronecker U gets both freezing modes for both panels. Fixed controls
        # use the strongest recipe, where every layer except U adapts.
        modes = FINETUNE_MODES[arm]
        jobs.extend(("finetune", arm, unfreeze)
                    for unfreeze in modes
                    if not finetune_done(arm, unfreeze))
    return jobs


def completed_jobs() -> int:
    count = 0
    for arm in ARMS:
        if not pretrain_done(arm):
            continue
        count += 1
        modes = FINETUNE_MODES[arm]
        count += sum(finetune_done(arm, mode) for mode in modes)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="auto",
                        help="auto or comma-separated GPU indices")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--monitor", action="store_true",
                        help="Only display live progress from result files")
    parser.add_argument("--refresh", type=float, default=2.0,
                        help="Seconds between --monitor screen updates")
    args = parser.parse_args()
    if args.refresh <= 0:
        parser.error("--refresh must be positive")
    if args.monitor:
        return monitor_progress(args.refresh)
    if args.devices == "auto":
        devices = idle_devices()
    else:
        try:
            devices = [int(value.strip()) for value in args.devices.split(",")
                       if value.strip()]
        except ValueError:
            parser.error("--devices must be auto or comma-separated integers")
    if not devices or len(devices) != len(set(devices)):
        parser.error("No distinct GPUs selected")

    PRETRAIN.mkdir(parents=True, exist_ok=True)
    FINETUNE.mkdir(parents=True, exist_ok=True)
    initially_done = completed_jobs()
    print(f"GPUs: {devices}; completed: {initially_done}/{TOTAL_JOBS}; "
          f"pretrain: {PRETRAIN}; fine-tune: {FINETUNE}", flush=True)
    if args.dry_run:
        for stage, arm, unfreeze in jobs_ready():
            detail = ("random-score task family" if stage == "pretrain" else
                      "only U frozen" if unfreeze else "U and middle frozen")
            print(f"{stage:9s} {arm:11s} {detail}")
        return 0

    active: dict[int, tuple[tuple[str, str, bool], subprocess.Popen, object]] = {}
    launched: set[tuple[str, str, bool]] = set()
    interrupted = False
    failed = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    bar = tqdm(total=TOTAL_JOBS, initial=initially_done, desc="U controls",
               unit="run", dynamic_ncols=True, file=sys.stdout)
    logs = PRETRAIN / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        while not interrupted and not failed:
            pending = [job for job in jobs_ready()
                       if job not in launched and job not in
                       [entry[0] for entry in active.values()]]
            for gpu in devices:
                if gpu in active or not pending:
                    continue
                job = pending.pop(0)
                stage, arm, unfreeze = job
                label = job_label(stage, arm, unfreeze)
                log_path = logs / f"{label}.log"
                log = log_path.open("a")
                command = (pretrain_command(arm, gpu) if stage == "pretrain"
                           else finetune_command(arm, unfreeze, gpu))
                process = subprocess.Popen(command, stdout=log,
                                           stderr=subprocess.STDOUT,
                                           start_new_session=True)
                active[gpu] = (job, process, log)
                launched.add(job)
                tqdm.write(f"START {label} GPU {gpu} pid={process.pid} "
                           f"log={log_path}", file=sys.stdout)
            if not active and not jobs_ready():
                break
            for gpu, (job, process, log) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                del active[gpu]
                stage, arm, unfreeze = job
                label = job_label(stage, arm, unfreeze)
                tqdm.write(f"END {label} GPU {gpu} exit={status}", file=sys.stdout)
                if status == 0:
                    bar.update(1)
                else:
                    failed = True
                    tqdm.write(f"See {logs / f'{label}.log'}", file=sys.stdout)
            compact = []
            for gpu, (job, _process, _log) in active.items():
                stage, arm, unfreeze = job
                result = read_json(
                    PRETRAIN / f"{pretrain_stem(arm)}.json" if stage == "pretrain"
                    else FINETUNE / f"{finetune_stem(arm, unfreeze)}.json")
                if result and result.get("history"):
                    row = result["history"][-1]
                    position = (f"s{row['step']}" if stage == "pretrain"
                                else f"e{row['epoch']}")
                    score = (result.get("best_validation_score")
                             if stage == "pretrain"
                             else result.get("best_validation_mae"))
                    compact.append(f"{arm}@{gpu}:{position},best={score:.3f}")
                else:
                    compact.append(f"{arm}@{gpu}:starting")
            bar.set_postfix_str(" | ".join(compact), refresh=True)
            time.sleep(1)
    finally:
        bar.close()
        if active:
            for _job, process, _log in active.values():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for _job, process, log in active.values():
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                log.close()
    if interrupted or failed:
        return 130 if interrupted else 1
    return subprocess.call([
        sys.executable, "-m", "deepsets_z.mnist8m.plot_digit_sum_u_controls"])


if __name__ == "__main__":
    raise SystemExit(main())
