"""Run a one-seed image-encoder sweep on GPUs idle at launch.

Only the image encoder changes. The U generator, adaptation and evaluation
remain those of meta_u_first_image_code. Re-running skips completed variants
and resumes partial ones from their latest validation checkpoint.
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

from .meta_u_first_image_code import ENCODERS, ImageCodeU


DEFAULT_ENCODERS = ("mlp32", "mlp128", "mlp256", "mlp512",
                    "mlp256x2", "conv32", "conv64")
ARM = "generated_ortho"


def stem(encoder: str, seed: int) -> str:
    return (f"{ARM}_seed{seed}" if encoder == "mlp64"
            else f"{ARM}_{encoder}_seed{seed}")


def idle_devices() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    return [int(parts[0]) for line in output.splitlines()
            if (parts := [piece.strip() for piece in line.split(",")])
            and len(parts) == 2 and int(parts[1]) == 0]


def run_state(out: Path, encoder: str, seed: int, steps: int) -> str:
    path = out / f"{stem(encoder, seed)}.json"
    latest = out / f"{stem(encoder, seed)}_latest.pt"
    if not path.exists():
        return "resume" if latest.exists() else "new"
    data = json.loads(path.read_text())
    config = data["config"]
    if (config["arm"] != ARM or config.get("encoder", "mlp64") != encoder
            or config["seed"] != seed or config["steps"] != steps):
        raise ValueError(f"Existing run has different settings: {path}")
    if "test" in data and "test_no_image_code" in data:
        return "done"
    if not latest.exists():
        raise ValueError(f"Partial run lacks checkpoint: {latest}")
    return "resume"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", default=",".join(DEFAULT_ENCODERS),
                        help="Comma-separated names; mlp64 baseline is already saved")
    parser.add_argument("--devices", default="auto",
                        help="auto selects GPUs with zero current utilization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--out", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_encoder_sweep"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    encoders = tuple(name.strip() for name in args.encoders.split(",") if name.strip())
    if (not encoders or len(set(encoders)) != len(encoders)
            or any(name not in ENCODERS for name in encoders)):
        parser.error("--encoders must contain distinct known encoder names")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.dry_run:
        for name in encoders:
            model = ImageCodeU(ARM, args.seed, name)
            count = sum(p.numel() for p in model.image_encoder.parameters())
            print(f"{name:10s} encoder_parameters={count:>8,d} "
                  f"model_parameters={sum(p.numel() for p in model.parameters()):>8,d}")
        print("Reference: precomputed mlp64, dense and direct-U seed-42 runs")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.devices == "auto":
        devices = idle_devices()
    else:
        try:
            devices = [int(part.strip()) for part in args.devices.split(",")]
        except ValueError:
            parser.error("--devices must be auto or comma-separated GPU indices")
    if not devices or len(set(devices)) != len(devices):
        parser.error("No distinct idle GPUs selected")
    states = {name: run_state(args.out, name, args.seed, args.steps)
              for name in encoders}
    pending = [name for name in encoders if states[name] != "done"]
    print(f"GPUs: {devices}; completed: {len(encoders) - len(pending)}; "
          f"pending: {len(pending)}; output: {args.out}", flush=True)
    active: dict[int, tuple[str, subprocess.Popen, object]] = {}
    interrupted = False
    failed = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while pending or active:
            for gpu in devices:
                if interrupted or failed or not pending or gpu in active:
                    continue
                name = pending.pop(0)
                log_path = args.out / "logs" / f"{stem(name, args.seed)}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("a")
                command = [
                    sys.executable, "-m",
                    "deepsets_z.mnist8m.meta_u_first_image_code",
                    "--arm", ARM, "--encoder", name,
                    "--device", f"cuda:{gpu}", "--seed", str(args.seed),
                    "--steps", str(args.steps), "--data-dir", str(args.data_dir),
                    "--out", str(args.out)]
                if states[name] == "resume":
                    command.append("--resume")
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                active[gpu] = (name, process, log)
                print(f"START {name} GPU {gpu} pid={process.pid} "
                      f"log={log_path}", flush=True)
            for gpu, (name, process, log) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                del active[gpu]
                print(f"END {name} GPU {gpu} exit={status}", flush=True)
                if status != 0:
                    failed = True
            if interrupted or failed:
                break
            time.sleep(0.5)
    finally:
        if active:
            for _name, process, _log in active.values():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 20
            for _name, process, log in active.values():
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
    if args.seed == 42 and args.steps == 5000 and set(encoders) == set(DEFAULT_ENCODERS):
        command = [sys.executable, "-m",
                   "deepsets_z.mnist8m.summarize_encoder_sweep",
                   "--results", str(args.out)]
        return subprocess.call(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
