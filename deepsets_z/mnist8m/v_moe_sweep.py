"""Sweep the number and orthogonality of image-routed v experts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .encoder_sweep import idle_devices
from .meta_u_first_v_moe import VExpertMoE


SPECS = ((1, 0.0), (10, 0.0), (10, 0.05), (10, 0.2),
         (32, 0.0), (32, 0.05), (32, 0.2))


def stem(experts: int, weight: float, seed: int) -> str:
    return f"moe_k{experts}_ortho{f'{weight:g}'.replace('.', 'p')}_seed{seed}"


def run_state(out: Path, experts: int, weight: float, seed: int,
              steps: int) -> str:
    name = stem(experts, weight, seed)
    path = out / f"{name}.json"
    latest = out / f"{name}_latest.pt"
    if not path.exists():
        return "resume" if latest.exists() else "new"
    data = json.loads(path.read_text())
    config = data["config"]
    if (config["experts"] != experts or config["ortho_weight"] != weight
            or config["seed"] != seed or config["steps"] != steps):
        raise ValueError(f"Existing result has different settings: {path}")
    if "test" in data and "test_uniform_route" in data and "routing" in data:
        return "done"
    if not latest.exists():
        raise ValueError(f"Partial run lacks checkpoint: {latest}")
    return "resume"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="auto",
                        help="auto uses only zero-utilization GPUs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_v_moe"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.dry_run:
        for experts, weight in SPECS:
            model = VExpertMoE(experts, args.seed)
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{stem(experts, weight, args.seed):27s} "
                  f"parameters={count:>8,d} adapted={experts + 31}")
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
    states = {(k, w): run_state(args.out, k, w, args.seed, args.steps)
              for k, w in SPECS}
    pending = [spec for spec in SPECS if states[spec] != "done"]
    print(f"GPUs: {devices}; completed: {len(SPECS)-len(pending)}; "
          f"pending: {len(pending)}; output: {args.out}", flush=True)
    active: dict[int, tuple[tuple[int, float], subprocess.Popen, object]] = {}
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
                spec = pending.pop(0)
                k, weight = spec
                name = stem(k, weight, args.seed)
                log_path = args.out / "logs" / f"{name}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("a")
                command = [
                    sys.executable, "-m",
                    "deepsets_z.mnist8m.meta_u_first_v_moe",
                    "--experts", str(k), "--ortho-weight", str(weight),
                    "--device", f"cuda:{gpu}", "--seed", str(args.seed),
                    "--steps", str(args.steps), "--out", str(args.out),
                    "--data-dir", str(args.data_dir)]
                if states[spec] == "resume":
                    command.append("--resume")
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                active[gpu] = (spec, process, log)
                print(f"START {name} GPU {gpu} pid={process.pid} "
                      f"log={log_path}", flush=True)
            for gpu, (spec, process, log) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                del active[gpu]
                print(f"END {stem(*spec, args.seed)} GPU {gpu} exit={status}",
                      flush=True)
                if status != 0:
                    failed = True
            if interrupted or failed:
                break
            time.sleep(0.5)
    finally:
        if active:
            for _spec, process, _log in active.values():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 20
            for _spec, process, log in active.values():
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
    if args.seed == 42 and args.steps == 5000:
        status = subprocess.call([
            sys.executable, "-m", "deepsets_z.mnist8m.diagnose_v_moe_routing",
            "--results", str(args.out), "--data-dir", str(args.data_dir),
            "--device", f"cuda:{devices[0]}"])
        if status != 0:
            return status
        return subprocess.call([
            sys.executable, "-m", "deepsets_z.mnist8m.summarize_v_moe_sweep",
            "--results", str(args.out)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
