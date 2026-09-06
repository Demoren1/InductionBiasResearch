"""Run the eight independent normalized-search shards on GPU 0 through 7."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/normalized_task_agreement/20260906")
    args = parser.parse_args()
    logs = args.out / "logs"; logs.mkdir(parents=True, exist_ok=True)
    jobs = [(profile, gap, shard) for profile, gaps in (("interp", (5, 8)), ("extrap", (3, 4)))
            for gap in gaps for shard in (0, 1)]
    running, completed = {}, {}
    try:
        for gpu, (profile, gap, shard) in enumerate(jobs):
            label = f"gpu{gpu}_{profile}_gap{gap}_shard{shard}"
            stream = (logs / f"{label}.log").open("a")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
            command = [sys.executable, str(ROOT / "evaluation/normalized_task_agreement.py"), "--out", str(args.out),
                       "--profile", profile, "--gap", str(gap), "--shard", str(shard)]
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            running[label] = process, stream; print(f"START {label} pid={process.pid}", flush=True)
        last = time.monotonic()
        while running:
            for label, (process, stream) in list(running.items()):
                result = process.poll()
                if result is not None:
                    completed[label] = result; stream.close(); del running[label]
                    print(f"EXIT {label} code={result}", flush=True)
            if time.monotonic() - last >= 30:
                print(f"STATUS running={len(running)} complete={len(completed)}", flush=True); last = time.monotonic()
            if running: time.sleep(2)
    except BaseException:
        for process, _ in running.values(): process.terminate()
        for process, stream in running.values(): process.wait(); stream.close()
        raise
    if any(completed.values()): raise SystemExit(f"Some workers failed: {completed}")
    print("All eight shards completed.", flush=True)


if __name__ == "__main__":
    main()
