"""Launch the complete MLP bank -> top-10% CVAE -> interpolation evaluation."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from .common import patterns_by_split, source_hashes, write_json
from .config import Config

ROOT = Path(__file__).resolve().parents[2]


def terminate_run(signum, frame):
    """Turn timeout/SIGTERM into an exception so worker cleanup always runs."""
    raise SystemExit(128 + signum)


def run_stage(out, name, commands, gpus):
    pending = list(enumerate(commands))
    running, statuses = {}, {}

    def start(gpu):
        index, arguments = pending.pop(0)
        log = (out / "logs" / f"{name}_{index}.log").open("w")
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                       "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen([sys.executable, *arguments], cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        running[gpu] = index, process, log
        print(f"START stage={name} job={index} gpu={gpu} pid={process.pid}", flush=True)

    try:
        for gpu in gpus:
            if pending:
                start(gpu)
        last = time.monotonic()
        while running:
            for gpu, (index, process, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[gpu]
                statuses[index] = code
                print(f"EXIT stage={name} job={index} code={code}", flush=True)
                if pending:
                    start(gpu)
            if time.monotonic() - last > 30:
                print(f"STATUS stage={name} running={len(running)} pending={len(pending)} complete={len(statuses)}", flush=True)
                last = time.monotonic()
            if running:
                time.sleep(1)
    except BaseException:
        for _, process, _ in running.values():
            process.terminate()
        for _, process, log in running.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
        raise
    write_json(out / f"{name}_done.json", statuses)
    if any(statuses.values()):
        raise RuntimeError(f"Stage {name} failed; inspect {out / 'logs'}: {statuses}")


def main():
    signal.signal(signal.SIGTERM, terminate_run)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--bank-mlps", type=int, default=2000)
    parser.add_argument("--bank-steps", type=int, default=2000)
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU IDs must be nonempty and distinct")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment: {out}")
    c = (Config(bank_mlps=20, bank_steps=3, bank_val_size=64, support_pool_size=64,
                cvae_epochs=2, cvae_batch=8, cvae_hidden=32, latent_dim=4,
                eval_masks=2, eval_repeats=1, eval_steps=3, eval_test_size=64,
                task_limit=1, eval_task_limit=2) if args.smoke else
         Config(bank_mlps=args.bank_mlps, bank_steps=args.bank_steps))
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir()
    write_json(out / "protocol.json", {"config": c.to_dict(), "patterns": patterns_by_split(c),
                                       "source_sha256": source_hashes(), "smoke": args.smoke,
                                       "condition": "one continuous scalar: 2*(length-3)/5-1",
                                       "mask_budget": "32*target_length for all final masks, including wrong conditions"})
    started = time.monotonic()
    common = ["--out", str(out)]
    run_stage(out, "bank", [["-m", "pattern.length_interp.bank", *common, "--shard", str(i),
                             "--shards", str(len(args.gpus))] for i in range(len(args.gpus))], args.gpus)
    run_stage(out, "cvae", [["-m", "pattern.length_interp.train_cvae", *common, "--seed", str(seed)]
                            for seed in c.cvae_seeds], args.gpus)
    run_stage(out, "evaluate", [["-m", "pattern.length_interp.evaluate", *common, "--seed", str(seed),
                                 "--length", str(k), "--shard", str(shard), "--shards", "2"]
                                for seed in c.cvae_seeds for k in c.heldout_lengths for shard in range(2)], args.gpus)
    subprocess.run([sys.executable, "-m", "pattern.length_interp.report", *common], cwd=ROOT, check=True)
    write_json(out / "done.json", {"seconds": time.monotonic() - started, "stages": ["bank", "cvae", "evaluate", "report"]})


if __name__ == "__main__":
    main()
