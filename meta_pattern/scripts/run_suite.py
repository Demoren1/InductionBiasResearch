"""Run four full-U methods with two seeds on eight independent GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PRESETS = {
    "smoke": dict(outer_steps=3, inner_steps=3, support_size=32, query_size=64,
                  batch_size=32, validate_every=3, val_tasks_per_length=1,
                  eval_steps=[3, 20], eval_repeats=1, eval_tasks=1, test_size=128),
    "pilot": dict(outer_steps=100, inner_steps=20, support_size=256, query_size=256,
                  batch_size=128, validate_every=25, val_tasks_per_length=2,
                  eval_steps=[20, 100, 500], eval_repeats=2, eval_tasks=2, test_size=1024),
    "full": dict(outer_steps=1000, inner_steps=20, support_size=256, query_size=256,
                 batch_size=128, validate_every=50, val_tasks_per_length=2,
                 eval_steps=[20, 100, 500], eval_repeats=3, eval_tasks=0, test_size=2048),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preset", choices=PRESETS, default="pilot")
    p.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--train-lengths", nargs="+", type=int, default=[3, 4, 5, 6, 7, 8])
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        p.error("GPU IDs must be distinct")
    jobs = [(method, seed) for method in ("generator", "table", "random", "ideal") for seed in args.seeds]
    if len(set(jobs)) != len(jobs):
        p.error("Seeds must be distinct")
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to write a suite into a nonempty directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "suite.json"
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite suite {args.out}")
    settings = PRESETS[args.preset]
    manifest.write_text(json.dumps({"preset": args.preset, "settings": settings, "gpus": args.gpus,
                                    "jobs": jobs, "train_lengths": args.train_lengths}, indent=2) + "\n")
    running, completed = {}, {}

    def launch(gpu, job, stage):
        method, seed = job
        name = f"{method}_seed{seed}"
        folder = args.out / name
        if stage == "train":
            command = [sys.executable, "-m", "meta_pattern.train", "--out", str(folder),
                       "--method", method, "--seed", str(seed), "--train-lengths", *map(str, args.train_lengths)]
            for key, value in settings.items():
                if not key.startswith("eval_") and key != "test_size":
                    command.extend(["--" + key.replace("_", "-"), str(value)])
        else:
            command = [sys.executable, "-m", "meta_pattern.evaluate", "--checkpoint", str(folder / "best.pt"),
                       "--out", str(folder / "evaluation.json"), "--steps", *map(str, settings["eval_steps"]),
                       "--repeats", str(settings["eval_repeats"]), "--tasks-per-length", str(settings["eval_tasks"]),
                       "--test-size", str(settings["test_size"])]
        stream = (args.out / f"{name}_{stage}.log").open("w")
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                       "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        running[gpu] = (process, stream, job, stage)
        print(f"START gpu={gpu} {name} {stage} pid={process.pid}", flush=True)

    try:
        for gpu in args.gpus:
            if jobs:
                launch(gpu, jobs.pop(0), "train")
        last = time.monotonic()
        while running:
            for gpu, (process, stream, job, stage) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del running[gpu]
                print(f"EXIT gpu={gpu} {job} {stage} code={code}", flush=True)
                if code == 0 and stage == "train":
                    launch(gpu, job, "evaluate")
                else:
                    completed[f"{job[0]}_seed{job[1]}"] = code
                    if jobs:
                        launch(gpu, jobs.pop(0), "train")
            if time.monotonic() - last > 30:
                print(f"STATUS running={len(running)} completed={len(completed)}", flush=True)
                last = time.monotonic()
            if running:
                time.sleep(1)
    except BaseException:
        for process, _, _, _ in running.values():
            process.terminate()
        for process, stream, _, _ in running.values():
            process.wait()
            stream.close()
        raise
    (args.out / "suite_done.json").write_text(json.dumps(completed, indent=2) + "\n")
    if any(completed.values()):
        raise SystemExit(f"Failed workers; inspect suite logs: {completed}")
    subprocess.run([sys.executable, "-m", "meta_pattern.report", "--suite", str(args.out)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
