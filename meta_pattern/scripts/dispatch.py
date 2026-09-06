"""Execute a manifest of independent jobs on the user-authorized GPUs only."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_GPUS = {0, 4, 5, 6, 7}


def terminate(signum, frame):
    raise SystemExit(128 + signum)


def dispatch(manifest, out, gpus):
    if not gpus or not set(gpus) <= ALLOWED_GPUS or len(set(gpus)) != len(gpus):
        raise ValueError("This experiment is authorized only on distinct GPUs 0,4,5,6,7")
    jobs = json.loads(Path(manifest).read_text())
    if len({j["name"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate job names")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "dispatch.json").exists():
        raise FileExistsError(out / "dispatch.json")
    (out / "dispatch.json").write_text(json.dumps({"gpus": gpus, "jobs": jobs}, indent=2) + "\n")
    pending, running, complete = list(jobs), {}, {}

    def launch(gpu):
        job = pending.pop(0)
        stream = (out / f"{job['name']}.log").open("w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen([sys.executable, *job["args"]], cwd=ROOT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        running[gpu] = job, process, stream, time.monotonic()
        print(f"START gpu={gpu} job={job['name']} pid={process.pid}", flush=True)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        for gpu in gpus:
            if pending:
                launch(gpu)
        last = time.monotonic()
        while running:
            for gpu, (job, process, stream, started) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                complete[job["name"]] = {"exit_code": code, "seconds": time.monotonic() - started, "gpu": gpu}
                del running[gpu]
                print(f"EXIT gpu={gpu} job={job['name']} code={code}", flush=True)
                if pending:
                    launch(gpu)
            if time.monotonic() - last > 30:
                print(f"STATUS running={len(running)} queued={len(pending)} complete={len(complete)}", flush=True)
                last = time.monotonic()
            if running:
                time.sleep(1)
    finally:
        for _, process, _, _ in running.values():
            process.terminate()
        for _, process, stream, _ in running.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()
        (out / "status.json").write_text(json.dumps(complete, indent=2) + "\n")
    if any(job["exit_code"] for job in complete.values()):
        raise SystemExit("Some jobs failed; inspect logs and status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 4, 5, 6, 7])
    dispatch(**vars(parser.parse_args()))
