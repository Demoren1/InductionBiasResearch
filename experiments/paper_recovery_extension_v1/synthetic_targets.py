"""Run every frozen synthetic basis with the common fresh numerical solver."""

import concurrent.futures
import json
import subprocess
import sys

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.single_task_recovery_v1.audit import index_metrics

from .prepare import OUTPUT, ROOT
from .synthetic import METHODS, ORIGINAL, score, target_prepare


def run():
    import numpy as np

    prepared = [
        (
            ROOT / f"synthetic_{n}" / "targets" / f"case_{n:02d}_c0/structures.npz"
        ).exists()
        for n in (4, 5)
    ]
    if not any(prepared):
        target_prepare()
    elif not all(prepared):
        raise RuntimeError("Incomplete target preparation")
    audit = {}
    for number in (4, 5):
        root = ROOT / f"synthetic_{number}"
        models = np.load(root / "fits/structures.npz")
        truth = np.load(ORIGINAL / "private" / f"case_{number:02d}.npz")["index"]
        audit[str(number)] = {
            "source_models_sha256": digest(root / "fits/structures.npz")
        }
        for method in ("g34", "pair", "lutq", "yeh_map"):
            b = models[method]
            index = np.where(
                np.abs(b).max(-1) > 0,
                (np.abs(b).argmax(-1) + 1) * np.sign(b.sum(-1)),
                0,
            ).astype(np.int16)
            audit[str(number)][method] = index_metrics(index, truth)
    save_json(OUTPUT / "synthetic_structure_audit.json", audit)
    jobs = [(number, method) for number in (4, 5) for method in ("dense", *METHODS)]

    def worker(gpu):
        for number, method in jobs[gpu::8]:
            log = OUTPUT / "synthetic_jobs" / f"target_{number}_{method}.log"
            command = [
                sys.executable,
                "-u",
                "-m",
                "experiments.paper_recovery_extension_v1.regression_target",
                "--root",
                str((ROOT / f"synthetic_{number}").resolve()),
                "--case",
                f"case_{number:02d}",
                "--method",
                method,
                "--device",
                f"cuda:{gpu}",
            ]
            with log.open("w") as stream:
                subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=True
                )
            print(
                json.dumps(
                    {"synthetic_target": [number, method], "status": "complete"}
                ),
                flush=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))
    score()


if __name__ == "__main__":
    run()
