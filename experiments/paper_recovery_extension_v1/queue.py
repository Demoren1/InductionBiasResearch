"""One experiment subprocess per GPU, with immutable completion receipts."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time

from experiments.sharing_structure_research_v1.blind_contract import save_json

from .prepare import GRID, OUTPUT, ROOT


def jobs(phase, datasets):
    result = []
    for dataset in datasets:
        root, output = (ROOT / dataset).resolve(), (OUTPUT / dataset).resolve()
        if phase == "target":
            frozen = json.loads((output / "development_frozen.json").read_text())
            for method in frozen["models"]:
                result.append(
                    (
                        f"{dataset}_target_{method}",
                        "experiments.paper_recovery_extension_v1.workers",
                        [
                            "--root",
                            str(root),
                            "--output",
                            str(output),
                            "--phase",
                            "target",
                            "--method",
                            method,
                        ],
                    )
                )
        elif phase == "native":
            for method in ("lutq", "msr", "yeh"):
                for codes in GRID:
                    result.append(
                        (
                            f"{dataset}_{method}_{codes}",
                            "experiments.paper_structure_benchmark_v1.methods.real_competitors",
                            [
                                "--root",
                                str(root),
                                "--method",
                                method,
                                "--codes",
                                str(codes),
                            ],
                        )
                    )
            result.append(
                (
                    f"{dataset}_g34_proposals",
                    "experiments.paper_structure_benchmark_v1.methods.real_project",
                    ["--root", str(root)],
                )
            )
        elif phase == "refit":
            for method in ("g34", "pair", "lutq", "yeh_soft", "yeh_map", "msr"):
                for codes in GRID:
                    if (root / "fits" / f"{method}_k{codes}/selected.npz").exists():
                        continue
                    proposal = (
                        "current_proposals"
                        if method == "g34"
                        else "ratio_proposals"
                        if method == "pair"
                        else f"{'yeh' if method.startswith('yeh') else method}_k{codes}"
                    )
                    metrics = root / "source" / proposal / "metrics.json"
                    if (
                        not metrics.exists()
                        or json.loads(metrics.read_text())["status"] != "complete"
                    ):
                        continue
                    result.append(
                        (
                            f"{dataset}_refit_{method}_{codes}",
                            "experiments.paper_recovery_extension_v1.workers",
                            [
                                "--root",
                                str(root),
                                "--output",
                                str(output),
                                "--phase",
                                "refit",
                                "--method",
                                method,
                                "--codes",
                                str(codes),
                            ],
                        )
                    )
        elif phase == "tune":
            for method in (
                "dense",
                "g34",
                "pair",
                "lutq",
                "yeh_soft",
                "yeh_map",
                "msr",
            ):
                result.append(
                    (
                        f"{dataset}_{phase}_{method}",
                        "experiments.paper_recovery_extension_v1.workers",
                        [
                            "--root",
                            str(root),
                            "--output",
                            str(output),
                            "--phase",
                            phase,
                            "--method",
                            method,
                        ],
                    )
                )
    return result


def run(phase, datasets):
    folder = OUTPUT / "jobs"
    folder.mkdir(exist_ok=True)
    pending = []
    for name, module, arguments in jobs(phase, datasets):
        receipt = folder / f"{name}.json"
        if receipt.exists() and json.loads(receipt.read_text())["status"] == "complete":
            continue
        pending.append((name, module, arguments))
    # Distribute the expensive high-budget jobs across devices rather than stacking them.
    buckets = [pending[gpu::8] for gpu in range(8)]

    def worker(gpu, items):
        for name, module, arguments in items:
            command = [
                sys.executable,
                "-u",
                "-m",
                module,
                *arguments,
                "--device",
                f"cuda:{gpu}",
            ]
            state = {
                "name": name,
                "status": "running",
                "gpu": gpu,
                "command": command,
                "started": time.time(),
                "parent_pid": os.getpid(),
            }
            receipt = folder / f"{name}.json"
            save_json(receipt, state)
            with (folder / f"{name}.log").open("w") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT
                )
                state["pid"] = process.pid
                save_json(receipt, state)
                code = process.wait()
            state.update(
                status="complete" if code == 0 else "failed",
                returncode=code,
                finished=time.time(),
            )
            save_json(receipt, state)
            print(
                json.dumps({"job": name, "status": state["status"], "gpu": gpu}),
                flush=True,
            )
            if code:
                raise RuntimeError(f"{name} exited {code}; inspect {name}.log")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, gpu, items) for gpu, items in enumerate(buckets)]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("native", "refit", "tune", "target"), required=True
    )
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion"])
    args = parser.parse_args()
    run(args.phase, args.datasets)
