"""Classify binary 0110 after transferring only the source-frozen structure."""

import argparse
import json
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from .adaptation import fit_adam, signed_basis, values
from .target_fit import TargetBoundary

METHODS = ("current", "lutq", "yeh_map", "yeh_soft", "msr", "random", "dense")


def export(root):
    protocol = json.loads((root / "protocol.json").read_text())
    for case in protocol["cases"]:
        if case["inputs"] != 11:
            continue
        for condition in protocol["conditions"]:
            key = f"{case['name']}_c{condition}"
            folder = root / "binary_targets" / key
            if (folder / "ready.json").exists():
                continue
            folder.mkdir(parents=True, exist_ok=False)
            origin = root / "targets" / key / "structures.npz"
            shutil.copy2(origin, folder / "structures.npz")
            with np.load(root / "private" / f"{case['name']}_targets.npz") as archive:
                x, y, order = (
                    archive["binary_x"],
                    archive["binary_y"],
                    archive["binary_partition"],
                )
                for name, indices in (
                    ("support", order[:512]),
                    ("selection", order[512:640]),
                ):
                    np.savez_compressed(
                        folder / f"{name}.npz",
                        x=x[indices][None],
                        y=y[indices][None, :, None],
                    )
                np.savez_compressed(folder / "query_x.npz", x=x[order[640:]][None])
            save_json(
                folder / "sizes.json",
                {k: case[k] for k in ("inputs", "hidden", "codes", "outputs")},
            )
            save_json(
                folder / "ready.json",
                {
                    "source_structure_sha256": digest(origin),
                    "support": 512,
                    "numerical_selection": 128,
                    "query": 1408,
                    "loss": "binary_cross_entropy_with_logits",
                    "query_answers_exported": False,
                    "numeric_parameters_transferred": False,
                },
            )


def fit(root, key):
    torch.set_num_threads(2)
    folder = root / "binary_targets" / key
    boundary = TargetBoundary(root, folder)
    boundary.install()
    size = json.loads((folder / "sizes.json").read_text())
    h, k, o = size["hidden"], size["codes"], size["outputs"]
    structures = np.load(folder / "structures.npz")
    data = [
        dict(np.load(folder / name))
        for name in ("support.npz", "selection.npz", "query_x.npz")
    ]
    x, y, vx, vy, qx = [
        torch.tensor(a, dtype=torch.float32)
        for a in (data[0]["x"], data[0]["y"], data[1]["x"], data[1]["y"], data[2]["x"])
    ]
    output = folder / "fitted"
    output.mkdir(exist_ok=False)
    save_json(
        output / "registered.json",
        {
            "code_sha256": digest(Path(__file__)),
            "adapter_sha256": digest(Path(__file__).with_name("adaptation.py")),
            "steps": 2000,
            "restarts": 18,
            "selection": "Fresh numeric parameters; H0 binary CE checkpoint; query inputs only",
            "cpu_threads": 2,
        },
    )
    shutil.copy2(__file__, output / "executed_code.py")
    shutil.copy2(Path(__file__).with_name("adaptation.py"), output / "adaptation.py")
    result, curves, measurements = {}, {}, {}
    status = "failed"
    try:
        for method in METHODS:
            basis = (
                None
                if method == "dense"
                else torch.tensor(structures[method], dtype=torch.float32)
                if method in ("yeh_soft", "msr")
                else signed_basis(
                    structures[method], k, torch.device("cpu"), torch.float32
                )
            )
            counter = FlopCounterMode(display=False)
            tick = time.perf_counter()
            with counter:
                parameter, trace = fit_adam(
                    x, y, vx, vy, basis, h, o, loss="binary_cross_entropy", seed=773003
                )
                with torch.no_grad():
                    result[method] = values(qx, basis, parameter[:1], h, o)[0].numpy()
                    result[f"{method}_parameters"] = parameter[0].numpy()
            curves.update(
                {
                    f"{method}_{name}": value
                    for name, value in trace.items()
                    if isinstance(value, np.ndarray)
                }
            )
            measurements[method] = {
                "wall_seconds": time.perf_counter() - tick,
                "torch_operator_flops": counter.get_total_flops(),
            }
            print(
                json.dumps(
                    {
                        "binary_case": key,
                        "method": method,
                        "H0_binary_CE": float(trace["numerical_selection_loss"][0, 0]),
                        **measurements[method],
                    }
                ),
                flush=True,
            )
        np.savez_compressed(output / "frozen_predictions.npz", **result)
        np.savez_compressed(output / "curves.npz", **curves)
        status = "complete"
    finally:
        save_json(
            output / "metrics.json",
            {
                "status": status,
                "methods": measurements,
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "predictions_sha256": digest(output / "frozen_predictions.npz")
                if status == "complete"
                else None,
            },
        )
        save_json(
            output / "access_receipt.json",
            {
                "status": status,
                "violations": boundary.violations,
                "opened_workspace_files": sorted(boundary.opened),
            },
        )


def launch(root):
    export(root)
    folders = sorted((root / "binary_targets").glob("*/ready.json"))

    def execute(path):
        folder = path.parent
        if (folder / "fitted" / "metrics.json").exists():
            return
        log = folder / "fit.log"
        with log.open("w") as stream:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.paper_structure_benchmark_v1.fitting.binary_control",
                    "--root",
                    str(root),
                    "--phase",
                    "fit",
                    "--key",
                    folder.name,
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode:
            raise RuntimeError(f"Binary control failed: {log}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(execute, folders))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("launch", "fit"), default="launch")
    parser.add_argument("--key")
    args = parser.parse_args()
    if args.phase == "launch":
        launch(args.root.resolve())
    else:
        fit(args.root.resolve(), args.key)


if __name__ == "__main__":
    main()
