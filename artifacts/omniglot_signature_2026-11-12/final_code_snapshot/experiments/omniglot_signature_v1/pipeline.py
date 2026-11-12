"""Finish registered extraction, development choices, and sealed target fits."""

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

METHODS = ("dense", "current", "signature", "signature_random")


def launch(module, args, log):
    command = [sys.executable, "-u", "-m", module, *map(str, args)]
    with Path(log).open("w") as stream:
        result = subprocess.run(
            command, check=False, stdout=stream, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"{module}: exit {result.returncode}; see {log}")


def wait_for(paths):
    while not all(path.exists() for path in paths):
        time.sleep(5)


def export_targets(root, output):
    folder = root / "targets"
    folder.mkdir()
    protocol = json.loads((root / "protocol.json").read_text())
    arrays = dict(np.load(root / "private" / "target.npz"))
    np.savez_compressed(
        folder / "support.npz", x=arrays["support_x"], y=arrays["support_y"]
    )
    np.savez_compressed(folder / "query_x.npz", x=arrays["query_x"])
    save_json(
        folder / "sizes.json",
        {key: protocol[key] for key in ("inputs", "hidden", "outputs", "loss")},
    )
    manifest = {}
    for method in METHODS:
        origin = root / "validation_fit" / method
        destination = folder / method
        destination.mkdir()
        for name in ("model.npz", "selected.json"):
            shutil.copy2(origin / name, destination / name)
        manifest[method] = {
            name: digest(destination / name) for name in ("model.npz", "selected.json")
        }
    save_json(
        output / "development_choices_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "models": manifest,
            "support_sha256": digest(folder / "support.npz"),
            "query_inputs_sha256": digest(folder / "query_x.npz"),
            "query_answers_exported": False,
        },
    )


def run(root, output):
    wait_for([output / "partitions_frozen.json"])
    launch(
        "experiments.omniglot_signature_v1.rank",
        ["--root", root, "--output", output],
        "/tmp/ibr-omniglot-signature-rank.log",
    )
    wait_for(
        [root / "fits" / f"current_k{k}" / "selected.json" for k in (8, 16, 32, 64)]
    )
    save_json(
        root / "source_choices_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "signature": digest(output / "geometric_choices_frozen.json"),
            "current": {
                str(k): digest(root / "fits" / f"current_k{k}" / "selected.npz")
                for k in (8, 16, 32, 64)
            },
            "target_answers_used": False,
        },
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for method, gpu in (("signature", 0), ("current", 1)):
            futures.append(
                pool.submit(
                    launch,
                    "experiments.omniglot_signature_v1.validate",
                    [
                        "--root",
                        root,
                        "--output",
                        output,
                        "--method",
                        method,
                        "--device",
                        f"cuda:{gpu}",
                    ],
                    f"/tmp/ibr-omniglot-signature-dev-{method}.log",
                )
            )
        for future in futures:
            future.result()
    launch(
        "experiments.omniglot_signature_v1.validate",
        [
            "--root",
            root,
            "--output",
            output,
            "--method",
            "signature_random",
            "--device",
            "cuda:2",
        ],
        "/tmp/ibr-omniglot-signature-dev-random.log",
    )
    wait_for([root / "validation_fit/dense/selected.json"])
    export_targets(root, output)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for gpu, method in enumerate(METHODS):
            futures.append(
                pool.submit(
                    launch,
                    "experiments.paper_structure_benchmark_v1.fitting.real_target",
                    ["--root", root, "--method", method, "--device", f"cuda:{gpu}"],
                    f"/tmp/ibr-omniglot-signature-target-{method}.log",
                )
            )
        for future in futures:
            future.result()
    save_json(
        output / "predictions_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "methods": {
                method: digest(root / "targets" / method / "frozen_predictions.npz")
                for method in METHODS
            },
            "no_query_scoring_before_this_freeze": True,
        },
    )
    print(json.dumps({"all_predictions_frozen": True}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
