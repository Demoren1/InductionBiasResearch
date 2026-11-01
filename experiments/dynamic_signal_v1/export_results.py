"""Export compact evidence, retaining metrics and indices but not large raw data."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="experiments/dynamic_signal_v1/protocol.json"
    )
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["run"])
    output = Path("experiments/dynamic_signal_v1/results") / root.name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hashes = {}
    for source in sorted(root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(root)
        # Generated observations are reproducible from the published seed.
        if (
            relative.parts[0] in ["source", "target", "calibration"]
            and source.name != "teacher_audit.npz"
        ):
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".pt":
            checkpoint_hashes[str(relative)] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
        elif source.suffix == ".npz":
            with np.load(source) as arrays:
                kept = {
                    key: arrays[key]
                    for key in arrays.files
                    if not key.startswith("parameter_")
                }
                # Preserve training weights used by both published extractors.
                if relative.parts[0] in [
                    "banks",
                    "banks_large_tuned",
                ] and source.name in ["round_4096.npz", "round_256.npz"]:
                    kept.update(
                        {
                            key: arrays[key]
                            for key in arrays.files
                            if key.startswith("parameter_")
                        }
                    )
                if (
                    "parameter_0" in arrays.files
                    and "target" in str(relative)
                    and "2048" in str(relative)
                    and "noise_1" in str(relative)
                ):
                    chosen = []
                    for task in np.unique(arrays["task_ids"]):
                        ids = np.flatnonzero(arrays["task_ids"] == task)
                        chosen.append(ids[arrays["query_loss"][ids].argmin()])
                    kept.update(
                        {
                            "selected_" + key: arrays[key][chosen]
                            for key in arrays.files
                            if key.startswith("parameter_")
                        }
                    )
                if kept:
                    np.savez_compressed(destination, **kept)
        elif source.suffix in [".json", ".gz"]:
            shutil.copyfile(source, destination)
    (output / "checkpoint_hashes.json").write_text(
        json.dumps(checkpoint_hashes, indent=2) + "\n"
    )
    hashes = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*")
        if path.is_file() and path.name != "export_manifest.json"
    }
    (output / "export_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
