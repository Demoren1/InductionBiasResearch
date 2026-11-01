"""Compact exported HPO evidence; full target trajectories remain unchanged."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    for version in ["v1", "v2"]:
        root = Path("experiments/dynamic_signal_v1/results") / version
        for path in sorted(root.rglob("*.npz")):
            relative = str(path.relative_to(root))
            if "calibrate" not in relative and "calibration_fit" not in relative:
                continue
            with np.load(path) as loaded:
                if "indices" not in loaded.files:
                    continue
                arrays = {key: loaded[key] for key in loaded.files if key != "indices"}
                indices = loaded["indices"]
            if indices.min() < 0 or indices.max() > 255:
                raise ValueError("Indexator does not fit in one byte")
            dictionary, unique, inverse = {}, [], []
            for row in indices:
                encoded = row.astype(np.uint8)
                key = encoded.tobytes()
                if key not in dictionary:
                    dictionary[key] = len(unique)
                    unique.append(encoded)
                inverse.append(dictionary[key])
            arrays.update(
                indices_unique=np.stack(unique),
                indices_inverse=np.array(inverse, np.int32),
            )
            np.savez_compressed(path, **arrays)
        for path in sorted(root.rglob("*_history.json.gz")):
            relative = str(path.relative_to(root))
            if "calibrate" not in relative and "calibration_fit" not in relative:
                continue
            model_path = path.parent / (
                path.name.removesuffix("_history.json.gz") + ".npz"
            )
            with np.load(model_path) as arrays:
                ids = arrays["config_ids"]
            configs = np.unique(ids)
            with gzip.open(path, "rt") as stream:
                history = json.load(stream)
            summary = []
            for point in history:
                means = {"step": point["step"]}
                for split in ["train", "query"]:
                    means[split] = {}
                    counts = np.bincount(ids)
                    for name, values in point[split].items():
                        sums = np.bincount(ids, weights=np.asarray(values))
                        means[split][name] = (sums[configs] / counts[configs]).tolist()
                summary.append(means)
            destination = path.with_name(
                path.name.replace("_history.json.gz", "_history_by_config.json.gz")
            )
            with gzip.open(destination, "wt") as stream:
                json.dump(
                    {
                        "aggregation": "Mean over tasks and restarts per configuration; full target histories are retained separately.",
                        "config_ids": configs.tolist(),
                        "history": summary,
                    },
                    stream,
                )
            path.unlink()
        manifest = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file() and path.name != "export_manifest.json"
        }
        (root / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
