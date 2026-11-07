"""Export source-frozen structures and target observations, without target answers."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def export(root):
    protocol = json.loads((root / "protocol.json").read_text())
    for case in protocol["cases"]:
        name = case["name"]
        for condition in protocol["conditions"]:
            key = f"{name}_c{condition}"
            folder = root / "targets" / key
            if (folder / "ready.json").exists():
                continue
            requirements = {
                "current": root / "fits" / key / "joint_selected.npz",
                "lutq": root / "competitors" / key / "lutq" / "selected.npz",
                "yeh_map": root / "competitors" / key / "yeh" / "selected.npz",
                "yeh_soft": root
                / "competitors"
                / key
                / "yeh"
                / "selected_continuous.npz",
                "msr": root
                / "competitors"
                / key
                / "msr_maml"
                / "selected_continuous.npz",
            }
            if any(not path.exists() for path in requirements.values()):
                continue
            folder.mkdir(exist_ok=False)
            structures = {}
            for method, path in requirements.items():
                with np.load(path) as archive:
                    field = (
                        "joint_functional_index"
                        if method == "current"
                        else "index"
                        if method in ("lutq", "yeh_map")
                        else "basis"
                    )
                    structures[method] = archive[field].copy()
            rng = np.random.default_rng(762001 + int(name.rsplit("_", 1)[1]))
            structures["random"] = rng.permutation(
                structures["current"].ravel()
            ).reshape(structures["current"].shape)
            np.savez_compressed(folder / "structures.npz", **structures)
            with np.load(root / "private" / f"{name}_targets.npz") as archive:
                for panel in ("support", "selection"):
                    np.savez_compressed(
                        folder / f"{panel}.npz",
                        x=np.stack([archive[f"{panel}_x_{t}"] for t in range(8)]),
                        y=np.stack([archive[f"{panel}_y_{t}"] for t in range(8)]),
                    )
                np.savez_compressed(
                    folder / "query_x.npz",
                    x=np.stack([archive[f"query_x_{t}"] for t in range(8)]),
                )
            save_json(
                folder / "sizes.json",
                {k: case[k] for k in ("inputs", "hidden", "codes", "outputs")},
            )
            shutil.copy2(__file__, folder / "export_executed_code.py")
            save_json(
                folder / "ready.json",
                {
                    "frozen_source_utc": datetime.now(UTC).isoformat(),
                    "source_structures": {
                        method: {"path": str(path), "sha256": digest(path)}
                        for method, path in requirements.items()
                    },
                    "exported_sha256": {
                        p.name: digest(p) for p in folder.glob("*.npz")
                    },
                    "transfer": "Structures only, no source numeric parameters",
                    "budgets": [64, 256, 1024],
                    "numerical_selection_counts": [16, 64, 256],
                    "methods": [*structures, "dense"],
                    "target_answers_exported": False,
                    "ground_truth_structure_accessed": False,
                },
            )
            print(json.dumps({"exported": key}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    export(args.root.resolve())


if __name__ == "__main__":
    main()
