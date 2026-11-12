"""Regenerate source observations separately from candidate fitting."""

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np

from archive.experiments.g34_theory_v1.general import general_index, general_numbers
from experiments.paper_structure_benchmark_v1.protocol.prepare import (
    outputs_for,
    prepare,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def prepare_inputs(bundle, output):
    output.mkdir(parents=True, exist_ok=False)
    prepare(output / "standard", "confirmation", 3, 16, seed=27111054013)
    root = output / "general"
    (root / "public").mkdir(parents=True)
    folds = {"fixed_train": 4096, "selection": 512, "meta": 1024, "test": 2048}
    for number in range(6):
        case = f"case_{number:02d}"
        variant = ("local", "quartet_mix", "random")[number // 2]
        replicate, outputs = number % 2, 1 + number % 2
        rng = np.random.default_rng(27111064019 + 100003 * number)
        index = general_index(rng, variant)
        code, bias, head, offset = general_numbers(rng, variant, replicate, 16, outputs)
        panels = {name: {} for name in folds if name != "test"}
        for task in range(16):
            for name, count in folds.items():
                x = rng.normal(size=(count, 16))
                if name in ("fixed_train", "selection"):
                    x *= np.resize([0.5, 1, 2, 4], count)[:, None]
                clean = outputs_for(
                    x, index, code[task], bias[task], head[task], offset[task]
                )
                noise = rng.normal(size=clean.shape)
                if name == "test":
                    continue
                for condition, sigma in enumerate((0.0, 0.2)):
                    panels[name][f"x_{condition}_{task}"] = x
                    panels[name][f"y_{condition}_{task}"] = clean + sigma * noise
        for condition in (0, 1):
            folder = root / "public" / f"{case}_c{condition}"
            folder.mkdir()
            for name, panel in panels.items():
                np.savez_compressed(
                    folder / f"{name}.npz",
                    **{
                        f"{key}_{task}": panel[f"{key}_{condition}_{task}"]
                        for task in range(16)
                        for key in ("x", "y")
                    },
                )
    sizes = json.loads((bundle / "case_sizes.json").read_text())
    for case_key, value in sizes.items():
        cohort, case = case_key.split("/")
        save_json(output / cohort / "public" / f"{case}.json", value)
    retained = {}
    with tarfile.open(bundle / "numerical_state.tar.gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            permitted = (
                len(parts) == 4
                and parts[0] in ("standard", "general")
                and (
                    (parts[1] == "source" and parts[3] == "learned.npz")
                    or (parts[1] == "fits" and parts[3] == "fitted.npz")
                )
            ) or (
                len(parts) == 6
                and parts[0] in ("standard", "general")
                and parts[1:4] == ("methods", "full", "fits")
                and parts[-1] == "joint_fitted.npz"
            )
            if not member.isfile() or not permitted:
                continue
            destination = output / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())
            retained[member.name] = digest(destination)
    save_json(
        output / "inputs.json",
        {"archive_inputs": retained, "source_data_regenerated": True},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=Path("artifacts/anchor_start_2026-11-10")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_inputs(args.bundle, args.output)


if __name__ == "__main__":
    main()
