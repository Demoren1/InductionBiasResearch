"""Prepare one new numerical source task for each retained random geometry."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.protocol.prepare import (
    outputs_for,
)
from experiments.paper_structure_benchmark_v1.protocol.prepare_general import (
    general_index,
    general_numbers,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "public",
        "private",
        "dense",
        "source",
        "fits",
        "logs",
        "receipts",
        "metrics",
        "ratio",
        "diagnostics",
        "targets",
        "code_snapshot",
    ):
        (root / name).mkdir()
    project = Path(__file__).resolve().parents[2]
    paths = sorted(
        (project / "experiments/sharing_structure_research_v1").glob(
            "blind_*.py"
        )
    )
    save_json(
        root / "registered_sources.json",
        {
            "code_sha256": {
                str(path.relative_to(project)): digest(path) for path in paths
            },
        },
    )
    registration = {
        "created_utc": datetime.now(UTC).isoformat(),
        "cases": ["case_04", "case_05"],
        "geometry": (
            "Retained general/random geometries; new source numbers/examples"
        ),
        "sources_per_case": 1,
        "source_seed": 2611141729,
        "data_seed": 2611142341,
        "source_noise": 0.0,
        "folds": {"fixed_train": 4096, "selection": 512, "meta": 1024},
        "source_fitting": (
            "Unchanged 12-start Adam/2500, 4-start GN/32, repair/2"
        ),
        "g34": (
            "Unchanged initialize_bank, propose, "
            "fit(limit=32, steps=48), select"
        ),
        "truth_policy": (
            "Only preparation, explicit oracle controls, and post-freeze "
            "audit open private files"
        ),
        "generalization_scope": (
            "Two historical geometries, one new numerical source each; "
            "exploratory"
        ),
        "software_tests": False,
    }
    save_json(root / "registration.json", registration)
    for number in (4, 5):
        case = f"case_{number:02d}"
        geometry_rng = np.random.default_rng(27111064019 + 100003 * number)
        index = general_index(geometry_rng, "random")
        rng = np.random.default_rng(2611141729 + 100003 * number)
        outputs = 1 + number % 2
        code, bias, head, offset = general_numbers(
            rng, "random", number % 2, 1, outputs
        )
        np.savez_compressed(
            root / "private" / f"{case}.npz",
            index=index,
            codes=code,
            bias=bias,
            readout=head,
            offset=offset,
        )
        save_json(
            root / "public" / f"{case}.json",
            {
                "inputs": 16,
                "hidden": 16,
                "codes": 4,
                "outputs": outputs,
            },
        )
        folder = root / "public" / f"{case}_c0"
        folder.mkdir()
        data_rng = np.random.default_rng(2611142341 + 100003 * number)
        for name, count in registration["folds"].items():
            x = data_rng.normal(size=(count, 16))
            if name in ("fixed_train", "selection"):
                x *= np.resize([0.5, 1, 2, 4], count)[:, None]
            y = outputs_for(x, index, code[0], bias[0], head[0], offset[0])
            np.savez_compressed(folder / f"{name}.npz", x_0=x, y_0=y)
    print(root, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve())


if __name__ == "__main__":
    main()
