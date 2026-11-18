"""Register the extension without rewriting any completed observations."""

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

BASE = Path("artifacts/local/paper_recovery_comparison_2026-11-14")
ROOT = Path("artifacts/local/paper_recovery_extension_2026-11-14")
OUTPUT = Path("artifacts/paper_recovery_extension_2026-11-14")
GRID = (8, 16, 32, 64, 128, 512)
METHODS = ("dense", "g34", "pair", "lutq", "yeh_soft", "yeh_map", "msr")


def run():
    ROOT.mkdir(exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    for dataset in ("mnist", "fashion"):
        old, root, output = BASE / dataset, ROOT / dataset, OUTPUT / dataset
        root.mkdir()
        output.mkdir()
        for name in ("public", "validation"):
            shutil.copytree(old / name, root / name, copy_function=os.link)
        for name in ("source", "fits", "validation_fit", "private", "targets"):
            (root / name).mkdir()
        shutil.copytree(
            old / "source/dense", root / "source/dense", copy_function=os.link
        )
        for method in ("g34", "pair"):
            for codes in (8, 16, 32, 64):
                name = f"{method}_k{codes}"
                shutil.copytree(
                    old / "fits" / name, root / "fits" / name, copy_function=os.link
                )
        protocol = json.loads((old / "protocol.json").read_text())
        protocol.update(
            independent_code_grid=[128, 512],
            extension_code_grid=list(GRID),
            extension_registered_utc=datetime.now(UTC).isoformat(),
            original_protocol_sha256=digest(old / "protocol.json"),
            extension_protocol_sha256=digest(Path(__file__).with_name("PROTOCOL.md")),
        )
        save_json(root / "protocol.json", protocol)
        save_json(output / "registration.json", protocol)
        (output / "pair/code_snapshot").mkdir(parents=True)
        save_json(
            output / "pair/registration.json",
            {
                "previous_dense_sha256": digest(root / "source/dense/models.npz"),
                "meaning": "Compatibility key: this same shared source Dense archive.",
            },
        )
    shutil.copytree(
        Path(__file__).parent,
        OUTPUT / "code_snapshot",
        ignore=shutil.ignore_patterns("__pycache__"),
    )


if __name__ == "__main__":
    run()
