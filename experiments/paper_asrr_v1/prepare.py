"""Restore immutable benchmark inputs in a separate aSRR run directory."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.paper_recovery_extension_v1 import har
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

ROOT = Path("artifacts/local/paper_asrr_2026-11-17")
OUTPUT = Path("artifacts/paper_asrr_2026-11-17")
PREVIOUS = Path("artifacts/paper_recovery_extension_2026-11-14")
BASE = Path("artifacts/local/paper_recovery_comparison_2026-11-14")
ORIGINAL = Path("artifacts/single_task_recovery_2026-11-14")
GRID = (8, 16, 32, 64, 128, 512)


def register():
    ROOT.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "jobs").mkdir()
    shutil.copytree(
        Path(__file__).parent,
        OUTPUT / "code_snapshot",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    save_json(
        OUTPUT / "registration.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(Path(__file__).with_name("PROTOCOL.md")),
            "budgets": GRID,
            "original_results": str(PREVIOUS),
            "target_cohort": "Exact previously evaluated episode draws; added-method follow-up",
            "query_responses_used_for_asrr_selection": False,
        },
    )
    for dataset in ("mnist", "fashion"):
        root, output = ROOT / dataset, OUTPUT / dataset
        root.mkdir()
        output.mkdir()
        for name in ("public", "validation", "source/dense"):
            shutil.copytree(BASE / dataset / name, root / name)
        for name in ("fits", "validation_fit", "targets", "private"):
            (root / name).mkdir()
        shutil.copy2(
            PREVIOUS / dataset / "registration.json", root / "protocol.json"
        )
        shutil.copytree(
            PREVIOUS / dataset / "development/dense",
            root / "validation_fit/dense",
        )
        expected = json.loads(
            (PREVIOUS / dataset / "pair/registration.json").read_text()
        )["previous_dense_sha256"]
        actual = digest(root / "source/dense/models.npz")
        if actual != expected:
            raise RuntimeError(f"Different source weights for {dataset}")
        save_json(
            output / "source_reuse.json", {"sha256": actual, "identical": True}
        )
    for number in (4, 5):
        root, output = (
            ROOT / f"synthetic_{number}",
            OUTPUT / f"synthetic_{number}",
        )
        root.mkdir()
        output.mkdir()
        for name in ("public", "source/dense", "fits", "targets", "private"):
            (root / name).mkdir(parents=True, exist_ok=True)
        case = f"case_{number:02d}"
        sizes = json.loads((ORIGINAL / "public" / f"{case}.json").read_text())
        save_json(root / "protocol.json", {**sizes, "loss": "mse"})
        for source, destination in (
            ("fixed_train", "support"),
            ("selection", "numerical"),
            ("meta", "structure"),
        ):
            with np.load(
                ORIGINAL / "public" / f"{case}_c0/{source}.npz"
            ) as data:
                np.savez_compressed(
                    root / "public" / f"{destination}.npz",
                    x=data["x_0"][None].astype(np.float32) * 255,
                    y=data["y_0"][None].astype(np.float32),
                )
        with np.load(ORIGINAL / "dense" / f"{case}_c0/fitted.npz") as data:
            np.savez_compressed(
                root / "source/dense/models.npz",
                parameters=data["parameters"].astype(np.float32),
            )


def prepare_har(archive):
    original_archive = json.loads(
        (PREVIOUS / "har/registration.json").read_text()
    )["archive_sha256"]
    (ROOT / "raw").mkdir(exist_ok=True)
    shutil.copy2(archive, ROOT / "raw/uci_har.zip")
    har.ROOT, har.OUTPUT, har.RAW = ROOT, OUTPUT, ROOT / "raw/uci_har.zip"
    har.prepare()
    expected = json.loads((PREVIOUS / "har/registration.json").read_text())
    actual = json.loads((ROOT / "har/protocol.json").read_text())
    if actual["source_sha256"] != expected["source_sha256"]:
        raise RuntimeError("HAR source observations differ from the original")
    save_json(
        OUTPUT / "har/observation_restoration.json",
        {
            "original_archive_sha256": original_archive,
            "downloaded_archive_sha256": digest(archive),
            "identical_source_observations": True,
            "container_note": "The official legacy link serves the dataset zip directly; the newer link wraps it in another zip.",
        },
    )
    shutil.copytree(
        PREVIOUS / "har/development/dense", ROOT / "har/validation_fit/dense"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("register", "har"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.phase == "register":
        register()
    else:
        prepare_har(args.archive)
