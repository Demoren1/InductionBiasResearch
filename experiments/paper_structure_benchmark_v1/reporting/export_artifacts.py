"""Publish a compact, auditable copy of an already frozen benchmark."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .evaluate import REAL_DATA, expected_predictions


def export(base, destination):
    """Keep exact predictions and audit answers, omitting bulk training states."""
    if not (base / "report/scoring.json").is_file():
        raise RuntimeError("Export requires completed scoring after prediction freeze")
    original = base / "report/frozen_results.json"
    frozen = json.loads(original.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    records = {}

    def retain(source, *, arrays=None, relative=None):
        relative = relative or source.relative_to(base)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if arrays is None:
            shutil.copy2(source, target)
        else:
            np.savez_compressed(target, **arrays)
        records[str(relative)] = {
            "source_sha256": digest(source),
            "sha256": digest(target),
            "bytes": target.stat().st_size,
            "arrays": None if arrays is None else list(arrays),
        }

    for source in sorted(base.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(base)
        if "private" in relative.parts or "real_sources" in relative.parts:
            continue
        if relative.parts[0] == "report":
            continue
        if source.suffix in (".json", ".csv", ".md") and (
            source.stat().st_size <= 256 * 1024
            or "receipt" in source.name
            or source.name.endswith("_fleet.json")
        ):
            retain(source)
    retain(original)
    retain(
        base / "report/scoring.json", relative=Path("provenance/original_scoring.json")
    )
    retain(base / "pre_query_decisions.json")
    retain(base / "report/post_cleanup_integrity.json")

    for source in expected_predictions(base):
        relative = str(source.relative_to(base))
        expected = frozen["files_sha256"][relative]
        if digest(source) != expected:
            raise RuntimeError(f"Changed frozen prediction: {relative}")
        with np.load(source) as archive:
            arrays = {
                key: archive[key][:1] if key.startswith("parameters") else archive[key]
                for key in archive.files
            }
        retain(source, arrays=arrays)

    synthetic = base / "synthetic_confirmation"
    protocol = json.loads((synthetic / "protocol.json").read_text())
    for case in protocol["cases"]:
        name = case["name"]
        source = synthetic / "private" / f"{name}.npz"
        with np.load(source) as archive:
            retain(source, arrays={"index": archive["index"]})
        source = synthetic / "private" / f"{name}_targets.npz"
        with np.load(source) as archive:
            retain(
                source,
                arrays={
                    key: archive[key]
                    for key in archive.files
                    if key.startswith("query_y_")
                    or key in ("binary_y", "binary_partition")
                },
            )
        for condition in protocol["conditions"]:
            key = f"{name}_c{condition}"
            retain(synthetic / "targets" / key / "structures.npz")
            source = synthetic / "fits" / key / "joint_fitted.npz"
            with np.load(source) as archive:
                retain(source, arrays={"indices": archive["indices"]})
            source = synthetic / "dense" / key / "fitted.npz"
            with np.load(source) as archive:
                retain(source, arrays={"parameters": archive["parameters"][:1]})

    for case in ("case_00_c0", "case_03_c0"):
        for method in ("current", "dense", "lutq", "yeh_map", "yeh_soft", "msr"):
            source = synthetic / "targets" / case / method / "curves.npz"
            with np.load(source) as archive:
                retain(
                    source,
                    arrays={
                        key: archive[key]
                        for key in archive.files
                        if key.endswith("_1024")
                    },
                )

    for dataset in REAL_DATA:
        root = base / dataset
        source = root / "private/target.npz"
        with np.load(source) as archive:
            retain(source, arrays={"query_y": archive["query_y"]})
        for source in sorted((root / "targets").glob("*/model.npz")):
            retain(source)
        source = root / "source/dense/models.npz"
        with np.load(source) as archive:
            retain(source, arrays={"parameters": archive["parameters"][:1]})

    save_json(
        destination / "published_manifest.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "format_version": 1,
            "original_frozen_manifest_sha256": digest(original),
            "export_code_sha256": digest(Path(__file__)),
            "files_sha256": {key: value["sha256"] for key, value in records.items()},
            "records": records,
            "scope": (
                "Exact frozen query predictions and audit answers; selected indexators/"
                "continuous bases; bank indices; representative numerical states and "
                "display curves; protocols, selection, costs and access receipts. "
                "Full source observations, raw images, unselected numerical states and "
                "generating codebooks/biases/readouts are omitted. Private-named answer "
                "files are audit inputs only, exported after original scoring."
            ),
            "parameter_states": "Only the first task state is retained; predictions retain every task",
            "export_performs_model_training": False,
            "audit_answers_used_for_model_selection": False,
        },
    )
    print(
        json.dumps(
            {
                "published_files": len(records),
                "MiB": sum(item["bytes"] for item in records.values()) / 2**20,
                "destination": str(destination),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    export(args.base.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
