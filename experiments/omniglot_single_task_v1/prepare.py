"""Reproduce the existing cohort and register a new source-only extractor."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from experiments.omniglot_signature_v1.prepare import run as prepare_cohort
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)


def run(root, raw, output):
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(Path(__file__).parent.glob("*"))
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in paths:
        if path.is_file():
            shutil.copy2(path, snapshot / path.name)
    reference = Path("artifacts/omniglot_signature_2026-11-12")
    save_json(
        output / "registration.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(Path(__file__).with_name("PROTOCOL.md")),
            "runtime": str(root),
            "code_budgets": [8, 16, 32, 64],
            "target_episode_seed": 26111272013,
            "target_cohort": "Retrospective extension of 12 November cohort",
            "previous_dense_sha256": json.loads(
                (reference / "registration.json").read_text()
            )["dense_sha256"],
            "baseline_files": {
                str(p): digest(p)
                for method in ("current", "dense", "signature")
                for p in (
                    reference / "models" / method / "predictions.npz",
                    reference / "models" / method / "model.npz",
                    reference / "models" / method / "selected.json",
                )
            },
        },
    )
    prepare_cohort(root, raw)
    path = root / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol.update(
        extractor="Direct single-task ratios; exact sorted scalar forest",
        structure_selection="Independent source M after H numeric selection",
        target_sample="Same 1000 episodes as retained 12 November comparators",
        prepare_amendment_sha256=digest(Path(__file__)),
    )
    save_json(path, protocol)
    matches = json.loads((root / "source_panel_reproduction.json").read_text())
    if not all(matches.values()):
        raise RuntimeError(
            "Recreated source observations differ from reference"
        )
    for name in (
        "protocol.json",
        "panel_separation.json",
        "source_panel_reproduction.json",
    ):
        shutil.copy2(root / name, output / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.raw.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
