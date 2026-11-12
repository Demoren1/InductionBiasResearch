"""Recreate source/development panels and seal a new target episode sample."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.protocol.prepare_omniglot import (
    episodes,
    prepare,
)
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import split_panels
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def run(root, raw):
    prepare(root, raw, 128, 128, 1000)
    # The historical test episode answers are public. Replace that sample before
    # starting any learner, keeping the original source and development panels.
    target_seed = 26111272013
    data, details = episodes(raw, "test", 1000, target_seed)
    np.savez_compressed(root / "private" / "target.npz", **data)
    provenance_path = root / "private" / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["target"] = details
    save_json(provenance_path, provenance)
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update(
        target_episode_seed=target_seed,
        target_sample="New episodes; historical test class split reused",
        extractor="uncertainty-weighted SVD, first geometric signature candidate",
        structure_selection="No source functional reranking of signatures",
        prepare_amendment_sha256=digest(Path(__file__)),
    )
    save_json(protocol_path, protocol)
    split_panels(root)
    historical = Path("artifacts/benchmark_2026-11-07/omniglot_pilot")
    expected = json.loads((historical / "panel_separation.json").read_text())
    actual = json.loads((root / "panel_separation.json").read_text())
    matches = {
        name: actual["panels"][name] == value
        for name, value in expected["panels"].items()
    }
    save_json(root / "source_panel_reproduction.json", matches)
    print(json.dumps({"source_panels_match_historical": matches}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.raw.resolve())


if __name__ == "__main__":
    main()
