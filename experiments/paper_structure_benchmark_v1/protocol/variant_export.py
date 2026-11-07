"""Export a registered conditional variant without changing primary target models."""

import argparse
import json
import shutil
from pathlib import Path

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def run(root, method):
    origin = root / "validation_fit" / method
    output = root / "targets" / method
    output.mkdir(exist_ok=False)
    for name in ("model.npz", "selected.json"):
        shutil.copy2(origin / name, output / name)
    save_json(
        output / "export.json",
        {
            "status": "complete",
            "model_sha256": digest(output / "model.npz"),
            "selection_sha256": digest(output / "selected.json"),
            "source_and_development_only": True,
            "target_query_answers_used": False,
            "source_numeric_parameters_transferred": False,
        },
    )
    print(json.dumps({"exported_variant": method}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.root.resolve(), args.method)


if __name__ == "__main__":
    main()
