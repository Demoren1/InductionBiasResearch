"""Export finalized real-data transfer models and support/query-input panels."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

METHODS = ("dense", "current", "lutq", "yeh_map", "yeh_soft", "msr", "random")


def export(root):
    if not all(
        (root / "validation_fit" / method / "selected.json").exists()
        for method in METHODS
    ):
        return
    folder = root / "targets"
    if (folder / "ready.json").exists():
        return
    folder.mkdir(exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    arrays = dict(np.load(root / "private" / "target.npz"))
    np.savez_compressed(
        folder / "support.npz", x=arrays["support_x"], y=arrays["support_y"]
    )
    np.savez_compressed(folder / "query_x.npz", x=arrays["query_x"])
    save_json(
        folder / "sizes.json",
        {key: protocol[key] for key in ("inputs", "hidden", "outputs", "loss")},
    )
    manifest = {}
    for method in METHODS:
        origin = root / "validation_fit" / method
        model = folder / method
        model.mkdir()
        for name in ("model.npz", "selected.json"):
            shutil.copy2(origin / name, model / name)
        manifest[method] = {
            name: digest(model / name) for name in ("model.npz", "selected.json")
        }
    save_json(
        folder / "ready.json",
        {
            "models": manifest,
            "support_sha256": digest(folder / "support.npz"),
            "query_inputs_sha256": digest(folder / "query_x.npz"),
            "transfer": "Frozen indexator/basis and a global numerical learning rule; no source or development numerical values",
            "query_answers_exported": False,
        },
    )
    shutil.copy2(__file__, folder / "export_executed_code.py")
    print(json.dumps({"exported_real_targets": str(root)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    export(args.root.resolve())


if __name__ == "__main__":
    main()
