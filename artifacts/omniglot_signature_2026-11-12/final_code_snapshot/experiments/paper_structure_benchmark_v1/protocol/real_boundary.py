"""Keep real-data fitting, source selection and final queries in separate files."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


class StudyBoundary:
    def __init__(self, root, role="train"):
        self.root = Path(root).resolve()
        self.project = Path(__file__).resolve().parents[3]
        self.role = role
        self.opened = set()
        self.violations = []
        self.started = time.monotonic()

    def install(self):
        sys.addaudithook(self._audit)

    def _audit(self, event, args):
        if event != "open" or not args or isinstance(args[0], int):
            return
        try:
            path = Path(os.fsdecode(args[0])).resolve()
        except (TypeError, ValueError, OSError):
            return
        if not path.is_relative_to(self.project):
            return
        forbidden = "private" in path.parts
        forbidden |= "results" in path.parts and not path.is_relative_to(self.root)
        forbidden |= self.role != "validation" and path.is_relative_to(
            self.root / "validation"
        )
        forbidden |= (
            path.is_relative_to(self.root / "public") and path.name == "source.npz"
        )
        forbidden |= (
            self.role == "train" and path == self.root / "public" / "structure.npz"
        )
        if forbidden:
            self.violations.append({"role": self.role, "path": str(path)})
            raise PermissionError(f"Held-out data access is forbidden: {path}")
        self.opened.add((self.role, str(path)))

    def save(self, path, status):
        save_json(
            path,
            {
                "status": status,
                "seconds": time.monotonic() - self.started,
                "violations": self.violations,
                "opened_workspace_files": [
                    {"role": role, "path": name} for role, name in sorted(self.opened)
                ],
            },
        )


def split_panels(root):
    original = root / "public" / "source.npz"
    archive = dict(np.load(original))
    manifest = {"original_sha256": digest(original), "panels": {}}
    for name in ("support", "numerical", "structure"):
        path = root / "public" / f"{name}.npz"
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, x=archive[f"{name}_x"], y=archive[f"{name}_y"])
        manifest["panels"][name] = digest(path)
    save_json(root / "panel_separation.json", manifest)
    print(json.dumps({"separated_source_panels": str(root)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    split_panels(args.root.resolve())


if __name__ == "__main__":
    main()
