"""Enforce the observation boundary in actual learner processes."""

import json
import os
import sys
import time
from pathlib import Path


class ObservationBoundary:
    def __init__(self, root, case, condition):
        self.root = Path(root).resolve()
        self.project = Path(__file__).resolve().parents[2]
        self.role = "train"
        self.phase = "setup"
        self.opened = set()
        self.violations = []
        self.started = time.monotonic()
        self.case = case
        self.observation_folder = f"{case}_c{condition}"

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
        forbidden |= path.is_relative_to(self.project / "pattern")
        forbidden |= path.name.startswith(
            (
                "blind_fixed_protocol.",
                "blind_study.",
                "pattern_prefix.",
                "shared_structure_data.",
            )
        )
        if path.is_relative_to(self.root / "public"):
            local = path.relative_to(self.root / "public")
            forbidden |= local.parts[0] not in (
                f"{self.case}.json",
                self.observation_folder,
            )
            forbidden |= path.name == "test.npz"
            forbidden |= self.role == "train" and path.name == "meta.npz"
        for owner in ("source", "dense", "fits"):
            if path.is_relative_to(self.root / owner):
                local = path.relative_to(self.root / owner)
                forbidden |= bool(local.parts) and not local.parts[0].startswith(
                    self.observation_folder
                )
        if forbidden:
            record = {"phase": self.phase, "role": self.role, "path": str(path)}
            self.violations.append(record)
            raise PermissionError(f"Forbidden teacher or held-out access: {path}")
        self.opened.add((self.phase, self.role, str(path)))

    def save(self, path, status):
        record = {
            "status": status,
            "seconds": time.monotonic() - self.started,
            "violations": self.violations,
            "opened_workspace_files": [
                {"phase": phase, "role": role, "path": filename}
                for phase, role, filename in sorted(self.opened)
            ],
        }
        Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
