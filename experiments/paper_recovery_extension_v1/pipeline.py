"""Finish the registered source and development stages without opening queries."""

import json
import os
import subprocess
import sys
import time

from experiments.sharing_structure_research_v1.blind_contract import save_json

from .prepare import OUTPUT
from .queue import jobs


def run():
    datasets = ["mnist", "fashion", "har"]
    folder = OUTPUT / "jobs"
    expected = [name for name, _, _ in jobs("native", datasets)]
    journal = OUTPUT / "development_pipeline.json"
    while True:
        state = {}
        for name in expected:
            p = folder / f"{name}.json"
            state[name] = (
                json.loads(p.read_text())["status"] if p.exists() else "pending"
            )
        if "failed" in state.values():
            raise RuntimeError(f"Failed native job: {state}")
        save_json(journal, {"status": "waiting_native", "jobs": state})
        if all(v == "complete" for v in state.values()):
            break
        time.sleep(15)
    # The first ready-only refit queue may still own pending subprocesses.
    parents = {
        json.loads(p.read_text())["parent_pid"] for p in folder.glob("*_refit_*.json")
    }
    for parent in parents:
        while True:
            try:
                os.kill(parent, 0)
            except ProcessLookupError:
                break
            time.sleep(15)
    for phase in ("refit", "tune"):
        save_json(journal, {"status": "running", "phase": phase})
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "experiments.paper_recovery_extension_v1.queue",
                "--phase",
                phase,
                "--datasets",
                *datasets,
            ],
            check=True,
        )
    save_json(journal, {"status": "development_ready", "query_answers_opened": False})


if __name__ == "__main__":
    run()
