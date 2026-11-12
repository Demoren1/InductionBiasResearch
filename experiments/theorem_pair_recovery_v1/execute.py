"""Schedule independent source fits on GPUs 0-5 and freeze selections."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def execute(work, output, cache):
    cases = sorted(output.glob("*/case_*/candidates.json"))
    queue = [json.loads(path.read_text()) for path in cases]
    active, finished = {}, []
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    files = sorted(Path(__file__).parent.glob("*.py"))
    save_json(
        output / "functional_registration.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "code_sha256": {str(path): digest(path) for path in files},
            "proposal_freeze_sha256": digest(output / "proposal_freeze.json"),
            "gpus": list(range(6)),
            "numeric_limit": 32,
            "gn_steps": 48,
            "starts": 4,
            "reuses_identical_cached_baseline_fits": True,
            "selection": "independent source meta; no target observations",
            "additional_fit_cache": None if cache is None else str(cache),
        },
    )
    while queue or active:
        for gpu in range(6):
            if gpu in active or not queue:
                continue
            item = queue.pop(0)
            name = f"{item['cohort']}_{item['case']}_c{item['condition']}"
            stream = (logs / f"{name}.log").open("w")
            command = [
                sys.executable,
                "-m",
                "experiments.theorem_pair_recovery_v1.fit",
                "--work",
                str(work),
                "--output",
                str(output),
                "--cohort",
                item["cohort"],
                "--case",
                item["case"],
                "--condition",
                str(item["condition"]),
            ]
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "2",
            }
            if cache is not None:
                command.extend(["--cache", str(cache)])
            process = subprocess.Popen(
                command, env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
            active[gpu] = (process, stream, name)
            print(json.dumps({"started": name, "gpu": gpu}), flush=True)
        for gpu, (process, stream, name) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            stream.close()
            del active[gpu]
            finished.append({"name": name, "exit_code": status})
            print(
                json.dumps(
                    {
                        "finished": name,
                        "exit_code": status,
                        "done": len(finished),
                        "total": len(cases),
                    }
                ),
                flush=True,
            )
        save_json(
            output / "execution_state.json",
            {
                "done": finished,
                "live": {gpu: name for gpu, (_, _, name) in active.items()},
                "pending": len(queue),
            },
        )
        if active:
            time.sleep(2)
    failed = [item for item in finished if item["exit_code"]]
    if failed:
        raise RuntimeError(f"Source evaluation failed: {failed}")
    selections = [
        path.parent / name
        for path in cases
        for name in ("functional.json", "selection.npz", "functional_states.npz")
    ]
    save_json(
        output / "selection_freeze.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "teacher_access": False,
            "files": {
                str(path.relative_to(output)): digest(path) for path in selections
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    execute(args.work, args.output, args.cache)


if __name__ == "__main__":
    main()
