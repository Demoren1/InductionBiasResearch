"""Run experiment phases or inspect the published reference results."""

import argparse
import importlib
import json
import os
from pathlib import Path


def status(run, published):
    summary_path = run / "summary.json"
    if not summary_path.exists():
        summary_path = published / "summary.json"
    if not summary_path.exists():
        print(f"No completed results. Run directory: {run}")
        return
    summary = json.loads(summary_path.read_text())
    print(f"Results: {summary_path}")
    for name, record in summary["results"].items():
        print(
            f"{name:22s} {record['success']:4d}/{record['searches']} "
            f"{100 * record['rate']:6.2f}%"
        )


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        prog="python3 -m sine_mask_vae", description=__doc__
    )
    parser.add_argument(
        "phase",
        nargs="?",
        default="status",
        choices=(
            "status",
            "run",
            "pilot",
            "geometry",
            "population-geometry",
            "bank",
            "train",
            "validate",
            "evaluate",
            "curves",
            "report",
            "export",
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=directory / "runs/v1",
        help="Local run directory; use a new one for a new experiment",
    )
    parser.add_argument(
        "--device", default="cuda:1", help="SGD device, e.g. cuda:1 or cpu"
    )
    parser.add_argument(
        "--output", type=Path, help="Published directory (export phase)"
    )
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    os.environ["SINE_MASK_RUN"] = str(run)
    os.environ["SINE_MASK_DEVICE"] = args.device
    os.environ.setdefault("MPLCONFIGDIR", str(run / ".matplotlib"))
    published = directory / "results/v1"
    if args.phase == "status":
        status(run, published)
        return
    if args.phase == "run" and all(
        (run / name).exists()
        for name in (
            "TEST_COMPLETE.json",
            "REPORT.md",
            "training.png",
        )
    ):
        print(f"Experiment already complete: {run / 'REPORT.md'}")
        return
    if (run / "REFERENCE_RUN.json").exists() and args.phase != "export":
        parser.error(
            "The reference run is frozen. Use --run-dir sine_mask_vae/runs/reproduction."
        )
    if args.phase == "export":
        from .export import publish

        publish(run, args.output or published)
        return
    phases = [
        ("pilot", "pilot", run / "control/pilot.json"),
        ("geometry", "geometry", run / "control/geometry.json"),
        (
            "population-geometry",
            "geometry_population",
            run / "control/geometry_population.json",
        ),
        ("bank", "bank", run / "BANK_COMPLETE.json"),
        ("train", "train", run / "TRAIN_COMPLETE.json"),
        ("validate", "search", run / "selection.json"),
        ("evaluate", "search", run / "TEST_COMPLETE.json"),
        ("curves", "training_figures", run / "training.png"),
        ("report", "report", run / "REPORT.md"),
    ]
    for phase, module_name, marker in phases:
        if args.phase != "run" and args.phase != phase:
            continue
        if marker.exists():
            print(f"Skip {phase}: {marker}", flush=True)
            continue
        print(f"Start {phase}; run={run}; device={args.device}", flush=True)
        module = importlib.import_module(f".{module_name}", __package__)
        if module_name == "search":
            module.main("validate" if phase == "validate" else "test")
        else:
            module.main()


if __name__ == "__main__":
    main()
