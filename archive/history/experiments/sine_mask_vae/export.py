"""Publish compact evidence without banks or neural checkpoints."""

import csv
import hashlib
import json
import os
import platform
import shutil
from importlib.metadata import version
from pathlib import Path


def publish(run, output):
    """Copy completed evidence and its original source snapshot, without fits."""
    run, output = Path(run), Path(output)
    if not (run / "TEST_COMPLETE.json").exists():
        raise RuntimeError("Only a completed run can be published")
    if output.resolve() == run.resolve():
        raise ValueError("Published evidence and raw run must be separate")
    existing = output / "summary.json"
    if (
        existing.exists()
        and existing.read_bytes() != (run / "summary.json").read_bytes()
    ):
        raise RuntimeError(
            "Published results differ; choose another --output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "REPORT.md",
        "experiment.png",
        "combinations.png",
        "training.png",
        "summary.json",
        "plan.json",
        "selection.json",
        "models.json",
        "proposal_audit.json",
        "training_metrics.json",
        "BANK_COMPLETE.json",
        "TRAIN_COMPLETE.json",
        "TEST_COMPLETE.json",
    )
    for name in names:
        shutil.copy2(run / name, output / name)
    source_output = output / "source"
    source_output.mkdir(exist_ok=True)
    for path in (run / "source").glob("*.py"):
        shutil.copy2(path, source_output / path.name)
    for name in ("pilot.json", "geometry.json", "geometry_population.json"):
        shutil.copy2(run / "control" / name, output / name)
    shutil.copy2(run / "bank/manifest.json", output / "bank_manifest.json")
    models = json.loads((run / "models.json").read_text())
    history = output / "history"
    history.mkdir(exist_ok=True)
    for record in models:
        source = (run / record["checkpoint"]).parent / "history.json"
        shutil.copy2(
            source, history / f"{record['variant']}_{record['seed']}.json"
        )
    summary = json.loads((run / "summary.json").read_text())
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "method",
                "success",
                "searches",
                "rate",
                "mean_selected_population_mse",
            )
        )
        for name, record in summary["results"].items():
            writer.writerow(
                (
                    name,
                    record["success"],
                    record["searches"],
                    record["rate"],
                    record["mean_selected_population_mse"],
                )
            )
    with (output / "curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "candidates", "success_rate"))
        for name, points in summary["curves"].items():
            writer.writerows(
                (name, budget, value) for budget, value in points.items()
            )
    source_hashes = {
        str(path.relative_to(output)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((output / "source").glob("*"))
        if path.is_file()
    }
    selection = json.loads((run / "selection.json").read_text())
    for name in ("core", "search"):
        if (
            source_hashes[f"source/{name}.py"]
            != selection[f"{name}_source_sha256"]
        ):
            raise RuntimeError(
                f"Frozen {name} source differs from validation provenance"
            )
    manifest = {
        "reference_protocol_date": "2026-10-21",
        "scope": "Completed reference experiment; original source snapshot retained verbatim",
        "omitted": [
            "bank arrays",
            "best.pt checkpoints",
            "raw per-search arrays",
            "logs",
        ],
        "source_sha256": source_hashes,
        "packaging_python": platform.python_version(),
        "note": "The runnable package was reorganized after this experiment. Its imports/paths differ from the frozen source; measurements and original selection hashes are unchanged.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    environment = {
        "captured": "during packaging, without rerunning the experiment",
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("numpy", "torch", "matplotlib")
        },
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
    report_path = output / "REPORT.md"
    package = Path(__file__).resolve().parent
    published_label = os.path.relpath(output, package.parent)
    raw_label = os.path.relpath(run, package.parent)
    readme_link = os.path.relpath(package / "README.md", output)
    report_path.write_text(
        report_path.read_text().replace(
            "## Артефакты\n",
            f"## Артефакты\n\nОпубликованные материалы находятся в `{published_label}`. "
            f"Полный прогон с банками и весами сохранён локально в `{raw_label}` "
            f"и исключён из Git. [Запуск и структура пакета]({readme_link}).\n",
        )
    )
    (output / "SOURCE.md").write_text(
        "# Provenance\n\n"
        "The files in `source/` are the original completed-run snapshot, retained "
        "verbatim. They contain historical temporary paths and are evidence, "
        "not the current entry point. `selection.json` hashes refer to that snapshot.\n\n"
        "Run the maintained package from the repository root with "
        "`python3 -m sine_mask_vae run --run-dir sine_mask_vae/runs/reproduction`. "
        "No measurements were recomputed during relocation. Banks, weights and "
        "per-search arrays stay in the ignored local run directory.\n"
    )
    print(f"Published evidence: {output}")
