"""Audit actual selection records and preserve the completed study."""

import json
import shutil
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .data import SETTINGS
from .replicate import OLD, OUTPUT, ROOT


def run():
    if (OUTPUT / "completion.json").exists():
        raise RuntimeError("Study already finalized")
    results = json.loads((OUTPUT / "tuned_results.json").read_text())
    choices = json.loads((OUTPUT / "choices_frozen.json").read_text())[
        "policies"
    ]
    checks, compute = [], {}
    for phase in ("d1", "refine", "d2", "target"):
        jobs = json.loads((OUTPUT / f"{phase}_jobs.json").read_text())
        seconds = 0
        for number in range(len(jobs)):
            receipt = json.loads(
                (OUTPUT / f"{phase}_jobs/{number:04d}.json").read_text()
            )
            if receipt["exit_code"]:
                raise RuntimeError(f"Incomplete {phase} job {number}")
            seconds += receipt["seconds"]
        compute[phase] = {"jobs": len(jobs), "gpu_hours": seconds / 3600}

    for setting, policies in choices.items():
        dataset, shots, validation, outputs = SETTINGS[setting]
        root = ROOT / "study" / setting
        registry = json.loads(
            (ROOT / "banks" / dataset / "models.json").read_text()
        )
        models = {row["name"]: row for row in registry}
        options = []
        destination = OUTPUT / "development" / setting
        destination.mkdir(parents=True)
        for phase in ("d1", "d2"):
            shutil.copytree(root / f"fits/{phase}", destination / phase)
            with np.load(root / f"panels/{phase}.npz") as panel:
                np.savez_compressed(
                    destination / f"{phase}_answers.npz", y=panel["query_y"]
                )
            shutil.copy2(
                root / f"panels/{phase}_provenance.json",
                destination / f"{phase}_provenance.json",
            )
        for path in (root / "fits/d2").glob("*.json"):
            fit = json.loads(path.read_text())
            model = models[fit["job"]["name"]]
            families = {
                "asrr" if family.startswith("asrr") else family
                for family in [
                    model["family"],
                    *model.get("family_aliases", []),
                ]
            }
            for key, metric in fit["metrics"].items():
                for family in families:
                    options.append(
                        {
                            "name": model["name"],
                            "key": key,
                            "family": family,
                            "codes": model["codes"],
                            "metric": metric,
                        }
                    )
        for policy, choice in policies.items():
            family, criterion = policy.split("_")
            eligible = [r for r in options if r["family"] == family]
            selected = next(
                r
                for r in eligible
                if (r["name"], r["key"]) == (choice["name"], choice["key"])
            )

            def ordering(row):
                if criterion == "accuracy":
                    return (
                        -row["metric"]["accuracy"],
                        row["metric"]["ce"],
                        row["codes"],
                    )
                return row["metric"]["ce"], row["codes"]

            if ordering(selected) != min(map(ordering, eligible)):
                raise RuntimeError(
                    "Frozen policy is not a development optimum"
                )
            checks.append({"setting": setting, "policy": policy, **choice})

        for cohort in ("test_a", "test_b"):
            receipts = root / f"fits/{cohort}"
            target = OUTPUT / "target_receipts" / setting / cohort
            target.mkdir(parents=True)
            for path in receipts.glob("*.json"):
                receipt = json.loads(path.read_text())
                if receipt["access_violations"]:
                    raise RuntimeError("Target data boundary violation")
                allowed = {
                    (root / f"panels/{cohort}.npz").resolve(),
                    path.resolve(),
                    path.with_suffix(".npz").resolve(),
                }
                name = receipt["job"]["name"]
                if name != "dense":
                    allowed.add(
                        (
                            ROOT / "banks" / dataset / f"models/{name}.npz"
                        ).resolve()
                    )
                for opened in receipt["opened_workspace_files"]:
                    opened = Path(opened)
                    if "artifacts" in opened.parts and opened not in allowed:
                        raise RuntimeError(
                            f"Unexpected target input: {opened}"
                        )
                shutil.copy2(path, target / path.name)
            with np.load(root / f"panels/{cohort}.npz") as panel:
                if "query_y" in panel.files:
                    raise RuntimeError(
                        "Query answers present in fitting panel"
                    )
                if panel["train_y"].shape[1:] != (shots * outputs, outputs):
                    raise RuntimeError("Unexpected support allocation")
                if validation and panel["valid_y"].shape[1:] != (
                    validation * outputs,
                    outputs,
                ):
                    raise RuntimeError("Unexpected selection allocation")
            provenance = OUTPUT / "episode_provenance" / setting
            provenance.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                root / f"private/{cohort}_provenance.json",
                provenance / f"{cohort}.json",
            )

    for dataset in {value[0] for value in SETTINGS.values()}:
        bank = ROOT / "banks" / dataset
        for path in (bank / "models").glob("*"):
            if path.suffix == ".npz":
                with np.load(path) as model:
                    if model.files != ["index"]:
                        raise RuntimeError(
                            "Bank contains numerical source values"
                        )
            shutil.copy2(
                path, OUTPUT / "banks" / dataset / "models" / path.name
            )
        profiles = (
            OLD / dataset / "source_profiles.npz"
            if dataset in ("mnist", "omniglot")
            else ROOT.parent
            / "paper_asrr_2026-11-17"
            / dataset
            / "source/asrr/profiles.npz"
        )
        shutil.copy2(
            profiles, OUTPUT / "banks" / dataset / "source_profiles.npz"
        )
    for dataset in ("mnist", "omniglot"):
        shutil.copytree(
            ROOT / "replication" / dataset / "models",
            OUTPUT / "replication" / dataset / "models",
        )

    snapshot = OUTPUT / "source_snapshot"
    shutil.copytree(
        Path("experiments"),
        snapshot / "experiments",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    save_json(
        OUTPUT / "scientific_audit.json",
        {
            "development_optima": checks,
            "stages": compute,
            "both_cohorts_used_unchanged_policies": True,
            "source_transfer": "Indexator only; fresh numerical parameters",
            "no_query_answers_in_fitting_panels": True,
            "target_access_violations": [],
            "scored_rows": len(results["rows"]),
            "software_tests_run": False,
        },
    )
    shutil.copy2(
        Path(__file__).with_name("PROTOCOL.md"), OUTPUT / "PROTOCOL.md"
    )
    (OUTPUT / "README.md").write_text(
        "# Broad aSRR tuning, 17 November 2026\n\n"
        "`REPORT.md` gives the results and numerical costs. "
        "`replication/` repeats the original frozen successful procedures; "
        "the wider search is a separate study. `choices_frozen.json` "
        "precedes both fresh cohorts, and `target_registration.json` fixes "
        "their observations and answers. `tuned_scoring.npz` retains all "
        "selected predictions and answers; `tuned_results.json` includes "
        "each cohort, the fixed pool and paired differences.\n\n"
        "`development/` retains all screened numerical predictions and "
        "selection metrics. `banks/` contains indexators, source profiles "
        "and refinement provenance; `target_receipts/` records target file "
        "access. `scientific_audit.json` checks actual development choices "
        "and sample allocations. `manifest.json` covers the preserved files.\n\n"
        "The source/class/subject environments were studied previously. "
        "The primary policy maximizes development accuracy; separately "
        "frozen CE choices are secondary. These are the best selected "
        "procedures in the declared search, not a global optimum or an "
        "equal-total-compute comparison. No software tests were run.\n\n"
        "Regenerate tables and report without GPU fitting:\n\n"
        "```bash\nMPLCONFIGDIR=/tmp/asrr-tuning-mpl "
        "python -m experiments.asrr_tuning_v1.report\nmake -C paper\n```\n\n"
        "Do not rerun preparation, freeze, scoring or finalization over "
        "the completed study. See `source_snapshot/experiments/asrr_tuning_v1/README.md` "
        "for the original execution sequence.\n"
    )
    save_json(
        OUTPUT / "manifest.json",
        {
            str(path.relative_to(OUTPUT)): digest(path)
            for path in sorted(OUTPUT.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    )
    save_json(
        OUTPUT / "completion.json",
        {
            "manifest_sha256": digest(OUTPUT / "manifest.json"),
            "paper_sha256": digest(Path("paper/main.pdf")),
            "policies": len(checks),
            "stages": compute,
        },
    )
    print(json.dumps({"policies": len(checks), "stages": compute}))


if __name__ == "__main__":
    run()
