"""Audit the completed scientific choices and export their provenance."""

import json
import shutil
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .prepare import GRID, OUTPUT, PREVIOUS, ROOT


def run():
    rows = []
    for dataset in ("mnist", "fashion", "har"):
        root, output = ROOT / dataset, OUTPUT / dataset
        summary = json.loads((output / "summary.json").read_text())
        source = json.loads((output / "source_reuse.json").read_text())
        if (
            not source["identical"]
            or not summary["same_original_observations"]
        ):
            raise RuntimeError(f"Unmatched observations for {dataset}")
        selection = json.loads(
            (output / "development/selected.json").read_text()
        )
        best = min(
            selection["all_budget_choices"],
            key=lambda row: row["validation_mean_loss"],
        )
        if selection["codes"] != best["codes"]:
            raise RuntimeError("Development did not select its minimum")
        with np.load(output / "development/curves.npz") as curves:
            for codes in GRID:
                source_folder = output / f"source_fit/k{codes}"
                source_choice = json.loads(
                    (source_folder / "selected.json").read_text()
                )
                risk = np.asarray(source_choice["all_source_M_losses"])
                if source_choice["source_M_loss"] != float(risk.min()):
                    raise RuntimeError("Source M did not select its minimum")
                record = next(
                    r
                    for r in selection["all_budget_choices"]
                    if r["codes"] == codes
                )
                score = curves[f"k{codes}_validation_history"].mean(-1)
                score = score.reshape(-1, 9, 2).mean(-1)
                step, setting = np.unravel_index(score.argmin(), score.shape)
                if record["validation_mean_loss"] != float(
                    score[step, setting]
                ) or record["numeric_updates"] != int(
                    curves[f"k{codes}_steps"][step]
                ):
                    raise RuntimeError(
                        "Development rule differs from saved curves"
                    )
                model_path = output / f"development/budgets/k{codes}/model.npz"
                with np.load(model_path) as model:
                    index = model["index"]
                    if len(np.unique(np.abs(index))) != codes or np.any(
                        index == 0
                    ):
                        raise RuntimeError("Wrong aSRR support or group count")
                    if "basis" in model and model["basis"].size:
                        basis = np.eye(codes + 1)[np.abs(index)][..., 1:]
                        basis *= np.sign(index)[..., None] * float(
                            model["scale"]
                        )
                        if not np.array_equal(basis, model["basis"]):
                            raise RuntimeError(
                                "Transferred basis is not determined by indexator"
                            )
                target = root / "targets" / f"asrr_k{codes}"
                if digest(model_path) != digest(target / "model.npz"):
                    raise RuntimeError("Transferred model changed")
                rows.append(
                    {
                        "dataset": dataset,
                        "codes": codes,
                        "source_candidate": source_choice["candidate"],
                        "development_average_restarts": 2,
                        "target_restarts": 1,
                        "structure_only": True,
                    }
                )
        for path in (
            output / "extraction_access.json",
            output / "development/access_receipt.json",
            *(output / f"source_fit/k{k}/access_receipt.json" for k in GRID),
        ):
            if json.loads(path.read_text())["violations"]:
                raise RuntimeError(f"Data access violation: {path}")
    save_json(
        OUTPUT / "scientific_audit.json",
        {
            "applied_budget_choices": rows,
            "same_source_dense_archives": True,
            "same_original_target_observations": True,
            "query_freezes_checked_during_scoring": True,
            "software_tests_run": False,
        },
    )
    snapshot = OUTPUT / "final_source_snapshot"
    snapshot.mkdir()
    packages = (
        "paper_asrr_v1",
        "srr_approx_v1",
        "paper_recovery_extension_v1",
        "paper_structure_benchmark_v1/fitting",
        "paper_structure_benchmark_v1/protocol",
    )
    for package in packages:
        shutil.copytree(
            Path("experiments") / package,
            snapshot / package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    save_json(
        OUTPUT / "source_manifest.json",
        {
            str(p.relative_to(snapshot)): digest(p)
            for p in sorted(snapshot.rglob("*"))
            if p.is_file()
        },
    )
    (OUTPUT / "README.md").write_text(
        "# aSRR under the existing manuscript protocol\n\n"
        "17 November 2026. Added-method evaluation on the original source, "
        "development and target observations. Baselines remain immutable in "
        f"`{PREVIOUS}`.\n\n"
        "Selected banks/curves and all query predictions/answers are retained; "
        "`scientific_audit.json` checks the 18 applied budget choices. "
        "`source_manifest.json` indexes the final source snapshot. "
        "Registration precedes candidate fitting; choices and predictions "
        "were frozen before aSRR scoring. HAR Dense was regenerated and "
        "matches the original archive SHA256 exactly.\n\n"
        "Report: `paper/theory/ASRR_MATCHED_RESULTS_2026-11-17.md`.\n\n"
        "Regenerate manuscript tables and the budget plot without GPUs:\n\n"
        "```bash\nMPLCONFIGDIR=/tmp/paper-asrr-mpl "
        "python -m experiments.paper_asrr_v1.report\nmake -C paper\n```\n\n"
        "Do not repeat freeze, scoring or finalization over completed folders. "
        "No software tests were created or run.\n"
    )


if __name__ == "__main__":
    run()
