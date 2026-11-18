"""One-time provenance and selection audit of the completed scientific run."""

import json
from datetime import UTC, datetime

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .prepare import BASE, METHODS, OUTPUT, ROOT


def run():
    destination = OUTPUT / "audit.json"
    if destination.exists():
        raise FileExistsError("The completed run has already been audited")
    result, diagnostics = {}, {}
    for dataset in ("mnist", "fashion", "har"):
        root, output = ROOT / dataset, OUTPUT / dataset
        protocol = json.loads((root / "protocol.json").read_text())
        for name, sha in protocol["source_sha256"].items():
            if digest(root / "public" / f"{name}.npz") != sha:
                raise RuntimeError(f"Changed source observations: {dataset}/{name}")
        if dataset != "har" and digest(root / "source/dense/models.npz") != digest(
            BASE / dataset / "source/dense/models.npz"
        ):
            raise RuntimeError("Extractors did not share the original Dense archive")
        source_count = 0
        for path in (root / "fits").glob("*/selected.json"):
            selected = json.loads(path.read_text())
            registered = json.loads(path.with_name("registered.json").read_text())
            count = registered.get("starts", registered.get("restarts"))
            steps = registered.get("steps", registered.get("adam_steps"))
            if count != 18 or steps != 2000:
                raise RuntimeError(f"Unequal numerical budget: {path}")
            risk = selected["all_source_M_losses"]
            if not np.isclose(
                min(risk), selected["source_M_loss"], rtol=1e-6, atol=1e-8
            ):
                raise RuntimeError(f"Incorrect source M selection: {path}")
            if digest(path.with_name("selected.npz")) != selected["selected_sha256"]:
                raise RuntimeError(f"Changed source choice: {path}")
            source_count += 1
        choices = 0
        for method in METHODS:
            folder = root / "validation_fit" / method
            selected = json.loads((folder / "selected.json").read_text())
            curves = np.load(folder / "curves.npz")
            for record in selected["all_budget_choices"]:
                codes = record["codes"]
                score = (
                    curves[f"k{codes}_validation_history"]
                    .mean(-1)
                    .reshape(-1, 9, 2)
                    .mean(-1)
                )
                checkpoint, setting = np.unravel_index(score.argmin(), score.shape)
                lr, quantile = [
                    (lr, q) for lr in (0.003, 0.01, 0.03) for q in (0.1, 0.5, 0.9)
                ][setting]
                if (
                    record["global_setting"]["lr"] != lr
                    or record["global_setting"]["quantile"] != quantile
                    or record["numeric_updates"]
                    != int(curves[f"k{codes}_steps"][checkpoint])
                    or not np.isclose(
                        record["validation_mean_loss"], score[checkpoint, setting]
                    )
                ):
                    raise RuntimeError(
                        f"Incorrect two-restart development minimum: {method}/{codes}"
                    )
                choices += 1
            if selected["validation_mean_loss"] != min(
                r["validation_mean_loss"] for r in selected["all_budget_choices"]
            ):
                raise RuntimeError(f"Incorrect budget choice: {dataset}/{method}")
        freeze = json.loads((output / "development_frozen.json").read_text())
        frozen_predictions = json.loads(
            (output / "predictions_frozen.json").read_text()
        )
        summary = json.loads((output / "summary.json").read_text())
        if freeze["created_utc"] >= frozen_predictions["created_utc"]:
            raise RuntimeError("Invalid freeze ordering")
        for name, files in freeze["models"].items():
            folder = root / "targets" / name
            for filename, sha in files.items():
                if digest(folder / filename) != sha:
                    raise RuntimeError(f"Changed frozen model: {name}")
            if (
                digest(folder / "frozen_predictions.npz")
                != frozen_predictions["predictions_sha256"][name]
            ):
                raise RuntimeError(f"Changed frozen predictions: {name}")
            model = dict(np.load(folder / "model.npz"))
            if set(model) - {"basis", "index", "codes", "scale", "unshared"}:
                raise RuntimeError(f"Unexpected transferred numerical state: {name}")
            if "codes" in model and model["codes"].ndim:
                raise RuntimeError("A learned codebook was transferred")
            if name in ("dense", "unshared"):
                continue
            codes = int(name.rsplit("_k", 1)[1])
            index = model["index"]
            key = f"{dataset}/{name}"
            if index.size:
                groups = np.bincount(np.abs(index).ravel(), minlength=codes + 1)
                active = max(1, np.count_nonzero(index))
                diagnostics[key] = {
                    "representation": "signed indexator",
                    "allocated_codes": codes,
                    "used_nonzero_codes": int(np.count_nonzero(groups[1:])),
                    "zero_fraction": float(np.mean(index == 0)),
                    "largest_nonzero_group_fraction": float(
                        groups[1:].max() / index.size
                    ),
                    "largest_group_fraction_of_active": float(
                        groups[1:].max() / active
                    ),
                    "empty_neurons": int(
                        np.count_nonzero(np.count_nonzero(index, axis=0) == 0)
                    ),
                }
                if "basis" in model:
                    expected = (
                        np.eye(codes + 1, dtype=np.float32)[np.abs(index)][..., 1:]
                        * np.sign(index)[..., None]
                    )
                    expected /= max(
                        float(np.sqrt(np.mean(np.sum(expected**2, -1)))), 1e-12
                    )
                    if not np.allclose(model["basis"], expected, rtol=1e-6, atol=1e-7):
                        raise RuntimeError(
                            f"Basis includes information beyond I: {name}"
                        )
                else:
                    expected_scale = np.sqrt(index.size / active)
                    if not np.isclose(model["scale"], expected_scale):
                        raise RuntimeError("Nonstructural numeric scale transferred")
            else:
                diagnostics[key] = {
                    "representation": "native continuous basis",
                    "allocated_coordinates": codes,
                    "basis_shape": list(model["basis"].shape),
                }
        receipts = [
            json.loads(p.read_text()) for p in root.rglob("*access_receipt.json")
        ]
        violations = [v for r in receipts for v in r.get("violations", [])]
        if violations:
            raise RuntimeError(f"Held-out access violations: {violations}")
        if not summary["full_code_endpoint_equals_dense_exactly"]:
            raise RuntimeError("The executed full-budget control is not exactly Dense")
        result[dataset] = {
            "source_budgets": source_count,
            "development_budget_rules": choices,
            "frozen_target_models": len(freeze["models"]),
            "access_receipts": len(receipts),
            "access_violations": 0,
            "source_and_development_selection_verified": True,
            "only_structure_transferred": True,
            "full_budget_predictions_equal_dense": True,
        }
    from .har import read_split

    x, y, people = read_split("train")
    ids = json.loads((OUTPUT / "har/observation_ids.json").read_text())
    all_panels = []
    for name, panels in ids["source"].items():
        ix = np.asarray(panels)
        panel = np.load(ROOT / "har/public" / f"{name}.npz")
        if not np.array_equal(panel["x"], x[ix]) or not np.array_equal(
            panel["y"], np.eye(6, dtype=np.float32)[y[ix]]
        ):
            raise RuntimeError(
                "HAR source observations differ from official original rows"
            )
        all_panels.append(set(ix.ravel()))
    if any(all_panels[a] & all_panels[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Overlapping HAR source fitting/selection rows")
    source_people = set(people[list(set.union(*all_panels))])
    dev_people = set(r["person"] for r in ids["development"])
    target_ids = json.loads((ROOT / "har/private/target_ids.json").read_text())
    target_people = set(r["person"] for r in target_ids)
    if (
        source_people & dev_people
        or source_people & target_people
        or dev_people & target_people
    ):
        raise RuntimeError("People overlap across HAR roles")
    if (len(source_people), len(dev_people), len(target_people)) != (16, 5, 9):
        raise RuntimeError("Unexpected HAR subject counts")
    if any(set(r["support_ids"]) & set(r["query_ids"]) for r in target_ids):
        raise RuntimeError("Overlapping support/query windows")
    result["har"]["official_source_rows_and_subject_separation_verified"] = True
    for number in (4, 5):
        root = ROOT / f"synthetic_{number}"
        frozen = json.loads((root / "fits/frozen.json").read_text())
        if digest(root / "fits/structures.npz") != frozen["models_sha256"]:
            raise RuntimeError("Synthetic choices changed after freeze")
        if json.loads((root / "fits/access_receipt.json").read_text())["violations"]:
            raise RuntimeError("Synthetic source boundary violation")
        for method in METHODS:
            folder = root / "targets" / f"case_{number:02d}_c0" / method
            if json.loads((folder / "access_receipt.json").read_text())["violations"]:
                raise RuntimeError("Synthetic target boundary violation")
    save_json(OUTPUT / "structure_diagnostics.json", diagnostics)
    save_json(
        destination,
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "passed",
            "datasets": result,
            "synthetic_models_frozen": True,
            "software_tests_created_or_run": False,
        },
    )


if __name__ == "__main__":
    run()
