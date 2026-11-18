"""Source bank calibration, independent development and frozen scoring."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import t

from experiments.paper_structure_benchmark_v1.fitting.adaptation import (
    fit_adam,
    losses,
    signed_basis,
    values,
)
from experiments.paper_structure_benchmark_v1.fitting.real_bank_fit import dense_initial
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def refit(root, output, method, codes, device):
    torch.set_num_threads(2)
    boundary = StudyBoundary(root)
    boundary.install()
    proposal = (
        root
        / "source"
        / ("current_proposals" if method == "g34" else "ratio_proposals")
    )
    metadata = json.loads((proposal / "candidates.json").read_text())["candidates"]
    rows = [r for r in metadata if r["codes"] == codes]
    candidates = dict(np.load(proposal / "candidates.npz"))
    folder = root / "fits" / f"{method}_k{codes}"
    folder.mkdir(exist_ok=False)
    save_json(
        folder / "registered.json",
        {
            "method": method,
            "codes": codes,
            "candidates": rows,
            "proposal_sha256": digest(proposal / "candidates.npz"),
            "steps": 2000,
            "starts": 18,
            "same_numerical_adapter": True,
            "adapter_sha256": digest(
                Path("experiments/paper_structure_benchmark_v1/fitting/adaptation.py")
            ),
        },
    )
    arrays = [
        dict(np.load(root / "public" / f"{name}.npz"))
        for name in ("support", "numerical")
    ]
    x, y, vx, vy = [
        torch.tensor(panel[key], device=device, dtype=torch.float32)
        for panel, key in (
            (arrays[0], "x"),
            (arrays[0], "y"),
            (arrays[1], "x"),
            (arrays[1], "y"),
        )
    ]
    x, vx = x / 255, vx / 255
    dense = torch.tensor(
        np.load(root / "source/dense/models.npz")["parameters"], device=device
    )
    bases, numeric, indices, traces = [], [], [], []
    for row in rows:
        index = candidates[row["name"]]
        basis = signed_basis(index, codes, device, torch.float32)
        basis /= basis.square().sum(-1).mean().sqrt().clamp_min(1e-12)
        initial = dense_initial(basis, dense, row["anchor"], 16, 2)
        fitted, trace = fit_adam(
            x,
            y,
            vx,
            vy,
            basis,
            16,
            2,
            loss="cross_entropy",
            steps=2000,
            initial=initial,
        )
        bases.append(basis.cpu().numpy())
        numeric.append(fitted[0].cpu().numpy())
        indices.append(index)
        traces.append(trace)
        print(
            json.dumps(
                {
                    "bank": method,
                    "codes": codes,
                    "candidate": row["name"],
                    "done": len(bases),
                    "total": len(rows),
                }
            ),
            flush=True,
        )
    boundary.role = "select"
    panel = dict(np.load(root / "public/structure.npz"))
    mx = torch.tensor(panel["x"], device=device, dtype=torch.float32) / 255
    my = torch.tensor(panel["y"], device=device, dtype=torch.float32)
    with torch.no_grad():
        risk = np.stack(
            [
                losses(
                    values(
                        mx,
                        torch.tensor(b, device=device),
                        torch.tensor(p, device=device)[None],
                        16,
                        2,
                    ),
                    my,
                    "cross_entropy",
                )[0]
                .cpu()
                .numpy()
                for b, p in zip(bases, numeric)
            ]
        )
    chosen = int(risk.mean(1).argmin())
    np.savez_compressed(
        folder / "selected.npz",
        basis=bases[chosen],
        index=indices[chosen],
        source_M_loss=risk[chosen],
    )
    save_json(
        folder / "selected.json",
        {
            "method": method,
            "codes": codes,
            "source_M_loss": float(risk[chosen].mean()),
            "candidate": rows[chosen],
            "all_source_M_losses": risk.mean(1).tolist(),
            "selected_sha256": digest(folder / "selected.npz"),
        },
    )
    np.savez_compressed(
        folder / "curves.npz",
        **{
            k: traces[chosen][k]
            for k in ("steps", "train_history", "validation_history")
        },
    )
    boundary.save(folder / "access_receipt.json", "complete")
    shutil.copytree(folder, output / "source_selection" / folder.name)


def tune(root, output, method, device):
    torch.set_num_threads(2)
    boundary = StudyBoundary(root, "validation")
    boundary.install()
    folder = root / "validation_fit" / method
    folder.mkdir(parents=True, exist_ok=False)
    with np.load(root / "validation/validation.npz") as panel:
        x, y, vx, vy = [
            torch.tensor(panel[key], device=device, dtype=torch.float32)
            for key in ("support_x", "support_y", "query_x", "query_y")
        ]
    x, vx = x / 255, vx / 255
    budgets = (
        [0]
        if method == "dense"
        else [
            k
            for k in (8, 16, 32, 64)
            if (root / "fits" / f"{method}_k{k}" / "selected.npz").exists()
        ]
    )
    best, records, all_curves = None, [], {}
    for codes in budgets:
        basis, index = None, np.empty(0, dtype=np.int16)
        if codes:
            with np.load(
                root / "fits" / f"{method}_k{codes}" / "selected.npz"
            ) as model:
                basis = torch.tensor(model["basis"], device=device)
                index = model["index"].copy()
        _, trace = fit_adam(
            x, y, vx, vy, basis, 16, 2, loss="cross_entropy", steps=2000, seed=769003
        )
        scores = trace["validation_history"].mean(-1).reshape(-1, 9, 2).mean(-1)
        checkpoint, setting = np.unravel_index(scores.argmin(), scores.shape)
        record = {
            "method": method,
            "codes": codes,
            "validation_mean_loss": float(scores[checkpoint, setting]),
            "numeric_updates": int(trace["steps"][checkpoint]),
            "global_setting": trace["settings"][2 * setting],
            "support_count": 32,
            "source_selection": {"codes": codes},
        }
        records.append(record)
        all_curves.update(
            {
                f"k{codes}_{key}": value
                for key, value in trace.items()
                if isinstance(value, np.ndarray)
            }
        )
        if (
            best is None
            or record["validation_mean_loss"] < best["validation_mean_loss"]
        ):
            best = record
            np.savez_compressed(
                folder / "model.npz",
                index=index,
                basis=np.empty(0) if basis is None else basis.cpu().numpy(),
            )
        print(json.dumps({"development": record}), flush=True)
    if best is None:
        save_json(
            folder / "selected.json", {"method": method, "status": "no_candidates"}
        )
        return
    save_json(
        folder / "selected.json",
        {
            **best,
            "all_budget_choices": records,
            "model_sha256": digest(folder / "model.npz"),
        },
    )
    np.savez_compressed(folder / "curves.npz", **all_curves)
    boundary.save(folder / "access_receipt.json", "complete")
    shutil.copytree(folder, output / "development" / method)


def export(root, output):
    protocol = json.loads((root / "protocol.json").read_text())
    private = root / "private/target.npz"
    if digest(private) != protocol["target_data_sha256"]:
        raise RuntimeError("Target observations changed after preparation")
    with np.load(private) as data:
        np.savez_compressed(
            root / "targets/support.npz", x=data["support_x"], y=data["support_y"]
        )
        np.savez_compressed(root / "targets/query_x.npz", x=data["query_x"])
    save_json(
        root / "targets/sizes.json",
        {k: protocol[k] for k in ("inputs", "hidden", "outputs", "loss")},
    )
    models = {}
    for method in ("dense", "g34", "pair"):
        origin = root / "validation_fit" / method
        selected = json.loads((origin / "selected.json").read_text())
        if selected.get("status") == "no_candidates":
            continue
        destination = root / "targets" / method
        destination.mkdir()
        for name in ("model.npz", "selected.json"):
            shutil.copy2(origin / name, destination / name)
        models[method] = {
            name: digest(destination / name) for name in ("model.npz", "selected.json")
        }
    save_json(
        output / "development_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "models": models,
            "private_sha256": digest(private),
            "target_query_labels_read": False,
            "target_support_sha256": digest(root / "targets/support.npz"),
            "target_query_inputs_sha256": digest(root / "targets/query_x.npz"),
        },
    )


def score(root, output):
    freeze = json.loads((output / "development_frozen.json").read_text())
    predictions = {}
    for method, files in freeze["models"].items():
        folder = root / "targets" / method
        for name, expected in files.items():
            if digest(folder / name) != expected:
                raise RuntimeError(f"Changed frozen model: {method}/{name}")
        predictions[method] = digest(folder / "frozen_predictions.npz")
    save_json(
        output / "predictions_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "predictions_sha256": predictions,
            "models": freeze["models"],
            "query_labels_scored": False,
        },
    )
    if digest(root / "private/target.npz") != freeze["private_sha256"]:
        raise RuntimeError("Target labels changed after preparation")
    with np.load(root / "private/target.npz") as data:
        truth = data["query_y"]
    opened = datetime.now(UTC).isoformat()
    arrays, rows, accuracy = {"truth": truth}, [], {}
    for method, expected in predictions.items():
        folder = root / "targets" / method
        if digest(folder / "frozen_predictions.npz") != expected:
            raise RuntimeError("Predictions changed after freeze")
        value = np.load(folder / "frozen_predictions.npz")["prediction"]
        acc = (value.argmax(-1) == truth.argmax(-1)).mean(1)
        ce = (logsumexp(value, axis=-1) - (value * truth).sum(-1)).mean(1)
        selected = json.loads((folder / "selected.json").read_text())
        codes = 12544 if method == "dense" else selected["source_selection"]["codes"]
        half = float(t.ppf(0.975, len(acc) - 1) * acc.std(ddof=1) / np.sqrt(len(acc)))
        rows.append(
            {
                "method": method,
                "accuracy": float(acc.mean()),
                "accuracy_half_ci": half,
                "cross_entropy": float(ce.mean()),
                "codes": codes,
                "task_parameters": codes + 50,
                "numeric_updates": selected["numeric_updates"],
                "development_ce": selected["validation_mean_loss"],
            }
        )
        accuracy[method] = acc
        arrays[f"{method}_prediction"] = value
        arrays[f"{method}_accuracy"] = acc
        arrays[f"{method}_ce"] = ce
        shutil.copytree(folder, output / "targets" / method)
    paired = {}
    if "pair" in accuracy:
        for other in ("g34", "dense"):
            delta = accuracy["pair"] - accuracy[other]
            half = float(
                t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta))
            )
            paired[f"pair_minus_{other}"] = {
                "mean": float(delta.mean()),
                "half_ci": half,
            }
    receipts = [json.loads(p.read_text()) for p in root.rglob("*access_receipt.json")]
    save_json(
        output / "summary.json",
        {
            "methods": rows,
            "paired": paired,
            "target_episodes": len(truth),
            "query_answers_opened_utc": opened,
            "access_violations": [v for r in receipts for v in r.get("violations", [])],
            "ci_scope": (
                "Episode variability conditional on class split and source cohort; "
                "images/classes may repeat"
            ),
            "true_neural_indexator_available": False,
        },
    )
    np.savez_compressed(output / "scoring.npz", **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("refit", "tune", "export", "score"), required=True
    )
    parser.add_argument("--method", choices=("g34", "pair", "dense"))
    parser.add_argument("--codes", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, output, device = (
        args.root.resolve(),
        args.output.resolve(),
        torch.device(args.device),
    )
    if args.phase == "refit":
        refit(root, output, args.method, args.codes, device)
    elif args.phase == "tune":
        tune(root, output, args.method, device)
    elif args.phase == "export":
        export(root, output)
    else:
        score(root, output)


if __name__ == "__main__":
    main()
