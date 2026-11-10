"""Fit and select common I1 without reading teacher metadata or parameters."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .blind_contract import canonical_index, digest, load_sizes, save_json
from .blind_numeric import fit_batch, initial_states, numpy_predict

BASE = Path(__file__).resolve().parent


def fit_bank(root, case, condition, sources, device, steps):
    started = time.monotonic()
    sizes = load_sizes(root, case)
    if set(sizes) != {"inputs", "hidden", "codes", "outputs"}:
        raise ValueError("Only public architecture sizes may enter fitting")
    out = root / "fits" / f"{case}_c{condition}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "code_snapshot").mkdir()
    for name in ("blind_numeric.py", "blind_pipeline.py"):
        shutil.copy2(BASE / name, out / "code_snapshot" / name)
    observed, bank, origins = [], {}, {}
    for task in range(sources):
        folder = root / "source" / f"{case}_c{condition}_t{task}"
        with np.load(folder / "learned.npz") as archive:
            data = dict(archive)
        observed.append(data)
        trace = json.loads((folder / "trace.json").read_text())
        for row in trace.get("candidates", []):
            index = canonical_index(data[f"index_{row['number']}"], sizes["codes"])
            key = index.tobytes()
            bank[key] = index
            origins.setdefault(key, []).append(
                {"task": task, "projection_mse": row["projection_mse"]}
            )
    if not bank:
        raise ValueError("No complete observed indexator candidates")
    indices = np.stack(list(bank.values()))
    with np.load(root / "public" / f"{case}_c{condition}" / "selection.npz") as archive:
        selection = dict(archive)
    fixed_path = root / "public" / f"{case}_c{condition}" / "fixed_train.npz"
    if fixed_path.exists():
        with np.load(fixed_path) as archive:
            x = np.stack([archive[f"x_{task}"] for task in range(sources)])
            y = np.stack([archive[f"y_{task}"] for task in range(sources)])
    else:
        x = np.stack([data["fit_x"] for data in observed])
        y = np.stack([data["fit_y"] for data in observed])
    vx = np.stack([selection[f"x_{task}"] for task in range(sources)])
    vy = np.stack([selection[f"y_{task}"] for task in range(sources)])
    keep = 2
    initial = []
    for number, index in enumerate(indices):
        states = []
        for task in range(sources):
            states.append(
                initial_states(
                    index,
                    observed[task],
                    sizes["codes"],
                    x[task],
                    y[task],
                    vx[task],
                    vy[task],
                    keep,
                    device=device,
                )
            )
        initial.append(states)
        if number % 8 == 0 or number == len(indices) - 1:
            print(
                json.dumps(
                    {
                        "phase": "initialize",
                        "case": case,
                        "condition": condition,
                        "candidate": number,
                        "bank": len(indices),
                    }
                ),
                flush=True,
            )
    initial = np.asarray(initial)
    fitted, scores, checkpoints = [], [], []
    train_history, validation_history = [], []
    for begin in range(0, len(indices), 4):
        end = min(begin + 4, len(indices))
        batch = end - begin
        packed_indices = np.repeat(indices[begin:end], sources * keep, axis=0)
        packed_initial = initial[begin:end].reshape(batch * sources * keep, -1)
        packed = [
            np.tile(np.repeat(array, keep, axis=0), (batch, 1, 1))
            for array in (x, y, vx, vy)
        ]
        parameters, trace = fit_batch(
            packed_indices, packed_initial, *packed, sizes["codes"], device, steps=steps
        )
        fitted.append(parameters.reshape(batch, sources, keep, -1))
        scores.append(trace["validation_mse"].reshape(batch, sources, keep))
        checkpoints.append(trace["selected_step"].reshape(batch, sources, keep))
        train_history.append(
            trace["train_history"].reshape(steps + 1, batch, sources, keep)
        )
        validation_history.append(
            trace["validation_history"].reshape(steps + 1, batch, sources, keep)
        )
        print(
            json.dumps(
                {
                    "phase": "polish",
                    "case": case,
                    "condition": condition,
                    "candidates": [begin, end],
                    "seconds": trace["seconds"],
                    "device": trace["device"],
                }
            ),
            flush=True,
        )
    fitted, scores, checkpoints = (
        np.concatenate(fitted),
        np.concatenate(scores),
        np.concatenate(checkpoints),
    )
    train_history, validation_history = (
        np.concatenate(train_history, axis=1),
        np.concatenate(validation_history, axis=1),
    )
    chosen = scores.argmin(axis=2)
    candidate_rows = np.arange(len(indices))[:, None]
    task_rows = np.arange(sources)[None]
    selected = fitted[candidate_rows, task_rows, chosen]
    initial_score = validation_history[0]
    initial_chosen = initial_score.argmin(axis=2)
    initial_selected = initial[candidate_rows, task_rows, initial_chosen]
    np.savez_compressed(
        out / "fitted.npz",
        indices=indices,
        parameters=selected,
        initial_parameters=initial_selected,
        validation_mse=scores[candidate_rows, task_rows, chosen],
        selected_step=checkpoints[candidate_rows, task_rows, chosen],
        train_history=train_history,
        validation_history=validation_history,
    )
    save_json(
        out / "fitted.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "case": case,
            "condition": condition,
            "sources": sources,
            "bank": len(indices),
            "sizes": sizes,
            "seconds": time.monotonic() - started,
            "device": str(device),
            "steps": steps,
            "origins": [origins[key] for key in bank],
            "sha256": {
                "fitted.npz": digest(out / "fitted.npz"),
                "selection.npz": digest(
                    root / "public" / f"{case}_c{condition}" / "selection.npz"
                ),
            },
            "teacher_access": False,
        },
    )


def select(root, case, condition):
    folder = root / "fits" / f"{case}_c{condition}"
    frozen = json.loads((folder / "fitted.json").read_text())
    if digest(folder / "fitted.npz") != frozen["sha256"]["fitted.npz"]:
        raise ValueError("Frozen numerical fits changed")
    sizes, sources = frozen["sizes"], frozen["sources"]
    with np.load(folder / "fitted.npz") as archive:
        fitted = dict(archive)
    with np.load(root / "public" / f"{case}_c{condition}" / "meta.npz") as archive:
        meta = dict(archive)
    records, predictions = {}, {}
    for method, key in (
        ("geometric_initialization", "initial_parameters"),
        ("functional_polish", "parameters"),
    ):
        losses, values = [], []
        for index, parameters in zip(fitted["indices"], fitted[key]):
            task_predictions = [
                numpy_predict(
                    meta[f"x_{task}"],
                    index,
                    parameters[task],
                    sizes["codes"],
                    sizes["outputs"],
                )
                for task in range(sources)
            ]
            losses.append(
                [
                    float(np.mean((value - meta[f"y_{task}"]) ** 2))
                    for task, value in enumerate(task_predictions)
                ]
            )
            values.append(task_predictions)
        losses = np.asarray(losses)
        chosen = int(losses.mean(axis=1).argmin())
        records[method] = {
            "candidate": chosen,
            "meta_mse": float(losses[chosen].mean()),
            "task_meta_mse": losses[chosen].tolist(),
            "all_meta_mse": losses.mean(axis=1).tolist(),
        }
        predictions[f"{method}_meta"] = np.stack(values[chosen])
        predictions[f"{method}_index"] = fitted["indices"][chosen]
        predictions[f"{method}_parameters"] = fitted[key][chosen]
    np.savez_compressed(folder / "selected.npz", **predictions)
    save_json(
        folder / "selected.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "methods": records,
            "sha256": {
                "selected.npz": digest(folder / "selected.npz"),
                "fitted.json": digest(folder / "fitted.json"),
                "meta.npz": digest(
                    root / "public" / f"{case}_c{condition}" / "meta.npz"
                ),
            },
            "teacher_access": False,
        },
    )
    print(
        json.dumps({"selected": case, "condition": condition, "methods": records}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fit", "select"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, required=True)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "fit":
        fit_bank(
            args.root.resolve(),
            args.case,
            args.condition,
            args.sources,
            torch.device(args.device),
            args.steps,
        )
    else:
        select(args.root.resolve(), args.case, args.condition)


if __name__ == "__main__":
    main()
