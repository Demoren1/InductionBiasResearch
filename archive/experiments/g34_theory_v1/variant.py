"""Refit a fixed-size alternative bank under the original observation boundary."""

import argparse
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    load_sizes,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_numeric import (
    fit_batch,
    numpy_predict,
)
from experiments.sharing_structure_research_v1.blind_orientation_refine import (
    initial_states,
)
from experiments.sharing_structure_research_v1.blind_runtime import (
    ObservationBoundary,
)

from .bank import build_pool, choose_bank


def propose(root, case, condition):
    folder = root / "fits" / f"{case}_c{condition}"
    base = dict(np.load(folder / "fitted.npz"))
    joint = dict(np.load(folder / "joint_bank.npz"))
    source = [
        dict(
            np.load(
                root / "source" / f"{case}_c{condition}_t{t}" / "learned.npz"
            )
        )
        for t in range(base["parameters"].shape[1])
    ]
    indices, guesses, records = build_pool(base, joint, source)
    np.savez_compressed(
        folder / "theory_pool.npz", indices=indices, code_guesses=guesses
    )
    save_json(folder / "theory_pool.json", {"records": records})
    for method in ("diversity", "theory"):
        target = folder / method
        target.mkdir(exist_ok=False)
        selected = choose_bank(records, method)
        np.savez_compressed(
            target / "bank.npz",
            indices=indices[selected],
            code_guesses=guesses[selected],
        )
        save_json(
            target / "bank.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "pool_count": len(records),
                "unique_pool_count": len({r["key"] for r in records}),
                "selected": selected,
                "records": [records[i] for i in selected],
                "refit_limit": 32,
                "teacher_access": False,
                "bank_sha256": digest(target / "bank.npz"),
            },
        )
    print(
        json.dumps({"case": case, "phase": "proposals", "pool": len(records)})
    )


def fit(root, case, condition, method, device):
    sizes = load_sizes(root, case)
    codes = sizes["codes"]
    common = root / "fits" / f"{case}_c{condition}"
    folder = common / method
    base, bank = [
        dict(np.load(path))
        for path in (common / "fitted.npz", folder / "bank.npz")
    ]
    tasks, keep = base["parameters"].shape[1], 4
    source = [
        dict(
            np.load(
                root / "source" / f"{case}_c{condition}_t{t}" / "learned.npz"
            )
        )
        for t in range(tasks)
    ]
    support = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "fixed_train.npz")
    )
    validation = dict(
        np.load(root / "public" / f"{case}_c{condition}" / "selection.npz")
    )
    x, y = [
        np.stack([support[f"{name}_{t}"] for t in range(tasks)])
        for name in ("x", "y")
    ]
    vx, vy = [
        np.stack([validation[f"{name}_{t}"] for t in range(tasks)])
        for name in ("x", "y")
    ]
    indices = list(base["indices"])
    states = list(base["parameters"])
    risks = list(base["validation_mse"])
    origins = [{"family": "anchor", "anchor": n} for n in range(len(indices))]
    lookup = {index.tobytes(): n for n, index in enumerate(indices)}
    selected, initial = [], []
    records = json.loads((folder / "bank.json").read_text())["records"]
    for number, index in enumerate(bank["indices"]):
        key = index.tobytes()
        if key not in lookup:
            lookup[key] = len(indices)
            indices.append(index)
            states.append(np.zeros_like(base["parameters"][0]))
            risks.append(np.full(tasks, np.inf))
            origins.append(records[number])
        target = lookup[key]
        selected.append(target)
        initial.append(
            [
                initial_states(
                    index,
                    {
                        "codes_joint": bank["code_guesses"][number, t],
                        **source[t],
                    },
                    codes,
                    x[t],
                    y[t],
                    vx[t],
                    vy[t],
                    device,
                    keep,
                )
                for t in range(tasks)
            ]
        )
    indices, states, risks = map(np.asarray, (indices, states, risks))
    history = []
    for begin in range(0, len(selected), 4):
        ids = selected[begin : begin + 4]
        count = len(ids)
        panels = [
            np.tile(np.repeat(a, keep, axis=0), (count, 1, 1))
            for a in (x, y, vx, vy)
        ]
        parameters, trace = fit_batch(
            np.repeat(indices[ids], tasks * keep, axis=0),
            np.asarray(initial[begin : begin + count]).reshape(
                count * tasks * keep, -1
            ),
            *panels,
            codes,
            device,
            steps=48,
        )
        parameters = parameters.reshape(count, tasks, keep, -1)
        loss = trace["validation_mse"].reshape(count, tasks, keep)
        chosen = loss.argmin(2)
        values = loss[
            np.arange(count)[:, None], np.arange(tasks)[None], chosen
        ]
        numerical = parameters[
            np.arange(count)[:, None], np.arange(tasks)[None], chosen
        ]
        for local, candidate in enumerate(ids):
            improved = values[local] < risks[candidate]
            states[candidate, improved] = numerical[local, improved]
            risks[candidate, improved] = values[local, improved]
        history.append(
            {
                "candidates": ids,
                "train": trace["train_history"].tolist(),
                "validation": trace["validation_history"].tolist(),
            }
        )
        print(
            json.dumps(
                {
                    "case": case,
                    "method": method,
                    "refitted": begin + count,
                    "total": len(selected),
                }
            ),
            flush=True,
        )
    np.savez_compressed(
        folder / "fitted.npz",
        indices=indices,
        parameters=states,
        validation_mse=risks,
    )
    save_json(
        folder / "fitted.json",
        {
            "history": history,
            "origins": origins,
            "refitted": selected,
            "steps": 48,
            "keep": keep,
            "teacher_access": False,
        },
    )


def select(root, case, condition, method):
    sizes = load_sizes(root, case)
    folder = root / "fits" / f"{case}_c{condition}" / method
    fitted = dict(np.load(folder / "fitted.npz"))
    meta = dict(np.load(root / "public" / f"{case}_c{condition}" / "meta.npz"))
    tasks = fitted["parameters"].shape[1]
    risk, predictions = [], []
    for index, parameters in zip(fitted["indices"], fitted["parameters"]):
        values = np.stack(
            [
                numpy_predict(
                    meta[f"x_{t}"],
                    index,
                    parameters[t],
                    sizes["codes"],
                    sizes["outputs"],
                )
                for t in range(tasks)
            ]
        )
        observed = np.stack([meta[f"y_{t}"] for t in range(tasks)])
        risk.append(float(np.mean((values - observed) ** 2)))
        predictions.append(values)
    chosen = int(np.argmin(risk))
    origin = json.loads((folder / "fitted.json").read_text())["origins"][
        chosen
    ]
    np.savez_compressed(
        folder / "selected.npz",
        index=fitted["indices"][chosen],
        parameters=fitted["parameters"][chosen],
        meta_predictions=predictions[chosen],
    )
    save_json(
        folder / "selected.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "candidate": chosen,
            "meta_mse": risk[chosen],
            "all_meta_mse": risk,
            "origin": origin,
            "teacher_access": False,
            "sha256": digest(folder / "selected.npz"),
        },
    )
    print(
        json.dumps({"case": case, "method": method, "meta_mse": risk[chosen]})
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument(
        "--method", choices=("theory", "diversity"), default="theory"
    )
    parser.add_argument(
        "--phase", choices=("proposals", "fit", "select"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(2)
    boundary = ObservationBoundary(root, args.case, args.condition)
    boundary.phase = f"{args.method}_{args.phase}"
    boundary.role = "select" if args.phase == "select" else "train"
    boundary.install()
    if root.name == "confirmation":
        registration = json.loads(
            (root / "theory_registration.json").read_text()
        )
        for name, expected in registration["code_sha256"].items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise RuntimeError(f"Registered proposal code changed: {name}")
    device = torch.device(args.device)
    if args.phase == "fit":
        torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    status = "failed"
    try:
        with counter:
            if args.phase == "proposals":
                propose(root, args.case, args.condition)
            elif args.phase == "fit":
                fit(root, args.case, args.condition, args.method, device)
            else:
                select(root, args.case, args.condition, args.method)
        if args.phase == "fit":
            torch.cuda.synchronize(device)
        status = "complete"
    finally:
        key = f"{args.case}_c{args.condition}_{args.method}_{args.phase}"
        save_json(
            root / "metrics" / f"{key}.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops(),
                "cpu_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                    device
                )
                if args.phase == "fit"
                else 0,
                "flop_scope": "PyTorch registered operators; NumPy/SciPy excluded",
            },
        )
        boundary.save(root / "receipts" / f"{key}.json", status)


if __name__ == "__main__":
    main()
