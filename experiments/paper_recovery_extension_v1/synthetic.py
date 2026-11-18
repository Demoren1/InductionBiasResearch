"""Native competitors and fresh numerical transfer on the two fixed geometries."""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.paper_structure_benchmark_v1.fitting.adaptation import (
    feature,
    fit_adam,
    signed_basis,
    values,
)
from experiments.paper_structure_benchmark_v1.fitting.numerical import fit_features
from experiments.paper_structure_benchmark_v1.fitting.real_bank_fit import candidates
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .prepare import OUTPUT, ROOT

ORIGINAL = Path("artifacts/single_task_recovery_2026-11-14")
METHODS = ("g34", "pair", "lutq", "yeh_map", "yeh_soft", "msr")


def prepare():
    for number in (4, 5):
        case = f"case_{number:02d}"
        root = ROOT / f"synthetic_{number}"
        root.mkdir()
        for name in ("public", "source/dense", "fits", "private", "targets"):
            (root / name).mkdir(parents=True, exist_ok=True)
        sizes = json.loads((ORIGINAL / "public" / f"{case}.json").read_text())
        save_json(
            root / "protocol.json",
            {**sizes, "loss": "mse", "independent_code_grid": [4]},
        )
        for source, destination in (
            ("fixed_train", "support"),
            ("selection", "numerical"),
            ("meta", "structure"),
        ):
            with np.load(ORIGINAL / "public" / f"{case}_c0" / f"{source}.npz") as data:
                np.savez_compressed(
                    root / "public" / f"{destination}.npz",
                    x=data["x_0"][None].astype(np.float32) * 255,
                    y=data["y_0"][None].astype(np.float32),
                )
        dense = np.load(ORIGINAL / "dense" / f"{case}_c0/fitted.npz")["parameters"]
        np.savez_compressed(
            root / "source/dense/models.npz", parameters=dense.astype(np.float32)
        )
    save_json(
        OUTPUT / "synthetic_registration.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "source": str(ORIGINAL),
            "methods": ["dense", *METHODS],
            "source_codes": 4,
            "source_calibration": "Common18 fresh Adam starts/2000 then4 GN/48; H checkpoints, M structure selection",
            "target": "Four new numerical functions per fixed geometry,256 fitting/64 H/2048 query; common18 Adam+4 GN48",
            "target_seed": 26111499113,
            "native_steps": {"lutq": 2500, "msr": 2000, "yeh": 400},
            "query_answers_opened": False,
            "continuous_methods_never_postclustered": True,
        },
    )


def native():
    folder = OUTPUT / "synthetic_jobs"
    folder.mkdir()
    jobs = [(n, m) for n in (4, 5) for m in ("lutq", "msr", "yeh")]

    def run(item):
        gpu, (number, method) = item
        command = [
            sys.executable,
            "-u",
            "-m",
            "experiments.paper_structure_benchmark_v1.methods.real_competitors",
            "--root",
            str((ROOT / f"synthetic_{number}").resolve()),
            "--method",
            method,
            "--codes",
            "4",
            "--device",
            f"cuda:{gpu}",
        ]
        with (folder / f"{number}_{method}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(
            json.dumps({"synthetic_native": [number, method], "status": "complete"}),
            flush=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(run, enumerate(jobs)))


def calibrated(basis, x, y, vx, vy, h, o, *, seed):
    initial, trace = fit_adam(
        x.float(),
        y.float(),
        vx.float(),
        vy.float(),
        basis.float(),
        h,
        o,
        seed=seed,
        loss="mse",
        steps=2000,
    )
    keep, tasks = initial.shape[:2]
    feat = (
        feature(x.double(), basis.double())[None]
        .expand(keep, -1, -1, -1, -1)
        .flatten(0, 1)
    )
    vf = (
        feature(vx.double(), basis.double())[None]
        .expand(keep, -1, -1, -1, -1)
        .flatten(0, 1)
    )
    gn, gn_trace = fit_features(
        feat,
        vf,
        initial.double().flatten(0, 1),
        y.double().repeat(keep, 1, 1),
        vy.double().repeat(keep, 1, 1),
        basis.shape[-1],
        steps=48,
    )
    score = gn_trace["numerical_selection_mse"].reshape(keep, tasks)
    fitted = gn.reshape(keep, tasks, -1)[
        score.argmin(0), torch.arange(tasks, device=x.device)
    ]
    return fitted, trace, gn_trace


def source(root, device):
    boundary = StudyBoundary(root)
    boundary.install()
    number = int(root.name.rsplit("_", 1)[1])
    case = f"case_{number:02d}"
    o = json.loads((root / "protocol.json").read_text())["outputs"]
    arrays = [
        dict(np.load(root / "public" / f"{n}.npz")) for n in ("support", "numerical")
    ]
    x, y, vx, vy = [
        torch.tensor(a[k], device=device, dtype=torch.float64)
        for a, k in (
            (arrays[0], "x"),
            (arrays[0], "y"),
            (arrays[1], "x"),
            (arrays[1], "y"),
        )
    ]
    x, vx = x / 255, vx / 255
    center, scale = y.mean(1), y.std((1, 2), correction=0).clamp_min(1e-8)
    y, vy = (
        (y - center[:, None]) / scale[:, None, None],
        (vy - center[:, None]) / scale[:, None, None],
    )
    # This archive contains recovered proposals only in the keys opened below.
    structures = np.load(ORIGINAL / "recovery_structures.npz")
    selected = {}
    for method in METHODS:
        if method in ("g34", "pair"):
            key = "ratio" if method == "pair" else method
            proposals = [
                (f"candidate_{i}", a, 0)
                for i, a in enumerate(structures[f"{case}_{key}_bank"])
            ]
        else:
            proposals = candidates(root, method, 4)[:8]
        bases, fitted_all, risk = [], [], []
        for name, array, _ in proposals:
            basis = (
                signed_basis(array, 4, device)
                if array.ndim == 2
                else torch.tensor(array, device=device, dtype=torch.float64)
            )
            basis /= basis.square().sum(-1).mean().sqrt().clamp_min(1e-12)
            fitted, _, _ = calibrated(basis, x, y, vx, vy, 16, o, seed=761003)
            bases.append(basis.cpu().numpy())
            fitted_all.append(fitted)
            print(
                json.dumps({"synthetic_source": method, "candidate": name}), flush=True
            )
        boundary.role = "select"
        panel = dict(np.load(root / "public/structure.npz"))
        mx = torch.tensor(panel["x"], device=device, dtype=torch.float64) / 255
        my = torch.tensor(panel["y"], device=device, dtype=torch.float64)
        my = (my - center[:, None]) / scale[:, None, None]
        with torch.no_grad():
            for b, p in zip(bases, fitted_all):
                value = values(mx, torch.tensor(b, device=device), p[None], 16, o)[0]
                risk.append(float((value - my).square().mean()))
        best = int(np.argmin(risk))
        selected[method] = bases[best]
        save_json(
            root / "fits" / f"{method}.json",
            {"candidate": proposals[best][0], "source_M_losses": risk},
        )
        boundary.role = "train"
    np.savez_compressed(root / "fits/structures.npz", **selected)
    save_json(
        root / "fits/frozen.json",
        {
            "models_sha256": digest(root / "fits/structures.npz"),
            "frozen_utc": datetime.now(UTC).isoformat(),
        },
    )
    boundary.save(root / "fits/access_receipt.json", "complete")


def target_prepare():
    from experiments.paper_structure_benchmark_v1.protocol.prepare import outputs_for
    from experiments.paper_structure_benchmark_v1.protocol.prepare_general import (
        general_numbers,
    )

    for number in (4, 5):
        root = ROOT / f"synthetic_{number}"
        case = f"case_{number:02d}"
        frozen = json.loads((root / "fits/frozen.json").read_text())
        if digest(root / "fits/structures.npz") != frozen["models_sha256"]:
            raise RuntimeError("Synthetic source choices changed")
        folder = root / "targets" / f"{case}_c0"
        folder.mkdir()
        models = dict(np.load(root / "fits/structures.npz"))
        sizes = json.loads((root / "protocol.json").read_text())
        np.savez_compressed(folder / "structures.npz", **models)
        save_json(folder / "sizes.json", sizes)
        rng = np.random.default_rng(26111499113 + number)
        code, bias, head, offset = general_numbers(
            rng, "random", number % 2, 4, sizes["outputs"]
        )
        teacher = np.load(ORIGINAL / "private" / f"{case}.npz")["index"]
        for name, count in (("support", 256), ("selection", 64), ("query", 2048)):
            x = rng.normal(size=(4, count, 16))
            y = np.stack(
                [
                    outputs_for(x[t], teacher, code[t], bias[t], head[t], offset[t])
                    for t in range(4)
                ]
            )
            if name == "query":
                np.savez_compressed(folder / "query_x.npz", x=x)
                np.savez_compressed(root / "private/answers.npz", y=y)
            else:
                np.savez_compressed(folder / f"{name}.npz", x=x, y=y)


def score():
    predictions = {}
    for number in (4, 5):
        root = ROOT / f"synthetic_{number}"
        for method in ("dense", *METHODS):
            p = (
                root
                / "targets"
                / f"case_{number:02d}_c0"
                / method
                / "frozen_predictions.npz"
            )
            predictions[str(p)] = digest(p)
    save_json(
        OUTPUT / "synthetic_predictions_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "predictions_sha256": predictions,
            "query_answers_opened": False,
        },
    )
    rows = []
    arrays = {}
    for number in (4, 5):
        root = ROOT / f"synthetic_{number}"
        truth = np.load(root / "private/answers.npz")["y"]
        arrays[f"case{number}_truth"] = truth
        for method in ("dense", *METHODS):
            p = (
                root
                / "targets"
                / f"case_{number:02d}_c0"
                / method
                / "frozen_predictions.npz"
            )
            value = np.load(p)["prediction_256"]
            mse = ((value - truth) ** 2).mean((1, 2))
            rows.append(
                {
                    "geometry": number,
                    "method": method,
                    "mse": float(mse.mean()),
                    "per_function_mse": mse.tolist(),
                }
            )
            arrays[f"case{number}_{method}"] = value
    save_json(
        OUTPUT / "synthetic_results.json",
        {"rows": rows, "scored_utc": datetime.now(UTC).isoformat()},
    )
    np.savez_compressed(OUTPUT / "synthetic_scoring.npz", **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("prepare", "native", "source", "target_prepare", "score"),
        required=True,
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "source":
        torch.set_num_threads(2)
        source(args.root.resolve(), torch.device(args.device))
    else:
        globals()[args.phase]()
