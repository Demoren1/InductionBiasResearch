"""Apply group-anchored completion to the paper's two global-sharing settings."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz, panel
from experiments.asrr_benchmark_families_v1.groups import signed_bank
from experiments.asrr_tuning_v1.data import episodes
from experiments.asrr_tuning_v1.fit import run as fit_har_development
from experiments.paper_recovery_extension_v1.adapter import fit_adam, values
from experiments.paper_transfer_v2 import linear
from experiments.paper_transfer_v2.common import SEED as PAPER_SEED
from experiments.paper_transfer_v2.har import ORIGINAL_ROOT, REFERENCE, ROOT
from experiments.paper_transfer_v2.report import confidence, grouped
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .common import OUT, SEED
from .contrasts import discriminant_basis, means_and_noise
from .fusion import fuse
from .reliable import covariance_differences, symmetric
from .unified import MOMENT_WEIGHT, RIDGE, span

HAR_LAYOUT = Layout(1, 561, 16, 6)


def har_guides():
    public = Path("artifacts/local/paper_asrr_2026-11-17/har/public")
    parts = [
        read_data(public / f"{name}.npz") for name in ("support", "structure")
    ]
    data = [(p["x"][:, :, None, :] / 255, p["y"].argmax(-1)) for p in parts]
    (a, ua), (b, ub) = [
        means_and_noise(x, y, HAR_LAYOUT.outputs) for x, y in data
    ]
    work = np.einsum("tcsi,tcsj->sij", a, b, optimize=True) / (
        a.shape[0] * a.shape[1]
    )
    mean = symmetric(work)[0]
    covariances = [
        covariance_differences(
            torch.from_numpy(x), torch.from_numpy(y), HAR_LAYOUT
        )
        for x, y in data
    ]
    moment = symmetric((covariances[0] @ covariances[1]).mean((0, 1)))[
        0
    ].numpy()
    mixed = (1 - MOMENT_WEIGHT) * mean / max(np.linalg.norm(mean), 1e-12)
    mixed += MOMENT_WEIGHT * moment / max(np.linalg.norm(moment), 1e-12)
    noise = (ua[0] + ub[0]) / 2
    empty = np.empty((HAR_LAYOUT.fan_in, 0))
    for name, signal in (("means", mean), ("moments", mixed)):
        directions = discriminant_basis(signal, noise, 8, RIDGE, empty)
        yield name, np.kron(directions, np.eye(HAR_LAYOUT.hidden))


def prepare_har(folder, model_dir):
    archive = ROOT / "banks" / "har" / "models"
    for method in ("asrr", "yeh_soft", "msr"):
        path = model_dir / f"{method}.npz"
        if not path.exists():
            shutil.copyfile(archive / path.name, path)
    config = json.loads((REFERENCE / "choices_frozen.json").read_text())
    candidates = {"asrr": config["asrr"]}
    group = span(
        linear.signed_design(
            read_data(model_dir / "asrr.npz")["index"].ravel()
        )
    )
    panel_path = folder / "study" / "har3" / "panels" / "d2.npz"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    if not panel_path.exists():
        shutil.copyfile(
            ORIGINAL_ROOT / "study" / "har3" / "panels" / "d2.npz", panel_path
        )
    for name, guide in har_guides():
        tag = f"projected_{name}"
        model_path = model_dir / f"{tag}.npz"
        basis = fuse(group, guide, guide, 0.0, 1).reshape(
            HAR_LAYOUT.fan_in, HAR_LAYOUT.hidden, -1
        )
        atomic_npz(model_path, basis=basis.astype(np.float32))
        candidates[tag] = {
            "name": tag,
            "model_sha256": digest(model_path),
            "panel_sha256": digest(panel_path),
        }
    return candidates, config


def har_test(inputs, path, rule, device):
    model = read_data(path) if path is not None else {}
    structure = None
    if "index" in model:
        index = model["index"]
        structure = {
            "index": torch.as_tensor(index, device=device, dtype=torch.long),
            "codes": int(np.abs(index).max()),
            "scale": float(model.get("scale", 1.0)),
        }
    elif "basis" in model:
        structure = torch.as_tensor(model["basis"], device=device)
    settings = [
        {**rule["global_setting"], "restart": i}
        for i in range(rule["numeric_starts"])
    ]
    result = []
    for x, y, qx in zip(
        *(np.array_split(row, 2) for row in inputs), strict=True
    ):
        x, y, qx = [
            torch.as_tensor(row, device=device, dtype=torch.float32)
            for row in (x, y, qx)
        ]
        parameter, _ = fit_adam(
            x,
            y,
            None,
            None,
            structure,
            HAR_LAYOUT.hidden,
            HAR_LAYOUT.outputs,
            fixed_step=rule["steps"],
            settings=settings,
            keep=len(settings),
            preserve_setting_order=True,
            loss="cross_entropy",
            seed=7611173301,
        )
        with torch.no_grad():
            prob = values(
                qx, structure, parameter, HAR_LAYOUT.hidden, HAR_LAYOUT.outputs
            )
            prediction = (
                prob[: rule["ensemble"]].softmax(-1).mean(0).argmax(-1)
            )
            result.append(prediction.cpu().numpy())
    return np.concatenate(result)


def har(device):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    folder = OUT / "har" / "unified"
    work = folder / "work"
    models = work / "banks" / "har" / "models"
    models.mkdir(parents=True, exist_ok=True)
    candidates, reference = prepare_har(work, models)
    for name, row in candidates.items():
        if name == "asrr":
            continue
        path = work / "study" / "har3" / "fits" / "d2" / f"{name}__screen.json"
        if not path.exists():
            fit_har_development(
                {**row, "setting": "har3", "phase": "d2"}, device, root=work
            )
        info = json.loads(path.read_text())
        key = min(
            info["metrics"],
            key=lambda k: (
                -info["metrics"][k]["accuracy"],
                info["metrics"][k]["ce"],
            ),
        )
        row.update(info["metrics"][key], rule=info["rules"][key])

    choice = min(
        candidates.values(), key=lambda row: (-row["accuracy"], row["ce"])
    )
    ref = {k: reference[k] for k in ("asrr", "yeh_soft", "msr")}
    policies = {"refined": choice, **ref}
    seed = SEED + 870241
    save_json(
        folder / "choices.json",
        {"policies": policies, "candidates": candidates, "panel_seed": seed},
    )
    observations, provenance = episodes("har3", "test", 360, seed)
    subjects = np.asarray([row["person"] for row in provenance])
    inputs = [observations[key] for key in ("train_x", "train_y", "query_x")]
    results = {}
    for key, row in policies.items():
        model = models / f"{row['name']}.npz"
        if digest(model) != row["model_sha256"]:
            raise ValueError(f"Changed frozen HAR structure {model}")
        if key == "asrr" and row["name"] == choice["name"]:
            results[key] = results["refined"]
        else:
            results[key] = har_test(inputs, model, row["rule"], device)
    truth = observations["query_y"].argmax(-1)
    acc = {key: (values == truth).mean(-1) for key, values in results.items()}
    report = {
        "metrics": {
            key: confidence(grouped(a, subjects)) for key, a in acc.items()
        },
        "refined_minus_method": {
            key: confidence(grouped(acc["refined"] - a, subjects))
            for key, a in acc.items()
            if key != "refined"
        },
        "episodes": len(subjects),
        "units": len(np.unique(subjects)),
    }
    save_json(folder / "results.json", report)
    atomic_npz(folder / "per_episode.npz", units=subjects, **acc)
    print("har", choice["name"], report, flush=True)


def synthetic():
    folder = OUT / "synthetic" / "unified"
    folder.mkdir(parents=True, exist_ok=True)
    cfg = linear.CFG
    outcomes = []
    for geometry in range(cfg["geometries"]):
        seed = PAPER_SEED + 2137 * geometry
        teacher = linear.teacher(np.random.default_rng(seed))
        source_x, source_y = panel(
            teacher,
            cfg["source_tasks"],
            cfg["source_examples"],
            seed + 1,
            cfg["source_noise"],
        )
        weight, _ = linear.source_regression(source_x, source_y)
        profiles = np.zeros((len(weight), *teacher.shape))
        profiles[:, linear.WINDOW, linear.OUTPUT] = weight
        index = signed_bank(
            profiles, teacher != 0, (linear.CODES,), partial=False
        )[linear.CODES]
        ward = linear.signed_design(
            index[linear.WINDOW, linear.OUTPUT].ravel()
        )
        left, right = [
            linear.source_regression(x, y)[0].reshape(len(weight), -1)
            for x, y in zip(
                np.split(source_x, 2, axis=1),
                np.split(source_y, 2, axis=1),
                strict=True,
            )
        ]
        signal = symmetric(left.T @ right) / len(weight)
        guide = eigh(
            signal,
            subset_by_index=(len(signal) - linear.CODES, len(signal) - 1),
        )[1]
        hybrid = fuse(span(ward), guide, guide, 0.0, 1)
        variants = {"ward": ward, "projected": hybrid}
        spans = max(cfg["shots"]) + cfg["query_examples"]
        dev_x, dev_y = panel(
            teacher, cfg["development_tasks"], spans, seed + 2, 0
        )
        choice, validation = linear.select(dev_x, dev_y, variants)
        save_json(
            folder / f"geometry_{geometry:02d}_choices.json",
            {"choice": choice, "validation": validation},
        )
        test_x, test_y = panel(teacher, cfg["test_tasks"], spans, seed + 3, 0)
        means = {}
        for shot, row in choice.items():
            prediction = next(
                linear.ridge_predictions(
                    test_x,
                    test_y,
                    int(shot),
                    variants[row["candidate"]],
                    [row["ridge"]],
                )
            )
            means[shot] = float(
                np.square(prediction - test_y[:, max(cfg["shots"]) :]).mean()
            )
        outcomes.append({"choice": choice, "mse": means})
    save_json(folder / "results.json", outcomes)
    print("synthetic", [row["mse"] for row in outcomes], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("har", "synthetic"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.dataset == "har":
        har(torch.device(args.device))
    else:
        synthetic()
