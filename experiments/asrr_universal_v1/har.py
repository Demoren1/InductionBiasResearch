"""Use the shared aSRR estimator with the article's HAR refit protocol."""

import argparse
import json
import shutil

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_transfer_refinement_v1.common import (
    OUT,
    SCREEN_EPISODES,
    SCREEN_RATES,
    SCREEN_STEPS,
    SEED,
)
from experiments.asrr_transfer_refinement_v1.study import (
    ordering,
    validation_rows,
)
from experiments.asrr_transfer_refinement_v1.unified_global import (
    HAR_LAYOUT,
    har_test,
)
from experiments.asrr_tuning_v1.data import episodes
from experiments.asrr_tuning_v1.fit import run as fit_har_development
from experiments.paper_transfer_v2.common import METHODS, freeze
from experiments.paper_transfer_v2.har import ORIGINAL_ROOT, REFERENCE, ROOT
from experiments.paper_transfer_v2.report import confidence, grouped
from experiments.paper_transfer_v2.target import fit_batch, metric
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .run import STAGE, model_bank

SHOTS = 3
RANK = 8


def har_source(root, device):
    support = read_data(root / "public" / "support.npz")
    query = read_data(root / "public" / "structure.npz")
    device = torch.device(device)
    sx = torch.as_tensor(support["x"][:, :, None], device=device) / 255
    qx = torch.as_tensor(query["x"][:, :, None], device=device) / 255
    sy = torch.as_tensor(support["y"].argmax(-1), device=device)
    qy = torch.as_tensor(query["y"].argmax(-1), device=device)
    order = sy.argsort(stable=True).reshape(len(sx), HAR_LAYOUT.outputs, -1)
    ordered = sx[torch.arange(len(sx), device=device)[:, None, None], order]

    dense = read_data(root / "source" / "dense" / "models.npz")
    ce = dense["numerical_cross_entropy"]
    state = dense["ensemble_parameters"][
        np.arange(len(ce))[:, None], ce.argsort()
    ]
    profiles = source_profiles(state[np.argsort(ce.min(1))], HAR_LAYOUT)
    return profiles, (sx, sy, {"query": qx}, qy), ordered


def screen(bank, panel, device):
    grid = np.linspace(
        0, len(panel["train_x"]), SCREEN_EPISODES, endpoint=False, dtype=int
    )
    sx, qx = [
        torch.as_tensor(panel[key][grid, :, None], device=device)
        for key in ("train_x", "query_x")
    ]
    sy = torch.as_tensor(panel["train_y"][grid].argmax(-1), device=device)
    qy = panel["query_y"][grid].argmax(-1)
    results = {}
    for name, row in bank.items():
        factors = torch.as_tensor(
            read_data(row["path"])["factor"], device=device
        )
        for rate in SCREEN_RATES:
            predictions = fit_batch(
                sx,
                sy,
                {"query": qx},
                HAR_LAYOUT,
                {"factor": factors},
                rate,
                SCREEN_STEPS,
                [(len(sx), SEED)],
            )
            for step, values in predictions.items():
                results[name, rate, step] = {
                    "query": np.column_stack(metric(values["query"], qy))
                }
    rows = validation_rows(results, None)
    shortlisted = []
    for family in sorted({row["family"] for row in bank.values()}):
        ordered = [row["candidate"] for row in sorted(rows, key=ordering)]
        shortlisted.extend(
            list(
                dict.fromkeys(
                    name for name in ordered if bank[name]["family"] == family
                )
            )[:2]
        )
    return shortlisted, rows


def layer_basis(factors):
    hidden = HAR_LAYOUT.hidden
    basis = np.zeros((HAR_LAYOUT.fan_in, hidden, hidden * RANK), np.float32)
    width = hidden // len(factors)
    for h in range(hidden):
        basis[:, h, h * RANK : (h + 1) * RANK] = factors[h // width]
    return basis


def development(folder, device, *, build=model_bank):
    manifest = folder / "bank.json"
    if manifest.exists():
        bank = json.loads(manifest.read_text())
    else:
        source = ROOT.parent.parent / "paper_asrr_2026-11-17" / "har"
        profiles, data, ordered = har_source(source, device)
        bank = build(folder, profiles, data, ordered, HAR_LAYOUT, RANK, SHOTS)
        print(f"HAR built {len(bank)} candidates", flush=True)

    work = folder / "work"
    models = work / "banks" / "har" / "models"
    models.mkdir(parents=True, exist_ok=True)
    source_panel = ORIGINAL_ROOT / "study" / "har3" / "panels" / "d2.npz"
    panel = work / "study" / "har3" / "panels" / "d2.npz"
    panel.parent.mkdir(parents=True, exist_ok=True)
    if not panel.exists():
        shutil.copyfile(source_panel, panel)
    if folder.name == STAGE:
        candidates, rows = screen(bank, read_data(panel), device)
        save_json(folder / "screen.json", rows)
    else:
        candidates = list(bank)

    policies = []
    for name in candidates:
        path = models / f"{name}.npz"
        factors = read_data(bank[name]["path"])["factor"]
        if not path.exists():
            atomic_npz(path, basis=layer_basis(factors))
        result = (
            work / "study" / "har3" / "fits" / "d2" / f"{name}__screen.json"
        )
        if not result.exists():
            fit_har_development(
                {
                    "name": name,
                    "setting": "har3",
                    "phase": "d2",
                    "model_sha256": digest(path),
                    "panel_sha256": digest(panel),
                },
                device,
                root=work,
            )
        fit = json.loads(result.read_text())
        best = min(
            fit["metrics"],
            key=lambda k: (
                -fit["metrics"][k]["accuracy"],
                fit["metrics"][k]["ce"],
            ),
        )
        policies.append(
            {
                "model": str(path),
                "sha256": digest(path),
                "candidate": name,
                "rule": fit["rules"][best],
                **fit["metrics"][best],
            }
        )
    best = min(
        (
            row
            for row in policies
            if bank[row["candidate"]]["group_weight"] > 0
        ),
        key=lambda row: (-row["accuracy"], row["ce"], row["candidate"]),
    )
    freeze(
        folder / "choices.json", {"selected": best, "development": policies}
    )
    print("HAR frozen", best, flush=True)


def confirmation(folder, device, replicate):
    decision = json.loads((folder / "choices.json").read_text())
    chosen = decision["selected"]
    reference = json.loads((REFERENCE / "choices_frozen.json").read_text())
    policies = {"universal": chosen}
    methods = (
        ("asrr", *METHODS)
        if folder.name != STAGE
        else ("asrr", "yeh_soft", "msr")
    )
    for name in methods:
        model = reference[name]["name"]
        path = ROOT / "banks" / "har" / "models" / f"{model}.npz"
        policies[name] = {
            "model": str(path) if name != "dense" else None,
            "sha256": reference[name].get("model_sha256"),
            "rule": reference[name]["rule"],
        }
    output = folder / f"confirmation_{replicate}"
    output.mkdir(exist_ok=True)
    if folder.name != STAGE:
        from .adaptive import combine
        from .fixed_blocks import STAGE as FIXED_STAGE
        from .penalized import STAGE as SPECTRAL_STAGE, shuffled_bases
        from .structure import spectral_basis

        bank = json.loads((folder / "bank.json").read_text())
        control = bank[chosen["candidate"]].get("guide", "mix_0")
        pure = next(
            row
            for row in decision["development"]
            if row["candidate"] == control
        )
        policies["guide"] = pure
        groups = bank[chosen["candidate"]].get("groups", "mix_1")
        layout = read_data(bank[groups]["path"])["factor"]
        guide = read_data(bank[control]["path"])["factor"][0]
        strength = bank[chosen["candidate"]]["group_weight"]
        for i in range(4):
            path = output / f"shuffle_{i}.npz"
            if not path.exists():
                rng = np.random.default_rng(SEED + 621173 + i)
                if folder.name == SPECTRAL_STAGE:
                    moment = bank[chosen["candidate"]]["moment_weight"]
                    factors = shuffled_bases(
                        folder / "operators.npz", moment, strength, rng
                    )
                else:
                    mix = (
                        spectral_basis
                        if folder.name == FIXED_STAGE
                        else combine
                    )
                    factors = [
                        mix(g[rng.permutation(len(g))], guide, strength)
                        for g in layout
                    ]
                atomic_npz(path, basis=layer_basis(factors))
            policies[f"shuffle_{i}"] = {
                **chosen,
                "model": str(path),
                "sha256": digest(path),
            }
    seed = SEED + 870241 + replicate * 25783
    freeze(output / "choices.json", {"seed": seed, "policies": policies})
    observations, provenance = episodes("har3", "test", 360, seed)
    subjects = np.asarray([row["person"] for row in provenance])
    truth = observations["query_y"].argmax(-1)
    inputs = [observations[key] for key in ("train_x", "train_y", "query_x")]
    scores = {}
    for name, policy in policies.items():
        if (
            policy["model"] is not None
            and digest(policy["model"]) != policy["sha256"]
        ):
            raise ValueError(f"Changed frozen HAR model for {name}")
        predictions = har_test(inputs, policy["model"], policy["rule"], device)
        scores[name] = (predictions == truth).mean(-1)
    shuffled = [
        value for name, value in scores.items() if name.startswith("shuffle_")
    ]
    if shuffled:
        scores["shuffled"] = np.mean(shuffled, axis=0)
    report = {
        "metrics": {
            name: confidence(grouped(value, subjects))
            for name, value in scores.items()
        },
        "universal_minus_method": {
            name: confidence(grouped(scores["universal"] - value, subjects))
            for name, value in scores.items()
            if name != "universal"
        },
        "episodes": len(subjects),
        "units": len(np.unique(subjects)),
    }
    save_json(output / "results.json", report)
    atomic_npz(output / "per_episode.npz", units=subjects, **scores)
    print("HAR confirmation", report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--confirm", type=int)
    parser.add_argument("--stage", default=STAGE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    directory = OUT / "har" / args.stage
    directory.mkdir(parents=True, exist_ok=True)
    if args.confirm is None:
        development(directory, args.device)
    else:
        confirmation(directory, args.device, args.confirm)
