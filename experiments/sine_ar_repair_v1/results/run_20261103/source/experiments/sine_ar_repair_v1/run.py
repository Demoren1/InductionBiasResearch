"""Fixed-bank prior ablations, query calibration, and fresh-function evaluation."""

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.sine_ar_current_v1.run import fit, tasks, teacher
from experiments.sine_ar_repair_v1.models import Prior, configurations

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PLAN = json.loads((BASE / "protocol.json").read_text())
SOURCE = ROOT / PLAN["source_run"]
OUTPUT = ROOT / PLAN["output"]
FIT_PLAN = json.loads((SOURCE / "protocol.json").read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def initialize(device):
    OUTPUT.mkdir(exist_ok=False, parents=True)
    snapshot = OUTPUT / "source"
    snapshot.mkdir()
    paths = list(BASE.glob("*.py")) + [
        BASE / "protocol.json",
        ROOT / "experiments/sine_ar_current_v1/run.py",
        ROOT / "pattern/autoregressive/model.py",
        ROOT / "archive/history/experiments/sine_mask_vae/models.py",
        ROOT / "archive/history/experiments/sine_mask_vae/core.py",
    ]
    hashes = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    with np.load(SOURCE / "bank.npz") as saved:
        bank = dict(saved)
    specs = json.loads((SOURCE / "source_tasks.json").read_text())
    source_plan = json.loads((SOURCE / "protocol.json").read_text())
    maximum = max(np.bincount(bank["task_ids"]))
    masks = np.zeros((16, maximum, 8, 2), dtype=np.float32)
    features = np.zeros((*masks.shape, 2), dtype=np.float32)
    present = np.zeros((16, maximum), dtype=bool)
    validation = np.zeros_like(present)
    agreement = []
    for index, spec in enumerate(specs):
        source_path = SOURCE / "source" / f"task_{index:02d}.npz"
        hashes[str(source_path.relative_to(ROOT))] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        with np.load(source_path) as saved:
            selected = saved["query"] < 0.1
            mask = saved["masks"][selected].reshape(-1, 8, 2)
            count = len(mask)
            masks[index, :count] = mask
            present[index, :count] = True
            validation[index, :count] = saved["heldout"][selected]
            weights = torch.tensor(
                saved["elite_weights"], device=device, requires_grad=True
            )
            amplitude = torch.tensor(saved["elite_amplitudes"], device=device)
            seed = source_plan["seed"] + index * 1009
            rng = torch.Generator(device=device).manual_seed(seed + 800001)
            low, high, small = teacher([spec], device)
            query = torch.zeros(count, device=device)
            gradients = torch.zeros_like(weights)
            for _ in range(16):
                x = torch.randn(256, 8, device=device, generator=rng)
                target = small[0] * torch.sin(x @ low[0]) + torch.sin(
                    x @ high[0]
                )
                prediction = amplitude[:, None] * torch.sin(
                    torch.einsum("bi,nij->nbj", x, weights)
                )
                loss = (prediction - target).square().mean((1, 2))
                query += loss.detach() / 16
                gradients += torch.autograd.grad(loss.sum(), weights)[0] / 16
            agreement.append(
                float(
                    np.max(
                        np.abs(query.cpu().numpy() - saved["query"][selected])
                    )
                )
            )
            values = np.stack(
                [
                    weights.detach().cpu().numpy(),
                    gradients.cpu().numpy() * mask,
                ],
                axis=-1,
            )
            scale = np.abs(values).max((1, 2), keepdims=True).clip(1e-12)
            features[index, :count] = values / scale
    np.savez_compressed(
        OUTPUT / "bank.npz",
        masks=masks,
        features=features,
        present=present,
        validation=validation,
    )
    dump(OUTPUT / "configurations.json", configurations())
    dump(OUTPUT / "protocol.json", PLAN)
    dump(
        OUTPUT / "provenance.json",
        {
            "source_sha256": hashes,
            "source_query_max_difference": agreement,
            "cards": int(present.sum()),
            "train_cards": int((present & ~validation).sum()),
            "validation_cards": int(validation.sum()),
            "feature_gradient": "mean empirical MSE gradient on original 4096 query examples",
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device),
        },
    )
    print(
        json.dumps(
            {"initialized": str(OUTPUT), "query_difference": max(agreement)}
        ),
        flush=True,
    )


def training_data(config, device):
    with np.load(OUTPUT / "bank.npz") as bank:
        masks = torch.tensor(bank["masks"], device=device)
        features = torch.tensor(bank["features"], device=device)
        if "mask_vae" in config["family"]:
            features = torch.stack([masks, torch.zeros_like(masks)], -1)
        weights = []
        for selected in [
            bank["present"] & ~bank["validation"],
            bank["validation"],
        ]:
            value = selected.astype(np.float32)
            if config["task_balanced"]:
                value /= value.sum(1, keepdims=True).clip(1)
            value /= value.sum()
            weights.append(torch.tensor(value, device=device))
    return masks, features, weights


def train(config, device):
    folder = OUTPUT / config["id"]
    folder.mkdir()
    masks, features, weights = training_data(config, device)
    for seed_index, seed in enumerate(PLAN["prior_seeds"]):
        torch.manual_seed(seed)
        model = Prior(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["decay"]
        )
        history = []
        started = time.perf_counter()
        for step in range(1, max(PLAN["checkpoints"]) + 1):
            loss = model.loss(masks, features, weights[0], step)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if "vae" in config["family"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 100 == 0:
                with torch.no_grad():
                    record = {
                        "step": step,
                        "train_objective": float(loss),
                        "train_clean": float(
                            model.loss(masks, features, weights[0], step, True)
                        ),
                        "validation_clean": float(
                            model.loss(masks, features, weights[1], step, True)
                        ),
                    }
                history.append(record)
            if step in PLAN["checkpoints"]:
                torch.save(
                    model.state_dict(),
                    folder / f"seed{seed_index}_step{step}.pt",
                )
        dump(folder / f"history_{seed_index}.json", history)
        print(
            json.dumps(
                {
                    "trained": config["id"],
                    "family": config["family"],
                    "seed": seed,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "seconds": time.perf_counter() - started,
                    "final": history[-1],
                }
            ),
            flush=True,
        )
    dump(folder / "TRAIN_COMPLETE.json", {"complete": True})


def load_models(setting, device):
    config = next(c for c in configurations() if c["id"] == setting["model"])
    values = []
    for index in range(3):
        model = Prior(config).to(device)
        path = OUTPUT / config["id"] / f"seed{index}_step{setting['step']}.pt"
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        values.append(model.eval().requires_grad_(False))
    return values


def source_masks(device):
    with np.load(OUTPUT / "bank.npz") as bank:
        selected = bank["present"] & ~bank["validation"]
        return torch.tensor(
            bank["masks"][selected].reshape(-1, 16),
            dtype=torch.long,
            device=device,
        )


def calibration_tasks(functions):
    familiar = tasks(functions[:1], [0, 1])
    novel = tasks(functions, [2])
    return familiar + novel


def query_scores(result):
    query = result["selected_query"]
    return {
        "mean_query": float(query.mean()),
        "familiar_query": float(query[:8].mean()),
        "novel_query": float(query[8:].mean()),
        "seed_query": [float(v.mean()) for v in np.split(query, 3, axis=1)],
    }


def solve(
    models,
    setting,
    specs,
    searches,
    seed,
    device,
    policy=None,
    keep_history=False,
):
    policy = policy or {
        "proposals": 32,
        "repeats": 1,
        "top": 1,
        "reuse": 4,
        "steps": 1024,
        "lr": 0.05,
        "scale": 2.0,
    }
    plan = copy.deepcopy(FIT_PLAN)
    plan["steps"] = policy["steps"]
    bank = source_masks(device)
    proposed = []
    count = len(specs) * searches * policy["proposals"]
    for index, model in enumerate(models):
        torch.manual_seed(seed + 11 + index * 1000)
        mask = model.sample(
            count, setting["scale"], bank, setting.get("sampling_noise", False)
        )
        proposed.append(
            mask.cpu()
            .numpy()
            .reshape(len(specs), searches, policy["proposals"], 8, 2)
        )
    masks = np.concatenate(proposed, axis=1)
    masks = np.repeat(masks, policy["repeats"], axis=2)
    config = {"optimizer": "sgd", "lr": policy["lr"], "scale": policy["scale"]}
    first, history = fit(
        plan, specs, masks, [config], seed, device, reuse=policy["reuse"]
    )
    refit_count = 65536 // policy["steps"] - masks.shape[2]
    selected_masks = []
    for t in range(len(specs)):
        task_masks = []
        for s in range(searches * 3):
            order = first["query"][t, s].argsort(kind="stable")
            choices, seen = [], set()
            for candidate in order:
                mask = masks[t, s, candidate]
                key = mask.tobytes()
                if key not in seen:
                    choices.append(mask)
                    seen.add(key)
                    if len(choices) == policy["top"]:
                        break
            task_masks.append(
                np.stack(
                    [choices[i % len(choices)] for i in range(refit_count)]
                )
            )
        selected_masks.append(np.stack(task_masks))
    second_masks = np.stack(selected_masks)
    second, history2 = fit(
        plan, specs, second_masks, [config], seed, device, phase=1
    )
    result = {
        k: np.concatenate([first[k], second[k]], axis=2)
        for k in first
        if k != "seconds"
    }
    result["seconds"] = first["seconds"] + second["seconds"]
    result["exploration_fits"] = np.asarray(masks.shape[2])
    selected = result["query"].argmin(2)
    ti, si = np.indices(selected.shape)
    result["selected"] = selected
    result["selected_query"] = result["query"][ti, si, selected]
    result["selected_population"] = result["population"][ti, si, selected]
    if keep_history:
        for item, later in zip(history, history2):
            for metric in ["train", "population"]:
                item[metric] = np.concatenate(
                    [item[metric], later[metric]], axis=2
                ).tolist()
    return result, history if keep_history else None


def screen(config, device):
    folder = OUTPUT / config["id"]
    if not (folder / "TRAIN_COMPLETE.json").exists():
        raise RuntimeError(f"Training incomplete: {config['id']}")
    specs = calibration_tasks(PLAN["screen_functions"])
    saved = folder / "screen_candidates"
    saved.mkdir()
    scales = (
        PLAN["vae_latent_scales"]
        if "vae" in config["family"]
        else PLAN["ar_temperatures"]
    )
    noise = [False, True] if "noise" in config["family"] else [False]
    rows = []
    for step in PLAN["checkpoints"]:
        setting = {
            "model": config["id"],
            "family": config["family"],
            "step": step,
        }
        models = load_models(setting, device)
        for scale in scales:
            for corrupt in noise:
                candidate = {
                    **setting,
                    "scale": scale,
                    "sampling_noise": corrupt,
                }
                result, _ = solve(
                    models,
                    candidate,
                    specs,
                    PLAN["screen_searches_per_seed"],
                    PLAN["screen_seed"],
                    device,
                )
                row = {**candidate, **query_scores(result)}
                rows.append(row)
                ti, si = np.indices(result["selected"].shape)
                chosen = result["selected"]
                np.savez_compressed(
                    saved / f"step{step}_scale{scale}_noise{int(corrupt)}.npz",
                    query=result["query"],
                    masks=result["masks"].astype(np.uint8),
                    selected=chosen,
                    selected_query=result["selected_query"],
                    selected_population=result["selected_population"],
                    selected_weights=result["weights"][ti, si, chosen],
                    selected_amplitude=result["amplitude"][ti, si, chosen],
                )
        print(
            json.dumps(
                {
                    "screened": config["id"],
                    "step": step,
                    "best": min(rows, key=lambda x: x["mean_query"]),
                }
            ),
            flush=True,
        )
        dump(folder / "screen.json", rows)
    dump(folder / "SCREEN_COMPLETE.json", {"complete": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True, choices=["initialize", "train", "screen"]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "initialize":
        initialize(args.device)
    else:
        for index, config in enumerate(configurations()):
            if index % args.shards == args.shard:
                {"train": train, "screen": screen}[args.phase](
                    config, args.device
                )


if __name__ == "__main__":
    main()
