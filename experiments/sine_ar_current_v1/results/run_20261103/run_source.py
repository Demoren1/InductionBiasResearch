"""Random source solutions, current MADE, and fresh SGD on sine tasks."""

import argparse
import copy
import gzip
import hashlib
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pattern.autoregressive.model import IndexMADE

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "results/run_20261103"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def tasks(functions, remainders):
    result = []
    for function, a, b in itertools.product(functions, range(4), range(4)):
        if (b - a) % 4 not in remainders:
            continue
        rng = np.random.default_rng(731001 + function * 97 + a * 11 + b)
        result.append({
            "function": function, "a": a, "b": b, "rows": [a, b + 4],
            "low": rng.uniform(0.65, 0.75, 2).tolist(),
            "high": rng.uniform(3.8, 4.2, 2).tolist(),
            "small": rng.uniform(0.35, 0.45, 2).tolist(),
        })
    return result


def teacher(specs, device):
    low = torch.zeros(len(specs), 8, 2, device=device)
    high = torch.zeros_like(low)
    for t, spec in enumerate(specs):
        low[t, spec["rows"], [0, 1]] = torch.tensor(spec["low"], device=device)
        high[t, spec["rows"], [0, 1]] = torch.tensor(spec["high"], device=device)
    small = torch.tensor([spec["small"] for spec in specs], device=device)
    return low, high, small


def population(weights, amplitude, spec):
    low, high, small = spec
    low, high, small = low[:, None], high[:, None], small[:, None]

    def kernel(left, right):
        return 0.5 * (
            (-(left - right).square().sum(-2) / 2).exp()
            - (-(left + right).square().sum(-2) / 2).exp()
        )

    target = (small.square() * kernel(low, low) + kernel(high, high)
              + 2 * small * kernel(low, high))
    return (amplitude.square() * kernel(weights, weights)
            - 2 * amplitude * (small * kernel(weights, low)
                               + kernel(weights, high)) + target).mean(-1)


def fit(plan, specs, masks, configs, seed, device, reuse=0, phase=0):
    """Vectorized independent models; real minibatches and autograd MSE."""
    started = time.perf_counter()
    masks = torch.as_tensor(masks, dtype=torch.float32, device=device)
    task_count, searches, starts, _, _ = masks.shape
    count = searches * starts
    masks = masks.reshape(task_count, count, 8, 2)
    generator = torch.Generator(device=device).manual_seed(seed + phase * 100003)
    data = torch.Generator(device=device).manual_seed(seed + 700001 + phase * 100003)
    query_rng = torch.Generator(device=device).manual_seed(seed + 800001)
    scales = torch.tensor([c["scale"] for c in configs], device=device)
    rates = torch.tensor([c["lr"] for c in configs], device=device)
    if len(configs) == 1:
        scales = scales.expand(count)
        rates = rates.expand(count)
    else:
        scales = scales.repeat_interleave(starts)
        rates = rates.repeat_interleave(starts)
    draw_starts = reuse or starts
    initial = torch.randn(task_count, searches, draw_starts, 8, 2,
                          generator=generator, device=device)
    if reuse:
        initial = initial.repeat(1, 1, starts // reuse, 1, 1)
    weights = (initial.reshape(task_count, count, 8, 2)
               * scales[None, :, None, None] * masks).requires_grad_()
    amplitude = torch.ones(task_count, count, 2, device=device, requires_grad=True)
    params = [weights, amplitude]
    is_adam = configs[0]["optimizer"] == "adam"
    if is_adam:
        first = [torch.zeros_like(p) for p in params]
        second = [torch.zeros_like(p) for p in params]
    spec = teacher(specs, device)
    low, high, small = spec
    history = []
    for step in range(plan["steps"] + 1):
        x = torch.randn(task_count, plan["batch"], 8, generator=data, device=device)
        y = (small[:, None] * torch.sin(torch.einsum("tbi,tij->tbj", x, low))
             + torch.sin(torch.einsum("tbi,tij->tbj", x, high)))
        prediction = amplitude[:, :, None] * torch.sin(
            torch.einsum("tbi,tnij->tnbj", x, weights)
        )
        losses = (prediction - y[:, None]).square().mean((-2, -1))
        if step % 128 == 0 or step == plan["steps"]:
            with torch.no_grad():
                history.append({
                    "step": step,
                    "train": losses.reshape(task_count, searches, starts).cpu().tolist(),
                    "population": population(weights, amplitude, spec).reshape(
                        task_count, searches, starts
                    ).cpu().tolist(),
                })
        if step == plan["steps"]:
            break
        gradients = torch.autograd.grad(losses.sum(), params)
        with torch.no_grad():
            for index, (parameter, gradient) in enumerate(zip(params, gradients)):
                rate = rates.reshape(1, count, *([1] * (parameter.ndim - 2)))
                if is_adam:
                    first[index].mul_(0.9).add_(gradient, alpha=0.1)
                    second[index].mul_(0.999).addcmul_(gradient, gradient, value=0.001)
                    m = first[index] / (1 - 0.9 ** (step + 1))
                    v = second[index] / (1 - 0.999 ** (step + 1))
                    parameter.add_(-rate * m / (v.sqrt() + 1e-8))
                else:
                    parameter.add_(-rate * gradient)
            weights.mul_(masks)
    with torch.no_grad():
        query = torch.zeros(task_count, count, device=device)
        for _ in range(plan["query"] // 256):
            x = torch.randn(task_count, 256, 8, generator=query_rng, device=device)
            y = (small[:, None] * torch.sin(torch.einsum("tbi,tij->tbj", x, low))
                 + torch.sin(torch.einsum("tbi,tij->tbj", x, high)))
            prediction = amplitude[:, :, None] * torch.sin(
                torch.einsum("tbi,tnij->tnbj", x, weights)
            )
            query += (prediction - y[:, None]).square().mean((-2, -1))
        query /= plan["query"] // 256
        result = {
            "query": query.reshape(task_count, searches, starts).cpu().numpy(),
            "population": population(weights, amplitude, spec).reshape(
                task_count, searches, starts
            ).cpu().numpy(),
            "weights": weights.reshape(task_count, searches, starts, 8, 2).cpu().numpy(),
            "amplitude": amplitude.reshape(task_count, searches, starts, 2).cpu().numpy(),
            "masks": masks.reshape(task_count, searches, starts, 8, 2).cpu().numpy(),
        }
    torch.cuda.synchronize(device)
    result["seconds"] = time.perf_counter() - started
    return result, history


def save_fit(path, result, history):
    np.savez_compressed(path.with_suffix(".npz"), **result)
    with gzip.open(path.with_name(path.name + "_history.json.gz"), "wt") as stream:
        json.dump(history, stream)


def random_masks(shape, rng):
    scores = rng.random((*shape, 16))
    indices = scores.argsort(-1)[..., :2]
    result = np.zeros_like(scores, dtype=np.uint8)
    np.put_along_axis(result, indices, 1, axis=-1)
    return result.reshape(*shape, 8, 2)


def collect(plan, device):
    specs = tasks(plan["source_functions"], plan["train_pair_remainders"])
    save_json(OUTPUT / "source_tasks.json", specs)
    rows, validation, task_ids, metrics = [], [], [], []
    directory = OUTPUT / "source"
    directory.mkdir()
    for t, spec in enumerate(specs):
        seed = plan["seed"] + t * 1009
        rng = np.random.default_rng(seed)
        count = plan["source_fits_per_task"]
        masks = random_masks((1, 1, count), rng)
        heldout = rng.random(count) < plan["source_validation_fraction"]
        result, _ = fit(plan, [spec], masks, [plan["source_sgd"]], seed, device)
        good = result["query"].ravel() < plan["success_mse"]
        np.savez_compressed(directory / f"task_{t:02d}.npz",
                            masks=masks.reshape(count, 16),
                            query=result["query"].ravel(),
                            population=result["population"].ravel(),
                            heldout=heldout,
                            elite_weights=result["weights"].reshape(count, 8, 2)[good],
                            elite_amplitudes=result["amplitude"].reshape(count, 2)[good])
        rows.append(masks.reshape(count, 16)[good])
        validation.append(heldout[good])
        task_ids.append(np.full(good.sum(), t))
        metrics.append({"task": t, "fits": count, "elites": int(good.sum()),
                        "validation_elites": int((good & heldout).sum()),
                        "seconds": result["seconds"]})
        print(json.dumps({"source": metrics[-1]}), flush=True)
    bank = {"masks": np.concatenate(rows), "validation": np.concatenate(validation),
            "task_ids": np.concatenate(task_ids)}
    np.savez_compressed(OUTPUT / "bank.npz", **bank)
    save_json(OUTPUT / "bank_summary.json", {
        "tasks": metrics, "total_fits": sum(m["fits"] for m in metrics),
        "elites": len(bank["masks"]),
        "unique_masks": len(np.unique(bank["masks"], axis=0)),
        "train_elites": int((~bank["validation"]).sum()),
        "validation_elites": int(bank["validation"].sum()),
    })
    return bank


def train_prior(plan, bank, device):
    training = torch.tensor(bank["masks"][~bank["validation"]],
                            dtype=torch.long, device=device)
    validation = torch.tensor(bank["masks"][bank["validation"]],
                              dtype=torch.long, device=device)
    models = []
    for index, seed in enumerate(plan["prior_seeds"]):
        torch.manual_seed(seed)
        model = IndexMADE(16, 2, plan["prior_width"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=plan["prior_learning_rate"])
        history, best, selected = [], float("inf"), None
        for step in range(plan["prior_steps"] + 1):
            if step % 25 == 0:
                with torch.no_grad():
                    train_loss = float(F.cross_entropy(model(training).flatten(0, 1),
                                                        training.flatten()))
                    val_loss = float(F.cross_entropy(model(validation).flatten(0, 1),
                                                      validation.flatten()))
                history.append({"step": step, "train_nll": train_loss,
                                "validation_nll": val_loss})
                if val_loss < best:
                    best, selected = val_loss, copy.deepcopy(model.state_dict())
            if step == plan["prior_steps"]:
                break
            loss = F.cross_entropy(model(training).flatten(0, 1), training.flatten())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.load_state_dict(selected)
        model.eval().requires_grad_(False)
        torch.save(selected, OUTPUT / f"prior_{index}.pt")
        save_json(OUTPUT / f"prior_{index}_history.json", history)
        models.append(model)
        print(json.dumps({"prior": index, "validation_nll": best}), flush=True)
    return models


@torch.no_grad()
def propose(method, shape, seed, bank, model, temperature, device, specs=None):
    rng = np.random.default_rng(seed)
    count = int(np.prod(shape))
    torch.manual_seed(seed)
    training = bank["masks"][~bank["validation"]]
    if method == "ar":
        values = model.sample_cached(count, temperature).cpu().numpy()
    elif method == "ar_budget2":
        values = torch.zeros(count, 16, dtype=torch.long, device=device)
        for p in range(16):
            logits = model(values)[:, p] / temperature
            used = values.sum(1)
            logits[used == 2, 1] = -torch.inf
            logits[2 - used == 16 - p, 0] = -torch.inf
            values[:, p] = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
        values = values.cpu().numpy()
    elif method == "frequency":
        values = rng.random((count, 16)) < training.mean(0)
    elif method == "uniform":
        return random_masks(shape, rng)
    elif method == "replay":
        values = training[rng.integers(len(training), size=count)]
    elif method in ["dense_sgd", "dense_adam"]:
        values = np.ones((count, 16), dtype=np.uint8)
    elif method == "oracle":
        values = np.zeros((*shape, 8, 2), dtype=np.uint8)
        for t, spec in enumerate(specs):
            values[t, ..., spec["rows"][0], 0] = 1
            values[t, ..., spec["rows"][1], 1] = 1
        return values
    else:
        raise ValueError(method)
    return values.astype(np.uint8).reshape(*shape, 8, 2)


def solve(plan, specs, method, searches, seed, bank, model, temperature,
          config, device):
    static = method in ["dense_sgd", "dense_adam", "oracle"]
    starts = 64 if static else plan["exploration"]
    shape = (len(specs), searches, starts)
    masks = propose(method, shape, seed + 11, bank, model, temperature, device, specs)
    first, history = fit(plan, specs, masks, [config], seed, device,
                         reuse=0 if static else plan["exploration_initial_vectors"])
    if static:
        result = first
    else:
        choice = first["query"].argmin(2)
        ti, si = np.indices(choice.shape)
        selected_masks = masks[ti, si, choice]
        second_masks = np.repeat(selected_masks[:, :, None], plan["refits"], axis=2)
        second, history2 = fit(plan, specs, second_masks, [config], seed, device, phase=1)
        result = {key: np.concatenate([first[key], second[key]], axis=2)
                  for key in first if key != "seconds"}
        result["seconds"] = first["seconds"] + second["seconds"]
        for item, later in zip(history, history2):
            for metric in ["train", "population"]:
                item[metric] = np.concatenate([item[metric], later[metric]], axis=2).tolist()
    selected = result["query"].argmin(2)
    ti, si = np.indices(selected.shape)
    result["selected"] = selected
    result["selected_population"] = result["population"][ti, si, selected]
    result["selected_query"] = result["query"][ti, si, selected]
    return result, history


def calibrate(plan, bank, models, device):
    specs = tasks(plan["calibration_functions"], plan["calibration_pair_remainders"])
    save_json(OUTPUT / "calibration_tasks.json", specs)
    directory = OUTPUT / "calibration"
    directory.mkdir()
    settings = {"dense": {}, "temperature": {}}
    for optimizer in ["sgd", "adam"]:
        configs = [{"optimizer": optimizer, "lr": lr, "scale": scale}
                   for lr, scale in itertools.product(plan[f"{optimizer}_rates"],
                                                      plan["dense_scales"])]
        masks = np.ones((len(specs), len(configs), plan["calibration_starts"], 8, 2))
        result, history = fit(plan, specs, masks, configs, plan["seed"] + 50000, device)
        save_fit(directory / f"dense_{optimizer}", result, history)
        scores = result["query"].min(2).mean(0)
        selected = int(scores.argmin())
        settings["dense"][optimizer] = configs[selected]
        settings[f"dense_{optimizer}_scores"] = [
            {**config, "mean_best_query": float(score)}
            for config, score in zip(configs, scores)
        ]
        print(json.dumps({"calibrated_dense": optimizer, **configs[selected]}), flush=True)
    for method in ["ar", "ar_budget2"]:
        scores = []
        for temperature in plan["temperatures"]:
            seed_scores = []
            for index, model in enumerate(models):
                result, history = solve(
                    plan, specs, method, plan["calibration_searches"],
                    plan["seed"] + 60000 + index * 1000, bank, model, temperature,
                    plan["structured_sgd"], device,
                )
                save_fit(directory / f"{method}_t{temperature}_seed{index}", result, history)
                seed_scores.append(float(result["selected_query"].mean()))
            scores.append({"temperature": temperature, "query": float(np.mean(seed_scores)),
                           "seed_query": seed_scores})
            print(json.dumps({"calibrated_method": method, **scores[-1]}), flush=True)
        settings["temperature"][method] = min(scores, key=lambda row: row["query"])["temperature"]
        settings[method + "_scores"] = scores
    settings["frozen_before_target_generation"] = True
    save_json(OUTPUT / "selection_frozen.json", settings)
    return settings


def evaluate(plan, bank, models, settings, device):
    for condition, remainders, searches in [
        ("novel", plan["novel_pair_remainders"], plan["novel_searches_per_task_seed"]),
        ("familiar", plan["train_pair_remainders"], plan["familiar_searches_per_task_seed"]),
    ]:
        specs = tasks(plan["target_functions"], remainders)
        directory = OUTPUT / condition
        directory.mkdir()
        save_json(directory / "tasks.json", specs)
        for index, model in enumerate(models):
            for method in plan["methods"]:
                config = (settings["dense"][method.removeprefix("dense_")]
                          if method.startswith("dense_") else plan["structured_sgd"])
                temperature = settings["temperature"].get(method, 1.0)
                seed = plan["seed"] + 100000 + index * 1000
                result, history = solve(plan, specs, method, searches, seed,
                                        bank, model, temperature, config, device)
                save_fit(directory / f"{method}_seed{index}", result, history)
                print(json.dumps({
                    "condition": condition, "method": method, "seed": index,
                    "success": int((result["selected_population"] < plan["success_mse"]).sum()),
                    "searches": result["selected_population"].size,
                    "mean_mse": float(result["selected_population"].mean()),
                    "seconds": result["seconds"],
                }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase", choices=["all", "prepare", "calibrate", "evaluate"], default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads((BASE / "protocol.json").read_text())
    if args.phase in ["all", "prepare"]:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        shutil.copy2(BASE / "protocol.json", OUTPUT / "protocol.json")
        shutil.copy2(__file__, OUTPUT / "run_source.py")
        shutil.copy2(BASE.parents[1] / "pattern/autoregressive/model.py", OUTPUT / "made_source.py")
        save_json(OUTPUT / "environment.json", {
            "torch": torch.__version__, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(args.device),
            "protocol_sha256": hashlib.sha256((BASE / "protocol.json").read_bytes()).hexdigest(),
        })
        bank = collect(plan, args.device)
        models = train_prior(plan, bank, args.device)
    else:
        with np.load(OUTPUT / "bank.npz") as raw:
            bank = dict(raw)
        models = []
        for index in range(len(plan["prior_seeds"])):
            model = IndexMADE(16, 2, plan["prior_width"]).to(args.device)
            model.load_state_dict(torch.load(OUTPUT / f"prior_{index}.pt", weights_only=True,
                                             map_location=args.device))
            models.append(model.eval().requires_grad_(False))
    if args.phase in ["all", "calibrate"]:
        settings = calibrate(plan, bank, models, args.device)
    elif args.phase == "evaluate":
        settings = json.loads((OUTPUT / "selection_frozen.json").read_text())
    if args.phase in ["all", "evaluate"]:
        evaluate(plan, bank, models, settings, args.device)
        save_json(OUTPUT / "COMPLETE.json", {"complete": True})


if __name__ == "__main__":
    main()
