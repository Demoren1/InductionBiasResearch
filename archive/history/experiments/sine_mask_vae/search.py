"""Validate dense SGD settings, then compare frozen policies on unseen pairs."""

import json
import time

import numpy as np
import torch

from .settings import (
    DIRECTORY,
    OPTIONS,
    RUN,
    SEEDS,
    digest,
    families,
    implementation_hashes,
    snapshot_sources,
)
from .core import correct_mask, fit, random_masks, save_json, task
from .train import load_cards, restore


METHODS = [
    "dense",
    "dense_tuned",
    "dense_long",
    "random",
    "elite_pool",
    "vanilla",
    "partwise_training",
    "empirical_product",
    "oracle",
]


def proposals(method, count, rng, spec, models, seed, training):
    if method.startswith("dense"):
        return np.ones((count, 8, 2), dtype=np.float32)
    if method == "random":
        return random_masks(count, rng)
    if method == "oracle":
        return correct_mask(spec, count)
    if method == "elite_pool":
        masks = np.stack([y[0].numpy() for _, y in training])
        return masks[rng.integers(len(masks), size=count)]
    if method == "empirical_product":
        frequencies = (
            torch.stack([y.mean(0) for _, y in training]).mean(0).numpy()
        )
        masks = np.zeros((count, 8, 2), dtype=np.float32)
        for j in range(2):
            rows = rng.choice(
                8, size=count, p=frequencies[:, j] / frequencies[:, j].sum()
            )
            masks[np.arange(count), rows, j] = 1
        return masks
    model = models[(method, seed)]
    latent = torch.tensor(
        rng.standard_normal((count, 16)), dtype=torch.float32
    )
    with torch.no_grad():
        logits = model.decode(latent).flatten(1).numpy()
    indices = np.argsort(-logits, axis=1, kind="stable")[:, :2]
    masks = np.zeros((count, 16), dtype=np.float32)
    np.put_along_axis(masks, indices, 1, axis=1)
    return masks.reshape(-1, 8, 2)


def summarize(values):
    values = np.asarray(values)
    return {
        "success": int((values < 0.1).sum()),
        "searches": len(values),
        "rate": float((values < 0.1).mean()),
        "mean_selected_population_mse": float(values.mean()),
    }


def select(result, trials, budget):
    index = result["empirical"].reshape(trials, budget).argmin(1)
    flat_index = np.arange(trials) * budget + index
    return result["losses"][flat_index], flat_index


def validate(models, training):
    path = RUN / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    trials, budget = 16, 64
    records, decoder_records = [], []
    for lr in (0.01, 0.05, 0.2):
        for scale in (0.35, 0.7, 1.25, 2.0):
            selected = []
            for a, b in families("validation"):
                spec = task(a, b, 100)
                seed = 180100 + a * 1009 + b * 17
                masks = np.ones((trials * budget, 8, 2), dtype=np.float32)
                result = fit(
                    masks, spec, seed, **{**OPTIONS, "lr": lr, "scale": scale}
                )
                selected.extend(select(result, trials, budget)[0].tolist())
            record = {"lr": lr, "scale": scale, **summarize(selected)}
            records.append(record)
            print("dense validation", record, flush=True)
    best = min(
        records,
        key=lambda r: (-r["success"], r["mean_selected_population_mse"]),
    )
    for method in (
        "random",
        "vanilla",
        "partwise_training",
        "empirical_product",
    ):
        selected = []
        for model_seed in SEEDS:
            for a, b in families("validation"):
                spec = task(a, b, 100)
                seed = 180100 + a * 1009 + b * 17 + model_seed * 31
                masks = proposals(
                    method,
                    trials * budget,
                    np.random.default_rng(seed + 3),
                    spec,
                    models,
                    model_seed,
                    training,
                )
                result = fit(masks, spec, seed, **OPTIONS)
                selected.extend(select(result, trials, budget)[0].tolist())
        record = {"method": method, **summarize(selected)}
        decoder_records.append(record)
        print("policy validation", record, flush=True)
    selection = {
        "dense_options": {**OPTIONS, "lr": best["lr"], "scale": best["scale"]},
        "dense_grid": records,
        "policies": decoder_records,
        "checkpoint_hashes": {
            str(k): digest(RUN / r["checkpoint"])
            for k, r in load_model_records().items()
        },
        "search_source_sha256": digest(__file__),
        "core_source_sha256": digest(DIRECTORY / "core.py"),
        "implementation_sha256": implementation_hashes(),
        "partwise_options": "Accepted method, unchanged; all 3 seeds retained",
    }
    save_json(path, selection)
    return selection


def load_model_records():
    return {
        (r["variant"], r["seed"]): r
        for r in json.loads((RUN / "models.json").read_text())
    }


def evaluate(models, training, selection):
    if selection.get("implementation_sha256") != implementation_hashes():
        raise RuntimeError(
            "Implementation changed after validation; use a new run directory"
        )
    if selection["search_source_sha256"] != digest(__file__) or selection[
        "core_source_sha256"
    ] != digest(DIRECTORY / "core.py"):
        raise RuntimeError("Protocol implementation changed after validation")
    destination = RUN / "test"
    destination.mkdir(exist_ok=True)
    trials = 64
    for function in (201, 202):
        for model_seed in SEEDS:
            for a, b in families("test"):
                spec = task(a, b, function)
                seed = (
                    280100
                    + function * 10007
                    + a * 1009
                    + b * 17
                    + model_seed * 31
                )
                for method in METHODS:
                    name = f"{method}_a{a}_b{b}_f{function}_s{model_seed}"
                    record_path = destination / f"{name}.json"
                    if record_path.exists():
                        continue
                    started = time.perf_counter()
                    budget = 8 if method == "dense_long" else 64
                    options = (
                        selection["dense_options"]
                        if method in ("dense_tuned", "dense_long")
                        else OPTIONS
                    )
                    options = {**options, "steps": 65536 // budget}
                    masks = proposals(
                        method,
                        trials * budget,
                        np.random.default_rng(seed + 3),
                        spec,
                        models,
                        model_seed,
                        training,
                    )
                    result = fit(masks, spec, seed, **options)
                    selected, indices = select(result, trials, budget)
                    arrays = {
                        "selected": selected,
                        "candidate_losses": result["losses"],
                        "candidate_empirical": result["empirical"],
                        "masks": masks,
                        "selected_indices": indices,
                    }
                    record = {
                        "method": method,
                        "a": a,
                        "b": b,
                        "function": function,
                        "model_seed": model_seed,
                        "options": options,
                        "budget": budget,
                        "seconds": time.perf_counter() - started,
                        **summarize(selected),
                    }
                    if method == "partwise_training":
                        # A good sparse solution must survive training the dense network.
                        continued = fit(
                            np.ones((trials, 8, 2), dtype=np.float32),
                            spec,
                            seed + 90001,
                            **OPTIONS,
                            initial=(
                                result["weights"][indices],
                                result["amplitude"][indices],
                            ),
                        )
                        arrays["unmasked_losses"] = continued["losses"]
                        record["success_retained_after_unmask"] = int(
                            (
                                (selected < 0.1) & (continued["losses"] < 0.1)
                            ).sum()
                        )
                    np.savez_compressed(destination / f"{name}.npz", **arrays)
                    save_json(record_path, record)
                    print(
                        "test",
                        name,
                        record["success"],
                        "/",
                        trials,
                        flush=True,
                    )
    records = [json.loads(p.read_text()) for p in destination.glob("*.json")]
    expected = 2 * len(SEEDS) * len(families("test")) * len(METHODS)
    if len(records) != expected:
        raise RuntimeError(f"Incomplete test: {len(records)} vs {expected}")
    save_json(
        RUN / "TEST_COMPLETE.json",
        {"contexts": len(records), "searches_per_method": 1536},
    )


def main(phase="all"):
    torch.set_num_threads(1)
    models = {
        key: restore(record) for key, record in load_model_records().items()
    }
    _, roles = load_cards()
    snapshot_sources()
    if phase in ("validate", "all"):
        selection = validate(models, roles[0])
    else:
        selection = json.loads((RUN / "selection.json").read_text())
    if phase in ("test", "all"):
        evaluate(models, roles[0], selection)


if __name__ == "__main__":
    main()
