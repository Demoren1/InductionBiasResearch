"""Autoregressive generation from a bank obtained by random structural search."""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.mlp_structure_discovery_v1.extract import select_settings
from experiments.mlp_structure_discovery_v1.fit import fit_models, read_observations
from experiments.random_structure_search_v1.search import decode
from pattern.autoregressive.model import IndexMADE


def load_prior(root, plan, device):
    bank = dict(np.load(root / "selection" / "bank.npz"))
    kind = json.loads((root / "selection" / "selection.json").read_text())["kind"]
    classes = int(bank["tokens"].max()) + 1
    model = IndexMADE(bank["best"].size, classes, plan["prior_width"]).to(device)
    model.load_state_dict(
        torch.load(root / "prior" / "made.pt", map_location=device, weights_only=True)
    )
    model.eval()
    probabilities = torch.as_tensor(
        np.load(root / "prior" / "frequencies.npz")["probabilities"], device=device
    )
    return bank, kind, model, probabilities


def proposals(method, temperature, count, bank, kind, model, probabilities, rng):
    shape = bank["best"].shape
    if method == "dense":
        tokens = np.ones((count, *shape), dtype=np.int16)
        return decode(tokens, "mask")
    if method == "ar":
        tokens = (
            model.sample_cached(count, temperature).cpu().numpy().reshape(count, *shape)
        )
    elif method == "frequency":
        if temperature == 0:
            tokens = probabilities.argmax(-1)[None].expand(count, -1)
        else:
            tokens = torch.multinomial(
                (probabilities.log() / temperature).softmax(-1), count, replacement=True
            ).T
        tokens = tokens.cpu().numpy().reshape(count, *shape)
    elif method == "bank_sample":
        tokens = bank["tokens"][rng.integers(len(bank["tokens"]), size=count)].copy()
    else:
        tokens = np.broadcast_to(bank["best"], (count, *shape)).copy()
        if method == "random":
            for i in range(count):
                tokens[i] = rng.permutation(tokens[i].ravel()).reshape(shape)
        elif method == "shuffled":
            for i in range(count):
                tokens[i] = tokens[i, rng.permutation(shape[0])]
    return decode(tokens, "mask" if method == "mask_only" else kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="experiments/random_structure_search_v1/protocol.json"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["output"])
    output = root / "prior"
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    bank = dict(np.load(root / "selection" / "bank.npz"))
    values = torch.as_tensor(
        bank["tokens"].reshape(len(bank["tokens"]), -1), device=device, dtype=torch.long
    )
    classes = int(values.max()) + 1
    rng = np.random.default_rng(plan["seed"] + 801)
    order = rng.permutation(len(values))
    split = max(1, len(values) // 4)
    validation_ids, training_ids = order[:split], order[split:]
    train, validation = values[training_ids], values[validation_ids]
    counts = F.one_hot(train, classes).float().sum(0) + 0.01
    probabilities = counts / counts.sum(-1, keepdim=True)
    torch.manual_seed(plan["seed"] + 802)
    model = IndexMADE(values.shape[1], classes, plan["prior_width"]).to(device)
    with torch.no_grad():
        model.layers[4].weight.zero_()
        model.layers[4].bias.copy_(probabilities.log().flatten())
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["prior_learning_rate"])
    best = float("inf")
    history = []
    for step in range(plan["prior_steps"] + 1):
        if step % 25 == 0:
            with torch.no_grad():
                loss = float(
                    F.cross_entropy(
                        model(validation).flatten(0, 1), validation.flatten()
                    )
                )
            history.append({"step": step, "validation_nll": loss})
            if loss < best:
                best = loss
                torch.save(model.state_dict(), output / "made.pt")
            print("PRIOR", step, round(loss, 6), flush=True)
        if step == plan["prior_steps"]:
            break
        loss = F.cross_entropy(model(train).flatten(0, 1), train.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    np.savez_compressed(
        output / "frequencies.npz", probabilities=probabilities.cpu().numpy()
    )
    (output / "training.json").write_text(
        json.dumps(
            {
                "history": history,
                "training_ids": training_ids.tolist(),
                "validation_ids": validation_ids.tolist(),
                "split_unit": "bank candidate; related evolutionary descendants can occur on both sides",
                "best_validation_nll": best,
            },
            indent=2,
        )
        + "\n"
    )
    del optimizer, model
    bank, kind, model, probabilities = load_prior(root, plan, device)
    data = read_observations(
        plan["source_data"], plan["selection_tasks"], plan["transfer_support"]
    )
    cases = [
        {"method": method, "temperature": temperature}
        for method in [
            "best",
            "bank_sample",
            "ar",
            "frequency",
            "random",
            "shuffled",
            "mask_only",
            "dense",
        ]
        for temperature in (
            plan["prior_temperatures"] if method in {"ar", "frequency"} else [0.0]
        )
    ]
    configs = list(itertools.product(plan["learning_rates"], plan["weight_decays"]))
    torch.manual_seed(plan["seed"] + 805)
    rng = np.random.default_rng(plan["seed"] + 806)
    count = len(data["support_y"]) * plan["selection_restarts"]
    candidates = [
        proposals(
            case["method"],
            case["temperature"],
            count,
            bank,
            kind,
            model,
            probabilities,
            rng,
        )
        for case in cases
    ]
    rows = [
        {
            "candidate": candidate,
            "config": config,
            "task": task,
            "restart": restart,
            "lr": rate,
            "weight_decay": decay,
        }
        for candidate in range(len(cases))
        for config, (rate, decay) in enumerate(configs)
        for task in range(len(data["support_y"]))
        for restart in range(plan["selection_restarts"])
    ]
    indices = np.concatenate(
        [np.tile(value, (len(configs), 1, 1)) for value in candidates]
    )
    arrays = fit_models(
        data,
        indices,
        rows,
        plan,
        output / "selection_fit",
        device,
        save_parameters=False,
    )
    grid = select_settings(arrays, rows)
    selected = {}
    for method in dict.fromkeys(case["method"] for case in cases):
        eligible = [row for row in grid if cases[row["candidate"]]["method"] == method]
        chosen = dict(min(eligible, key=lambda row: row["query_loss"]))
        chosen.update(cases[chosen["candidate"]])
        selected[method] = chosen
    (output / "selection.json").write_text(
        json.dumps({"chosen": selected, "cases": cases, "grid": grid}, indent=2) + "\n"
    )
    paths = [
        root / "protocol.json",
        Path(__file__).with_name("benchmark.json"),
        root / "selection" / "bank.npz",
        root / "selection" / "selection.json",
        output / "made.pt",
        output / "frequencies.npz",
        output / "selection.json",
        *Path(__file__).parent.glob("*.py"),
        Path("experiments/mlp_structure_discovery_v1/fit.py"),
        Path("experiments/mlp_structure_discovery_v1/extract.py"),
        Path("pattern/autoregressive/model.py"),
    ]
    (root / "frozen.json").write_text(
        json.dumps(
            {
                "frozen_unix": time.time(),
                "kind": kind,
                "selected": selected,
                "sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in paths
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("FROZEN", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
