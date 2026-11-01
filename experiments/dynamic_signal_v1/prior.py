"""Learn a frozen distribution of indexators from source weight structures."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.dynamic_signal_v1.fit import fit
from experiments.dynamic_signal_v1.run import choose, configurations
from pattern.autoregressive.model import IndexMADE


def train_prior(plan, bank_path, output, device):
    bank = np.load(bank_path / "bank.npz")
    values = torch.tensor(bank["indices"].reshape(-1, 4096), device=device)
    training, validation = values[bank["role"] == 0], values[bank["role"] == 1]
    counts = F.one_hot(training, 17).float().sum(0) + 0.01
    probabilities = counts / counts.sum(-1, keepdim=True)
    torch.manual_seed(plan["seed"] + 1000)
    model = IndexMADE(4096, 17, 128).to(device)
    with torch.no_grad():
        model.layers[4].weight.zero_()
        model.layers[4].bias.copy_(probabilities.log().flatten())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float("inf")
    history = []
    for step in range(1001):
        if step % 25 == 0:
            with torch.no_grad():
                loss = float(
                    F.cross_entropy(
                        model(validation).flatten(0, 1), validation.flatten()
                    )
                )
            history.append({"step": step, "validation_nll": loss})
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), output / "made.pt")
            print("PRIOR", step, loss, flush=True)
        if step == 1000:
            break
        loss = F.cross_entropy(model(training).flatten(0, 1), training.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.load_state_dict(torch.load(output / "made.pt", weights_only=True))
    # Numerical diagnosis of the cached sampler, without a separate test suite.
    greedy = model.sample_cached(2, 0)
    with torch.no_grad():
        logits = model(greedy)
        mismatch = int((greedy != logits.argmax(-1)).sum())
    if mismatch:
        raise RuntimeError(f"Cached greedy sampling disagrees at {mismatch} positions")
    np.savez_compressed(
        output / "frequencies.npz", probabilities=probabilities.cpu().numpy()
    )
    (output / "training.json").write_text(
        json.dumps(
            {
                "history": history,
                "best_validation_nll": best_loss,
                "cached_greedy_mismatches": mismatch,
                "bank": str(bank_path),
                "train_structures": len(training),
                "validation_structures": len(validation),
            },
            indent=2,
        )
        + "\n"
    )


def draw(method, temperature, count, model, probabilities, consensus, device):
    if method == "ar":
        return model.sample_cached(count, temperature).reshape(count, 128, 32)
    if method == "frequency":
        if temperature == 0:
            return probabilities.argmax(-1).reshape(1, 128, 32).expand(count, -1, -1)
        weights = (probabilities.log() / temperature).softmax(-1)
        return torch.multinomial(weights, count, replacement=True).T.reshape(
            count, 128, 32
        )
    value = torch.tensor(consensus, device=device).expand(count, -1, -1).clone()
    if method == "random":
        for index in range(count):
            value[index] = (
                value[index]
                .flatten()[torch.randperm(4096, device=device)]
                .reshape(128, 32)
            )
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-id", type=int, required=True)
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--name", default="weight_large")
    parser.add_argument(
        "--stage", choices=["train", "calibrate", "target"], required=True
    )
    parser.add_argument("--size", type=int)
    parser.add_argument(
        "--protocol", default="experiments/dynamic_signal_v1/protocol.json"
    )
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    torch.set_num_threads(2)
    device = torch.device("cuda")
    bank = Path(args.bank_dir)
    root = Path(plan["run"])
    prior = root / "priors" / args.name / f"noise_{args.noise_id}"
    if args.stage == "train":
        prior.mkdir(parents=True, exist_ok=False)
        train_prior(plan, bank, prior, device)
        return
    probabilities = torch.tensor(
        np.load(prior / "frequencies.npz")["probabilities"], device=device
    )
    consensus = np.load(bank / "extracted.npz")["consensus"]
    model = IndexMADE(4096, 17, 128).to(device)
    model.load_state_dict(torch.load(prior / "made.pt", weights_only=True))
    role = "calibration" if args.stage == "calibrate" else "target"
    data = dict(np.load(root / role / f"noise_{args.noise_id}.npz"))
    output = prior / f"{args.stage}_n{args.size}"
    if (output / "complete.json").exists():
        print("Already completed", output, flush=True)
        return
    output.mkdir(exist_ok=False)
    tasks = len(data["support_y"])
    restarts = (
        plan["calibration_restarts"]
        if args.stage == "calibrate"
        else plan["target_restarts"]
    )
    selected = {}
    for method in ["consensus", "ar", "frequency", "random"]:
        if args.stage == "calibrate":
            temperatures = [0, 0.1, 0.25, 1.0] if method in ["ar", "frequency"] else [0]
            candidates = list(
                itertools.product(temperatures, configurations(plan, "oracle"))
            )
        else:
            calibration_size = max(
                [n for n in plan["calibration_sizes"] if n <= args.size] or [256]
            )
            settings = json.loads(
                (
                    prior / f"calibrate_n{calibration_size}" / "selection.json"
                ).read_text()
            )[method]
            candidates = [(settings["temperature"], settings["config"])]
        torch.manual_seed(plan["seed"] + 1100)
        # Same proposals across numerical configurations at a given temperature.
        proposals = {
            temperature: draw(
                method,
                temperature,
                tasks * restarts,
                model,
                probabilities,
                consensus,
                device,
            )
            for temperature in dict.fromkeys(value for value, _ in candidates)
        }
        configs = [config for _, config in candidates]
        indices = torch.cat([proposals[temperature] for temperature, _ in candidates])
        arrays = fit(
            plan,
            data,
            method,
            configs,
            args.size,
            restarts,
            device,
            output / method,
            indices=indices,
        )
        best, grid = choose(arrays, configs, tasks, restarts)
        selected[method] = {
            "temperature": candidates[best][0],
            "config": configs[best],
            "grid": [
                dict(item, temperature=candidates[index][0])
                for index, item in enumerate(grid)
            ],
        }
        (output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    (output / "complete.json").write_text(
        json.dumps(
            {"bank": str(bank), "size": args.size, "stage": args.stage}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
