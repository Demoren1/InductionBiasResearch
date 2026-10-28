"""Fit new functions using frozen structure priors and explicit controls."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from prior import StructureVAE
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
METHODS = [
    "bank",
    "best_bank",
    "best_support",
    "column_sample",
    "column_search",
    "entry_search",
    "entry_uniform",
    "column_vae_sample",
    "column_vae_z",
    "global_vae_z",
    "untrained_column_vae_z",
    "random_codes",
    "shuffled_support",
    "dense_adam",
    "dense_sgd",
    "dense_transfer",
    "oracle",
]


def fit(out, x, labels, method, starts, steps, rates, fixed_indices=None, seed=283030):
    device = x.device
    tasks, count = len(labels), len(labels) * starts
    dense = method.startswith("dense")
    bank = np.load(out / "structure_bank.npz")
    split = np.load(out / "prior_split.npz")
    train = split["train"]
    source_indices = bank["indices"][train]
    mask = torch.tensor(np.load(out / "learned_support.npz")["mask"], device=device)
    reference = int(train[np.argmin(bank["quality"][train])])
    if method == "best_support":
        mask = torch.tensor(bank["indices"][reference] != 0, device=device)
    numeric_count = (
        2304 if dense else int(mask.sum()) if method == "best_support" else 36
    )
    if fixed_indices is not None:
        numeric_count = max(numeric_count, int(fixed_indices.max()))
    torch.manual_seed(seed)
    values = torch.randn(count, numeric_count, device=device) * (0.1 if dense else 0.3)
    if method == "dense_transfer":
        source_path = out / "original_dense_source_models.npz"
        source = np.load(
            source_path if source_path.exists() else out / "source_models.npz"
        )
        best = [
            ids[np.argmin(source["quality"][ids])]
            for task in np.unique(source["task"])
            for ids in [np.flatnonzero(source["task"] == task)]
        ]
        choices = torch.randint(len(best), (count,), device=device).cpu().numpy()
        values = torch.tensor(
            source["weights"][np.array(best)[choices]], device=device
        ).flatten(1)
    values.requires_grad_()
    numeric = (
        torch.optim.SGD([values], lr=1, momentum=0.9)
        if method == "dense_sgd"
        else torch.optim.Adam([values], lr=1)
    )
    learning_rates = torch.tensor(rates, device=device)
    learning_rates = (
        learning_rates.repeat(tasks)
        if len(rates) == starts
        else learning_rates.repeat(count)
    )
    y = labels.repeat_interleave(starts, 0)
    scale = y[:, :256].var(1, unbiased=False).clamp_min(1e-8)
    z = torch.zeros(count, 0, device=device)
    model, structural, options, option_flat = None, None, None, None
    entry_positions = None
    torch.manual_seed(283031)
    if dense:
        indices = torch.zeros(count, 64, 36, dtype=torch.long, device=device)
    elif fixed_indices is not None:
        indices = torch.tensor(fixed_indices, device=device).repeat_interleave(
            starts, 0
        )
    elif method == "best_bank":
        indices = torch.tensor(bank["indices"][reference], device=device)[None].expand(
            count, -1, -1
        )
    elif method == "best_support":
        template = torch.zeros(64, 36, dtype=torch.long, device=device)
        template[mask] = torch.arange(1, numeric_count + 1, device=device)
        indices = template[None].expand(count, -1, -1)
    elif method == "bank":
        choices = torch.randint(len(source_indices), (count,), device=device)
        indices = torch.tensor(source_indices, device=device)[choices]
    elif method.startswith("column_") and method in ["column_sample", "column_search"]:
        fallback = (
            np.stack([(source_indices == code).sum(0) for code in range(1, 37)]).argmax(
                0
            )
            + 1
        )
        source_indices = np.where(source_indices != 0, source_indices, fallback[None])
        columns = [np.unique(source_indices[:, :, h], axis=0) for h in range(36)]
        capacity = max(map(len, columns))
        padded = np.stack(
            [column[np.arange(capacity) % len(column)] for column in columns]
        )
        options = torch.tensor(padded, device=device)
        z = torch.randn(count, 36, capacity, device=device) * 0.1
        choices = z.argmax(-1)
        indices = options[torch.arange(36, device=device)[None], choices].transpose(
            1, 2
        )
        # Same source-derived support for column/VAE/random-code controls.
        indices = torch.where(mask[None], indices.clamp_min(1), 0)
        option_flat = options.permute(2, 0, 1).reshape(1, -1).expand(count, -1)
        if method == "column_search":
            z.requires_grad_()
            structural = torch.optim.Adam([z], lr=0.03)
    elif method in ["entry_search", "entry_uniform"]:
        entry_positions = mask.flatten().nonzero().flatten()
        counts = np.stack(
            [(source_indices == code).sum(0) for code in range(1, 37)], -1
        )
        counts = torch.tensor(counts, device=device, dtype=torch.float)[mask] + 0.5
        logits = (counts / counts.sum(-1, keepdim=True)).log()
        if method == "entry_uniform":
            logits = torch.zeros_like(logits)
        z = logits[None].repeat(count, 1, 1) + 0.05 * torch.randn(
            count, len(entry_positions), 36, device=device
        )
        z.requires_grad_()
        structural = torch.optim.Adam([z], lr=0.03)
        indices = (
            torch.zeros(count, 2304, dtype=torch.long, device=device)
            .scatter(1, entry_positions[None].expand(count, -1), z.argmax(-1) + 1)
            .reshape(count, 64, 36)
        )
    elif method in ["random_codes", "shuffled_support"]:
        indices = torch.randint(1, 37, (count, 64, 36), device=device)
        if method == "random_codes":
            indices = indices * mask[None]
        else:
            indices *= torch.stack(
                [
                    mask.flatten()[torch.randperm(2304, device=device)].reshape(64, 36)
                    for _ in range(count)
                ]
            )
    else:
        kind = "global" if method.startswith("global") else "column"
        model = StructureVAE(kind).to(device)
        name = (
            f"{kind}_untrained.pt"
            if method.startswith("untrained")
            else f"{kind}_prior.pt"
        )
        model.load_state_dict(
            torch.load(out / name, weights_only=True, map_location=device)
        )
        model.requires_grad_(False).eval()
        z = torch.randn(
            (count, 64) if kind == "global" else (count, 36, 8), device=device
        )
        with torch.no_grad():
            indices = (model.decode(z)[..., 1:].argmax(-1) + 1) * mask[None]
        if method.endswith("_z"):
            z.requires_grad_()
            structural = torch.optim.Adam([z], lr=0.03)

    def decoded():
        if model is not None:
            return (model.decode(z)[..., 1:].argmax(-1) + 1) * mask[None]
        if options is not None:
            ids = options[
                torch.arange(36, device=device)[None], z.argmax(-1)
            ].transpose(1, 2)
            return torch.where(mask[None], ids.clamp_min(1), 0)
        if entry_positions is not None:
            return (
                torch.zeros(count, 2304, dtype=torch.long, device=device)
                .scatter(1, entry_positions[None].expand(count, -1), z.argmax(-1) + 1)
                .reshape(count, 64, 36)
            )
        return indices

    def predict(batch, numbers, index, weights_override=None):
        if weights_override is not None:
            weights = weights_override
        elif dense:
            weights = numbers.reshape(count, 64, 36)
        else:
            codebook = torch.cat([numbers.new_zeros(count, 1), numbers], -1)
            weights = codebook.gather(1, index.flatten(1)).reshape(count, 64, 36)
        return torch.einsum("bd,mdh->mbh", batch, weights).relu().sum(-1) / 6

    best_values, best_indices, best_z = (
        values.detach().clone(),
        indices.clone(),
        z.detach().clone(),
    )
    best_error = torch.full((count,), torch.inf, device=device)
    best_step = torch.zeros(count, dtype=torch.long, device=device)
    history = []
    for step in range(steps + 1):
        if step % 128 == 0 or step == steps:
            with torch.no_grad():
                indices = decoded()
                error = torch.zeros(count, device=device)
                for ids in torch.arange(256, 768, device=device).split(128):
                    error += (
                        (predict(x[ids], values, indices) - y[:, ids]).square().sum(1)
                    )
                error /= 512 * scale
                improve = error < best_error
                best_error[improve] = error[improve]
                best_values[improve] = values[improve]
                best_indices[improve] = indices[improve]
                best_z[improve] = z[improve]
                best_step[improve] = step
                history.append({"step": step, "query_nmse": best_error.cpu().tolist()})
        if step == steps:
            break
        update = structural is not None and step >= steps // 4 and step % 4 == 0
        soft_weights = None
        if update:
            if model is not None:
                probabilities = model.decode(z)[..., 1:].softmax(-1)
                indices = (probabilities.detach().argmax(-1) + 1) * mask[None]
                hard = F.one_hot(probabilities.argmax(-1), 36).float()
                straight = hard + probabilities - probabilities.detach()
                soft_weights = (straight * values[:, None, None]).sum(-1) * mask[None]
            elif options is not None:
                probabilities = z.softmax(-1)
                hard = F.one_hot(probabilities.argmax(-1), options.shape[1]).float()
                straight = hard + probabilities - probabilities.detach()
                # Replace missing prototype entries in the common active support
                # consistently with the hard decoder's generic code-1 fallback.
                flat = option_flat.clamp_min(1)
                codebook = torch.cat([values.new_zeros(count, 1), values], -1)
                numbers = codebook.gather(1, flat).reshape(count, 64, 36, -1)
                soft_weights = (numbers * straight[:, None]).sum(-1) * mask[None]
                indices = decoded()
            else:
                probabilities = z.softmax(-1)
                hard = F.one_hot(probabilities.argmax(-1), 36).float()
                straight = hard + probabilities - probabilities.detach()
                active_weights = (straight * values[:, None]).sum(-1)
                soft_weights = (
                    active_weights.new_zeros(count, 2304)
                    .scatter(1, entry_positions[None].expand(count, -1), active_weights)
                    .reshape(count, 64, 36)
                )
                indices = decoded()
        loss = (
            (predict(x[:256], values, indices, soft_weights) - y[:, :256])
            .square()
            .mean(1)
            / scale
        ).sum()
        if update and model is not None:
            loss = loss + 0.0001 * z.square().sum()
        numeric.zero_grad(set_to_none=True)
        if structural is not None:
            structural.zero_grad(set_to_none=True)
        loss.backward()
        before = values.detach().clone()
        numeric.step()
        with torch.no_grad():
            values.copy_(before + (values - before) * learning_rates[:, None])
        if update:
            structural.step()
    test = torch.zeros(count, device=device)
    if len(x) > 768:
        with torch.no_grad():
            for ids in torch.arange(768, len(x), device=device).split(128):
                test += (
                    (predict(x[ids], best_values, best_indices) - y[:, ids])
                    .square()
                    .sum(1)
                )
            test /= (len(x) - 768) * scale
    return {
        "values": best_values.cpu().numpy(),
        "indices": best_indices.cpu().numpy(),
        "z": best_z.cpu().numpy(),
        "step": best_step.cpu().numpy(),
        "query_nmse": best_error.cpu().numpy(),
        "test_nmse": test.cpu().numpy(),
    }, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pilot_v1")
    parser.add_argument(
        "--stage", choices=["calibrate", "evaluate", "refit"], required=True
    )
    parser.add_argument("--steps", type=int, default=2048)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    shutil.copy2(__file__, args.out / f"run_{args.stage}_snapshot.py")
    if args.stage == "calibrate":
        observations = np.load(args.out / "calibration_data.npz")
        x, y = (torch.tensor(observations[key], device=device) for key in ["x", "y"])
        tuning = {}
        for method in [
            "bank",
            "best_support",
            "dense_adam",
            "dense_sgd",
            "dense_transfer",
        ]:
            rates = (
                [0.03, 0.1, 0.3, 1.0]
                if method == "dense_sgd"
                else [0.001, 0.003, 0.01, 0.03]
            )
            cached = args.out / f"calibration_{method}.npz"
            if cached.exists():
                data = dict(np.load(cached))
            else:
                data, history = fit(
                    args.out,
                    x,
                    y,
                    method,
                    8,
                    args.steps,
                    [rate for rate in rates for _ in range(2)],
                )
                np.savez_compressed(cached, **data)
                (args.out / f"calibration_{method}_history.json").write_text(
                    json.dumps(history)
                )
            scores = data["query_nmse"].reshape(2, 4, 2).min(-1).mean(0)
            tuning[method] = {
                "lr": rates[int(scores.argmin())],
                "query_scores": scores.tolist(),
            }
            print("TUNE", method, tuning[method], flush=True)
        (args.out / "evaluation_protocol.json").write_text(
            json.dumps(
                {
                    "methods": METHODS,
                    "steps": args.steps,
                    "tasks": 8,
                    "starts": 4,
                    "support": 256,
                    "query": 512,
                    "test": 4096,
                    "tuning": tuning,
                    "structural_lr": 0.03,
                    "structural_update_every": 4,
                    "frozen_support": "source-only majority over training indexators",
                    "calibration_reads_teacher": False,
                },
                indent=2,
            )
        )
        return
    spec = json.loads((args.out / "evaluation_protocol.json").read_text())
    observations = np.load(args.out / "target_data.npz")
    x, y = (torch.tensor(observations[key], device=device) for key in ["x", "y"])
    if args.stage == "evaluate":
        frozen = {
            name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
            for name in [
                "structure_bank.npz",
                "prior_split.npz",
                "learned_support.npz",
                "column_prior.pt",
                "global_prior.pt",
                "evaluation_protocol.json",
            ]
        }
        (args.out / "evaluation_frozen.json").write_text(json.dumps(frozen, indent=2))
        for method in spec["methods"]:
            if (args.out / f"{method}.npz").exists():
                continue
            reference = (
                method
                if method.startswith("dense") or method == "best_support"
                else "bank"
            )
            fixed = (
                np.load(args.out / "target_teacher.npz")["indices"]
                if method == "oracle"
                else None
            )
            data, history = fit(
                args.out,
                x,
                y,
                method,
                spec["starts"],
                spec["steps"],
                [spec["tuning"][reference]["lr"]],
                fixed,
            )
            np.savez_compressed(args.out / f"{method}.npz", **data)
            (args.out / f"{method}_history.json").write_text(json.dumps(history))
            choices = data["query_nmse"].reshape(8, 4).argmin(1)
            selected = data["test_nmse"].reshape(8, 4)[np.arange(8), choices]
            print("RESULT", method, float(selected.mean()), flush=True)
        (args.out / "evaluate_complete.json").write_text(json.dumps({"complete": True}))
    else:
        for method in spec["methods"]:
            if (
                method.startswith("dense")
                or (args.out / f"refit_{method}.npz").exists()
            ):
                continue
            original = np.load(args.out / f"{method}.npz")
            choices = original["query_nmse"].reshape(8, 4).argmin(1)
            indices = original["indices"][np.arange(8) * 4 + choices]
            data, history = fit(
                args.out,
                x,
                y,
                "fixed",
                4,
                spec["steps"],
                [
                    spec["tuning"][
                        "best_support" if method == "best_support" else "bank"
                    ]["lr"]
                ],
                indices,
                seed=283060,
            )
            np.savez_compressed(args.out / f"refit_{method}.npz", **data)
            (args.out / f"refit_{method}_history.json").write_text(json.dumps(history))
            choices = data["query_nmse"].reshape(8, 4).argmin(1)
            selected = data["test_nmse"].reshape(8, 4)[np.arange(8), choices]
            print("REFIT", method, float(selected.mean()), flush=True)
        (args.out / "refit_complete.json").write_text(json.dumps({"complete": True}))


if __name__ == "__main__":
    main()
