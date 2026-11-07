"""Stronger multi-task extension of Yeh's categorical bilevel objective."""

import argparse
import itertools
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .linear import (
    Adam,
    adaptation_errors,
    adapted,
    index_basis,
    load_observations,
    prediction,
)
from .run import execution_snapshot

NAMES = ("yeh_shared_soft", "yeh_shared_map")


def configurations():
    rows = []
    for lr, (entropy, nuclear), (zero_bias, temperature) in itertools.product(
        (0.003, 0.01, 0.03),
        ((0.0, 0.0), (0.1, 0.0), (0.01, 0.01)),
        ((False, 1.0), (True, 1.0), (False, 0.1), (False, 0.03)),
    ):
        rows.append(
            {
                "configuration": len(rows),
                "lr": lr,
                "entropy": entropy,
                "nuclear": nuclear,
                "zero_bias": zero_bias,
                "minimum_temperature": temperature,
            }
        )
    return rows


def train(rows, spec, plan, data, seed):
    device = data["train_support_x"].device
    count, codes = len(rows), spec["codes"]
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 100003 * seed)
    logits = 0.01 * torch.randn(
        (1, spec["outputs"], spec["inputs"], 2 * codes + 1),
        device=device,
        generator=generator,
    )
    logits = logits.repeat(count, 1, 1, 1)
    logits[..., 0] += logits.new_tensor(
        [np.log(2 * codes) if row["zero_bias"] else 0 for row in rows]
    )[:, None, None]
    logits.requires_grad_()
    minimum_temperature = logits.new_tensor(
        [row["minimum_temperature"] for row in rows]
    )
    rates = logits.new_tensor([row["lr"] for row in rows])
    entropy = logits.new_tensor([row["entropy"] for row in rows])
    nuclear = logits.new_tensor([row["nuclear"] for row in rows])
    optimizer = Adam([logits], rates)
    best = {name: logits.new_full((count,), torch.inf) for name in NAMES}
    saved = {name: None for name in NAMES}
    steps = {
        name: torch.zeros(count, device=device, dtype=torch.long) for name in NAMES
    }
    history = {name: [] for name in NAMES}
    started = time.perf_counter()
    for step in range(spec["gradient_steps"] + 1):
        cooling = max(0.0, (step / spec["gradient_steps"] - 0.3) / 0.7)
        temperature = minimum_temperature.pow(cooling)
        probability = (logits / temperature[:, None, None, None]).softmax(-1)
        basis = probability[..., 1 : codes + 1] - probability[..., codes + 1 :]
        coefficient = adapted(
            basis, data["train_support_x"], data["train_support_y"], ridge=1e-4
        )
        loss = (
            (
                prediction(basis, coefficient, data["train_query_x"])
                - data["train_query_y"][None]
            )
            .square()
            .mean((1, 2, 3))
        )
        if step % spec["checkpoint_every"] == 0 or step == spec["gradient_steps"]:
            labels = probability.detach().argmax(-1)
            index = torch.where(labels > codes, codes - labels, labels)
            for name, representation, local_basis in (
                (NAMES[0], basis.detach(), basis.detach()),
                (NAMES[1], index, index_basis(index, codes)),
            ):
                selection = adaptation_errors(
                    local_basis,
                    data["selection_support_x"],
                    data["selection_support_y"],
                    data["selection_query_x"],
                    data["selection_query_y"],
                ).mean(-1)
                source = adaptation_errors(
                    local_basis,
                    data["train_support_x"],
                    data["train_support_y"],
                    data["train_query_x"],
                    data["train_query_y"],
                ).mean(-1)
                improved = selection < best[name]
                if saved[name] is None:
                    saved[name] = representation.clone()
                saved[name][improved] = representation[improved]
                steps[name][improved] = step
                best[name] = torch.minimum(best[name], selection)
                history[name].append(
                    {
                        "step": step,
                        "train_mse": loss.detach().tolist(),
                        "source_query_mse": source.tolist(),
                        "selection_mse": selection.tolist(),
                        "best_selection_mse": best[name].tolist(),
                        "seconds": time.perf_counter() - started,
                    }
                )
            print(
                "shared Yeh",
                step,
                {name: float(best[name].min()) for name in NAMES},
                flush=True,
            )
        if step == spec["gradient_steps"]:
            break
        ent = -(probability * probability.clamp_min(1e-12).log()).sum(-1).mean((-1, -2))
        nuclear_loss = torch.zeros_like(loss)
        selected = nuclear > 0
        if selected.any():
            nuclear_loss[selected] = (
                nuclear[selected]
                * torch.linalg.svdvals(probability[selected].flatten(1, 2)).sum(-1)
                / np.sqrt(spec["inputs"] * spec["outputs"])
            )
        optimizer.step(
            torch.autograd.grad((loss + entropy * ent + nuclear_loss).sum(), [logits])
        )
        with torch.no_grad():
            logits.sub_(logits.mean(-1, keepdim=True))
    return {
        name: {
            "basis": (
                saved[name]
                if name.endswith("soft")
                else index_basis(saved[name], codes)
            )
            .cpu()
            .numpy(),
            **({"indices": saved[name].cpu().numpy()} if name.endswith("map") else {}),
            "selection_mse": best[name].cpu().numpy(),
            "step": steps[name].cpu().numpy(),
            "history": history[name],
        }
        for name in NAMES
    }, time.perf_counter() - started


def run_one(root, profile, family, phase, seed, rows, device):
    folder = root / profile / f"family_{family}"
    output = folder / phase
    if all((output / f"{name}_seed_{seed}.json").exists() for name in NAMES):
        return
    plan = json.loads((root / "protocol.json").read_text())
    spec = json.loads((folder / "specification.json").read_text())
    snapshot, hashes = execution_snapshot(root)
    results, seconds = train(rows, spec, plan, load_observations(folder, device), seed)
    output.mkdir(exist_ok=True)
    for i, (name, result) in enumerate(results.items()):
        path = output / f"{name}_seed_{seed}.npz"
        np.savez_compressed(
            path,
            **{
                key: value
                for key, value in result.items()
                if isinstance(value, np.ndarray)
            },
        )
        save_json(
            path.with_suffix(".json"),
            {
                "inputs": {
                    "method": name,
                    "seed": seed,
                    "rows": rows,
                    "observation_sha256": digest(folder / "observations.npz"),
                },
                "seconds": seconds if i == 0 else 0.0,
                "history": result["history"],
                "execution_snapshot": snapshot,
                "executed_sha256": hashes,
                "result_sha256": digest(path),
                "completed_utc": datetime.now(UTC).isoformat(),
                "adaptation": "One P shared across all source coefficient tasks; exact differentiable C_t support-fit, mean source-query upper objective. Same fixed-zero signed dictionary. This is an explicit multi-task extension, not the author's original single-task experiment.",
            },
        )


def freeze(root, profile, families, phase, seeds):
    choices = {}
    for name in NAMES:
        scores = []
        for family in families:
            for seed in seeds:
                with np.load(
                    root
                    / profile
                    / f"family_{family}"
                    / phase
                    / f"{name}_seed_{seed}.npz"
                ) as archive:
                    scores.append(archive["selection_mse"])
        mean = np.mean(scores, axis=0)
        chosen = int(mean.argmin())
        choices[name] = {
            "row": chosen,
            "settings": configurations()[chosen],
            "mean_selection_mse": float(mean[chosen]),
            "all_mean_selection_mse": mean.tolist(),
        }
    save_json(
        root / f"{profile}_shared_hpo_frozen.json",
        {"frozen_utc": datetime.now(UTC).isoformat(), "choices": choices},
    )
    return choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--stage", required=True, choices=("pilot", "confirmation", "native")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((args.root / "protocol.json").read_text())
    device = torch.device(args.device)
    if args.stage == "pilot":
        for family in plan["pilot_families"]:
            for seed in plan["pilot_seeds"]:
                run_one(
                    args.root,
                    "hidden",
                    family,
                    "shared_pilot",
                    seed,
                    configurations(),
                    device,
                )
        freeze(
            args.root,
            "hidden",
            plan["pilot_families"],
            "shared_pilot",
            plan["pilot_seeds"],
        )
    else:
        profile = "native" if args.stage == "native" else "hidden"
        if profile == "native":
            run_one(
                args.root,
                profile,
                301,
                "shared_calibration",
                1,
                configurations(),
                device,
            )
            choices = freeze(args.root, profile, [301], "shared_calibration", [1])
        else:
            choices = json.loads(
                (args.root / "hidden_shared_hpo_frozen.json").read_text()
            )["choices"]
        # Retain both independently selected settings if soft and MAP differ.
        rows = list(
            {
                choice["settings"]["configuration"]: choice["settings"]
                for choice in choices.values()
            }.values()
        )
        for family in [301] if profile == "native" else plan["confirmation_families"]:
            for seed in plan["confirmation_seeds"]:
                run_one(
                    args.root,
                    profile,
                    family,
                    "shared_confirmation",
                    seed,
                    rows,
                    device,
                )
        save_json(
            args.root / f"{profile}_shared_models_frozen.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "row_by_method": {
                    name: next(
                        i
                        for i, row in enumerate(rows)
                        if row["configuration"] == choice["settings"]["configuration"]
                    )
                    for name, choice in choices.items()
                },
                "sha256": {
                    str(path.relative_to(args.root)): digest(path)
                    for path in args.root.glob(
                        f"{profile}/family_*/shared_confirmation/*.npz"
                    )
                },
            },
        )


if __name__ == "__main__":
    main()
