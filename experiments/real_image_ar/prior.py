"""Learn source-mask distributions with a shared cardinality-only decoder."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pattern.autoregressive.model import IndexMADE


def forced_positions(labels, budget):
    preceding = labels.cumsum(-1) - labels
    remaining = budget - preceding
    left = labels.shape[-1] - torch.arange(labels.shape[-1], device=labels.device)
    return (remaining == 0) | (remaining == left)


def prior_scores(logits, labels, budget):
    forced = forced_positions(labels, budget)
    loss = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), reduction="none"
    ).reshape_as(labels)
    loss = loss.masked_fill(forced, 0).mean()
    predicted = torch.where(forced, labels, logits.argmax(-1))
    recall = ((predicted == 1) & (labels == 1)).sum() / (labels == 1).sum()
    return {
        "nll_per_position": float(loss),
        "teacher_forced_support_recall": float(recall),
    }


def train_prior(plan, name, device):
    root = Path(plan["run"]) / name
    if (root / "prior.json").exists():
        return
    bank = np.load(root / "bank.npz")
    order = np.random.default_rng(plan["seed"] + 300).permutation(plan["inputs"])
    labels = torch.tensor(bank["mask"][:, order].astype(np.int64), device=device)
    training = labels[bank["role"] == 0]
    validation = labels[bank["role"] == 1]
    counts = F.one_hot(training, 2).float().sum(0) + 0.01
    probabilities = counts / counts.sum(-1, keepdim=True)
    np.savez_compressed(
        root / "prior_inputs.npz",
        order=order,
        probabilities=probabilities.cpu().numpy(),
    )
    spec = plan["prior"]
    candidates, histories = [], []
    global_best = float("inf")
    for width in spec["widths"]:
        for rate in spec["learning_rates"]:
            for decay in spec["weight_decays"]:
                torch.manual_seed(plan["seed"] + 310)
                model = IndexMADE(plan["inputs"], 2, width).to(device)
                initial = {
                    key: value.clone() for key, value in model.state_dict().items()
                }
                with torch.no_grad():
                    model.layers[4].weight.zero_()
                    model.layers[4].bias.copy_(probabilities.log().flatten())
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=rate, weight_decay=decay
                )
                history, best, best_step = [], float("inf"), 0
                for step in range(spec["steps"] + 1):
                    if step % spec["checkpoint_every"] == 0:
                        with torch.no_grad():
                            train = prior_scores(
                                model(training), training, plan["active_features"]
                            )
                            val = prior_scores(
                                model(validation), validation, plan["active_features"]
                            )
                        history.append(
                            {"step": step, "train": train, "validation": val}
                        )
                        if val["nll_per_position"] < best:
                            best, best_step = val["nll_per_position"], step
                        if val["nll_per_position"] < global_best:
                            global_best = val["nll_per_position"]
                            torch.save(model.state_dict(), root / "prior.pt")
                            torch.save(initial, root / "untrained_ar.pt")
                            chosen = {
                                "width": width,
                                "learning_rate": rate,
                                "weight_decay": decay,
                                "step": step,
                            }
                    if step == spec["steps"]:
                        break
                    batch = training[
                        torch.randint(
                            len(training), (spec["batch_size"],), device=device
                        )
                    ]
                    logits = model(batch)
                    forced = forced_positions(batch, plan["active_features"])
                    loss = (
                        F.cross_entropy(
                            logits.flatten(0, 1), batch.flatten(), reduction="none"
                        )
                        .reshape_as(batch)
                        .masked_fill(forced, 0)
                        .mean()
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                candidates.append(
                    {
                        "width": width,
                        "lr": rate,
                        "weight_decay": decay,
                        "best_step": best_step,
                        "validation_nll": best,
                    }
                )
                histories.append({"config": candidates[-1], "history": history})
                print("PRIOR", name, candidates[-1], flush=True)
    model = IndexMADE(plan["inputs"], 2, chosen["width"]).to(device)
    model.load_state_dict(torch.load(root / "prior.pt", weights_only=True))
    held_out = labels[bank["role"] == 2]
    with torch.no_grad():
        audit = prior_scores(model(held_out), held_out, plan["active_features"])
        baseline = prior_scores(
            probabilities.log()[None].expand(len(held_out), -1, -1),
            held_out,
            plan["active_features"],
        )
    (root / "prior_history.json").write_text(json.dumps(histories) + "\n")
    (root / "prior.json").write_text(
        json.dumps(
            {
                "chosen": chosen,
                "candidates": candidates,
                "held_source_audit": audit,
                "frequency_held_source_audit": baseline,
                "role_counts": [int((bank["role"] == r).sum()) for r in range(3)],
                "budget": plan["active_features"],
                "order_seed": plan["seed"] + 300,
            },
            indent=2,
        )
        + "\n"
    )


@torch.no_grad()
def sample(plan, name, method, temperature, count, device):
    root = Path(plan["run"]) / name
    bank = np.load(root / "bank.npz")
    inputs = np.load(root / "prior_inputs.npz")
    order = inputs["order"]
    probabilities = torch.tensor(inputs["probabilities"], device=device)
    budget, positions = plan["active_features"], plan["inputs"]
    torch.manual_seed(plan["seed"] + 320)
    if method == "bank":
        masks = torch.tensor(bank["mask"][bank["role"] == 0], device=device)
        return masks[torch.randint(len(masks), (count,), device=device)].cpu().numpy()
    if method == "random":
        selected = torch.rand(count, positions, device=device).topk(budget, -1).indices
        return (
            torch.zeros(count, positions, device=device)
            .scatter_(1, selected, 1)
            .cpu()
            .numpy()
        )
    if method == "consensus":
        ordered = torch.zeros(count, positions, device=device)
        ordered[:, probabilities[:, 1].topk(budget).indices] = 1
    else:
        model = None
        if method in ["ar", "untrained_ar"]:
            width = json.loads((root / "prior.json").read_text())["chosen"]["width"]
            model = IndexMADE(positions, 2, width).to(device)
            filename = "prior.pt" if method == "ar" else "untrained_ar.pt"
            model.load_state_dict(torch.load(root / filename, weights_only=True))
            first, second, final = model.layers[0], model.layers[2], model.layers[4]
            first_weight = first.weight * first.mask
            second_weight = second.weight * second.mask
            final_weight = final.weight * final.mask
            activation = (
                (first_weight[:, ::2].sum(-1) + first.bias)[None]
                .expand(count, -1)
                .clone()
            )
        ordered = torch.zeros(count, positions, dtype=torch.long, device=device)
        remaining = torch.full((count,), budget, device=device)
        for position in range(positions):
            if model is None:
                logits = probabilities[position].log()[None].expand(count, -1)
            else:
                hidden = F.gelu(
                    F.linear(F.gelu(activation), second_weight, second.bias)
                )
                logits = F.linear(
                    hidden,
                    final_weight[2 * position : 2 * position + 2],
                    final.bias[2 * position : 2 * position + 2],
                )
            chosen = (
                logits.argmax(-1)
                if temperature == 0
                else torch.multinomial((logits / temperature).softmax(-1), 1).squeeze(1)
            )
            chosen = torch.where(remaining == 0, 0, chosen)
            chosen = torch.where(remaining == positions - position, 1, chosen)
            ordered[:, position] = chosen
            remaining -= chosen
            if model is not None:
                activation += (
                    first_weight[:, 2 * position + chosen].T
                    - first_weight[:, 2 * position][None]
                )
        if model is not None and temperature == 0:
            logits = model(ordered)
            free = ~forced_positions(ordered, budget)
            if ((ordered != logits.argmax(-1)) & free).any():
                raise RuntimeError("Cached greedy decoding disagrees with full MADE")
    result = torch.zeros_like(ordered)
    result[:, torch.tensor(order, device=device)] = ordered
    if not (result.sum(-1) == budget).all():
        raise RuntimeError("Sampled feature budget does not match protocol")
    return result.cpu().numpy()


def main():
    from experiments.real_image_ar.common import load_plan

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    train_prior(load_plan(), args.dataset, torch.device("cuda"))


if __name__ == "__main__":
    main()
