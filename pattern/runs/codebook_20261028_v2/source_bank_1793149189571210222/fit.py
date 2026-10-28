"""Fit task codebooks with fixed or learned indexators, using disjoint patterns."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from pattern.codebook.bank import predictions, scores
from pattern.codebook.models import IndexVAE, index_probabilities
from pattern.datasets import input_table, labels_for


def analytic_indices(device):
    rows = torch.arange(11, device=device)[:, None]
    columns = torch.arange(8, device=device)[None]
    return torch.where((rows >= columns) & (rows < columns + 4), rows - columns + 1, 0)


def fit(plan, patterns, method, lr, norm, steps, restarts, device):
    run = Path(plan["run"])
    _, x = input_table()
    x = x.to(device)
    split = {
        k: torch.tensor(v, device=device)
        for k, v in np.load(run / "splits.npz").items()
    }
    target = torch.stack([labels_for(x, p) for p in patterns]).repeat_interleave(
        restarts, 0
    )
    count = len(target)
    torch.manual_seed(plan["seed"] + 19)
    model = IndexVAE(len(plan["source_patterns"]), plan["vae"]).to(device)
    filename = "untrained_prior.pt" if method == "untrained_vae_z" else "prior.pt"
    model.load_state_dict(torch.load(run / filename, weights_only=True))
    model.requires_grad_(False).eval()
    z = torch.randn(count, plan["vae"]["latent_dim"], device=device)
    c = torch.randn(count, 4, device=device)
    c = c / c.norm(dim=1, keepdim=True) * norm
    numeric = torch.cat(
        [
            c,
            torch.zeros(count, 1, device=device),
            torch.ones(count, 1, device=device),
            torch.zeros(count, 1, device=device),
        ],
        1,
    ).requires_grad_()
    with torch.no_grad():
        logits = model.decode(z)
        if method == "oracle_indices":
            indices = analytic_indices(device).expand(count, -1, -1)
        elif method == "inferred_indices":
            indices = torch.tensor(
                np.load(run / "extracted.npz")["indices"], device=device
            )
            indices = indices.expand(count, -1, -1)
        elif method == "random_indices":
            template = torch.cat(
                [
                    torch.zeros(56, device=device),
                    torch.arange(1, 5, device=device).repeat_interleave(8),
                ]
            )
            indices = (
                torch.stack(
                    [template[torch.randperm(88, device=device)] for _ in range(count)]
                )
                .long()
                .reshape(count, 11, 8)
            )
        else:
            indices = index_probabilities(logits).argmax(-1)
    groups = [{"params": [numeric], "lr": lr}]
    if method in ("vae_z", "untrained_vae_z"):
        z.requires_grad_()
        groups.append({"params": [z], "lr": plan["fit"]["z_learning_rate"]})
    if method == "direct_indices":
        logits = logits.detach().requires_grad_()
        groups.append(
            {"params": [logits], "lr": plan["fit"]["direct_index_learning_rate"]}
        )
    if method in ("dense_adam", "dense_sgd"):
        torch.manual_seed(plan["seed"] + 20)
        dense = [
            torch.randn(count, 11, 8, device=device) * 0.1,
            torch.zeros(count, 8, device=device),
            torch.randn(count, 8, device=device) * 0.1,
            torch.zeros(count, device=device),
        ]
        for parameter in dense:
            parameter.requires_grad_()
        groups = [{"params": dense, "lr": lr}]
    optimizer_type = torch.optim.SGD if method == "dense_sgd" else torch.optim.Adam
    optimizer = optimizer_type(groups)

    def current():
        if method in ("dense_adam", "dense_sgd"):
            return dense
        if method in ("vae_z", "untrained_vae_z"):
            assignments = index_probabilities(model.decode(z), straight_through=True)
        elif method == "direct_indices":
            assignments = index_probabilities(logits, straight_through=True)
        else:
            assignments = torch.nn.functional.one_hot(indices, 5).float()
        weights = (assignments[..., 1:] * numeric[:, None, None, :4]).sum(-1)
        return [
            weights,
            numeric[:, 4:5].expand(-1, 8),
            numeric[:, 5:6].expand(-1, 8),
            numeric[:, 6],
        ]

    best_loss = torch.full((count,), torch.inf, device=device)
    best = [p.detach().clone() for p in current()]
    best_step = torch.zeros(count, device=device, dtype=torch.int64)
    best_indices = indices.clone()
    best_numeric = numeric.detach().clone()
    history = []
    for step in range(steps + 1):
        if step % plan["fit"]["checkpoint_every"] == 0:
            with torch.no_grad():
                parameters = current()
                query = scores(
                    predictions(parameters, x[split["query"]]),
                    target[:, split["query"]],
                )
                better = query["loss"] < best_loss
                best_loss[better] = query["loss"][better]
                best_step[better] = step
                for stored, parameter in zip(best, parameters):
                    stored[better] = parameter[better]
                if method in ("vae_z", "untrained_vae_z"):
                    indices = index_probabilities(model.decode(z)).argmax(-1)
                elif method == "direct_indices":
                    indices = index_probabilities(logits).argmax(-1)
                best_indices[better] = indices[better]
                best_numeric[better] = numeric[better]
        if step % plan["fit"]["log_every"] == 0:
            history.append(
                {
                    "step": step,
                    "query_bce": query["loss"].cpu().tolist(),
                    "query_accuracy": query["accuracy"].cpu().tolist(),
                }
            )
        if step == steps:
            break
        loss = scores(
            predictions(current(), x[split["support"]]),
            target[:, split["support"]],
            balanced=True,
        )["loss"].sum()
        if method in ("vae_z", "untrained_vae_z"):
            loss = loss + plan["fit"]["z_penalty"] * z.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        metrics = {}
        for name, ids in [
            ("query", split["query"]),
            ("test", split["test"]),
            ("population", torch.arange(2048, device=device)),
        ]:
            values = scores(predictions(best, x[ids]), target[:, ids])
            metrics[name] = {key: value.cpu().numpy() for key, value in values.items()}
    return (
        metrics,
        history,
        {
            **{f"parameter_{i}": p.cpu().numpy() for i, p in enumerate(best)},
            "indices": best_indices.cpu().numpy(),
            "numeric": best_numeric.cpu().numpy(),
            "step": best_step.cpu().numpy(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["tune", "evaluate"], required=True)
    parser.add_argument("--protocol", default="pattern/codebook/protocol.json")
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    spec = plan["fit"]
    output = Path(plan["run"])
    snapshot = output / ("source_fit_" + str(time.time_ns()))
    snapshot.mkdir()
    for file in Path("pattern/codebook").glob("*"):
        if file.is_file():
            shutil.copy2(file, snapshot / file.name)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    started = time.time()
    if args.stage == "tune":
        results = []
        for method, rates, norms in [
            (
                "inferred_indices",
                spec["codebook_learning_rates"],
                spec["codebook_initial_norms"],
            ),
            ("dense_adam", spec["dense_learning_rates"], [1.0]),
            ("dense_sgd", [0.01, 0.03, 0.1, 0.3], [1.0]),
        ]:
            for lr in rates:
                for norm in norms:
                    metrics, _, _ = fit(
                        plan,
                        plan["source_patterns"],
                        method,
                        lr,
                        norm,
                        spec["tuning_steps"],
                        spec["tuning_restarts"],
                        device,
                    )
                    bce = metrics["query"]["loss"].reshape(-1, spec["tuning_restarts"])
                    result = {
                        "method": method,
                        "lr": lr,
                        "norm": norm,
                        "query_bce": float(bce.min(1).mean()),
                    }
                    results.append(result)
                    print("TUNE", result, flush=True)
                    (output / "tuning.json").write_text(json.dumps(results, indent=2))
    else:
        tuning = json.loads((output / "tuning.json").read_text())
        methods = [*spec["methods"], "dense_sgd"]
        for method in methods:
            if (output / (method + ".npz")).exists():
                continue
            source_method = (
                method if method in ("dense_adam", "dense_sgd") else "inferred_indices"
            )
            option = min(
                (r for r in tuning if r["method"] == source_method),
                key=lambda r: r["query_bce"],
            )
            metrics, history, parameters = fit(
                plan,
                plan["target_patterns"],
                method,
                option["lr"],
                option["norm"],
                spec["evaluation_steps"],
                spec["evaluation_restarts"],
                device,
            )
            arrays = {
                f"{split}_{key}": value
                for split, values in metrics.items()
                for key, value in values.items()
            }
            np.savez_compressed(output / (method + ".npz"), **arrays, **parameters)
            (output / (method + "_history.json")).write_text(json.dumps(history))
            print(
                "EVALUATE",
                method,
                "population_acc",
                float(metrics["population"]["accuracy"].mean()),
                flush=True,
            )
    (output / (args.stage + "_complete.json")).write_text(
        json.dumps(
            {
                "seconds": time.time() - started,
                "stage": args.stage,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
