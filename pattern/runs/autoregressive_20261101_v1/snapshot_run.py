import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from pattern.autoregressive.model import IndexMADE
from pattern.codebook.bank import predictions, scores
from pattern.codebook.models import IndexVAE, index_probabilities
from pattern.datasets import input_table, labels_for, partitions


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_bank(plan, device):
    source = Path(plan["source_run"])
    bank = np.load(source / "bank.npz")
    extracted = np.load(source / "extracted.npz")
    consensus = extracted["indices"]
    seen = list(dict.fromkeys(consensus.flatten().tolist()))
    nonzero = [label for label in seen if label]
    mapping = np.zeros(5, dtype=np.int64)
    mapping[nonzero] = np.arange(1, len(nonzero) + 1)
    return (
        torch.tensor(mapping[extracted["targets"]].reshape(-1, 88), device=device),
        torch.tensor(bank["split_ids"], device=device),
        mapping,
        torch.tensor(mapping[consensus], device=device),
    )


def prior_metrics(logits, labels):
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    prediction = logits.argmax(-1)
    active = labels != 0
    recall = ((prediction != 0) & active).sum() / active.sum().clamp_min(1)
    return {"nll_per_index": float(loss), "support_recall": float(recall)}


def train_prior(plan, output, device):
    spec = plan["prior"]
    labels, roles, mapping, consensus = load_bank(plan, device)
    torch.manual_seed(plan["seed"])
    model = IndexMADE(width=spec["width"]).to(device)
    torch.save(model.state_dict(), output / "untrained_ar.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
    train = labels[roles == 0]
    validation = labels[roles == 1]
    history = []
    best = float("inf")
    for step in range(spec["steps"] + 1):
        if step % spec["checkpoint_every"] == 0:
            model.eval()
            with torch.no_grad():
                training = prior_metrics(model(train), train)
                held_out = prior_metrics(model(validation), validation)
            history.append({"step": step, "train": training, "validation": held_out})
            if held_out["nll_per_index"] < best:
                best = held_out["nll_per_index"]
                torch.save(model.state_dict(), output / "prior.pt")
            print("PRIOR", step, training, held_out, flush=True)
        if step == spec["steps"]:
            break
        model.train()
        batch = train[torch.randint(len(train), (spec["batch_size"],), device=device)]
        logits = model(batch)
        loss = F.cross_entropy(logits.flatten(0, 1), batch.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.load_state_dict(torch.load(output / "prior.pt", weights_only=True))
    model.eval().requires_grad_(False)
    frequency = F.one_hot(train, 5).float().sum(0) + 0.01
    frequency /= frequency.sum(-1, keepdim=True)
    with torch.no_grad():
        likelihood_audit = {
            "ar": prior_metrics(model(labels[roles == 2]), labels[roles == 2]),
            "frequency": prior_metrics(
                frequency.log()[None].expand(int((roles == 2).sum()), -1, -1),
                labels[roles == 2],
            ),
        }
    np.savez_compressed(
        output / "bank_inputs.npz",
        labels=labels.cpu(),
        roles=roles.cpu(),
        mapping=mapping,
        consensus=consensus.cpu(),
        frequency=frequency.cpu(),
    )
    save_json(output / "prior_history.json", history)
    save_json(output / "likelihood_audit.json", likelihood_audit)
    return model, train, frequency, mapping, consensus


def sample_proposals(plan, model, train, frequency, mapping, consensus, count, device):
    source_plan = json.loads(Path(plan["source_protocol"]).read_text())
    vae = IndexVAE(len(source_plan["source_patterns"]), source_plan["vae"]).to(device)
    vae.load_state_dict(
        torch.load(Path(plan["source_run"]) / "prior.pt", weights_only=True)
    )
    vae.eval().requires_grad_(False)
    untrained = IndexMADE(width=plan["prior"]["width"]).to(device)
    untrained.load_state_dict(
        torch.load(Path(plan["run"]) / "untrained_ar.pt", weights_only=True)
    )
    untrained.eval().requires_grad_(False)
    label_mapping = torch.tensor(mapping, device=device)
    with torch.no_grad():
        torch.manual_seed(plan["seed"] + 10)
        ar = model.sample(count)
        torch.manual_seed(plan["seed"] + 11)
        independent = torch.multinomial(frequency, count, replacement=True).T
        uniform = frequency.mean(0).expand(88, -1)
        random = torch.multinomial(uniform, count, replacement=True).T
        chosen = train[torch.randint(len(train), (count,), device=device)]
        empty_ar = untrained.sample(count)
        latent = torch.randn(count, source_plan["vae"]["latent_dim"], device=device)
        vae_indices = index_probabilities(vae.decode(latent)).argmax(-1)
        vae_indices = label_mapping[vae_indices].flatten(1)
    proposals = {
        "autoregressive": ar,
        "frequency": independent,
        "bank": chosen,
        "uniform": random,
        "untrained_ar": empty_ar,
        "vae_sample": vae_indices,
        "vae_z": vae_indices,
        "inferred": consensus.flatten().expand(count, -1),
    }
    return proposals, vae, latent, label_mapping, source_plan


def fit_method(plan, method, indices, vae, latent, mapping, patterns, split, device):
    spec = plan["fit"]
    count = len(patterns) * spec["restarts"]
    _, inputs = input_table()
    inputs = inputs.to(device)
    target = torch.stack([labels_for(inputs, p) for p in patterns])
    target = target.repeat_interleave(spec["restarts"], 0)
    torch.manual_seed(plan["seed"] + 20)
    numeric = torch.zeros(count, 7, device=device)
    code = torch.randn(count, 4, device=device)
    numeric[:, :4] = F.normalize(code, dim=1) * spec["codebook_initial_norm"]
    numeric[:, 5] = 1
    numeric.requires_grad_()
    indices = indices.reshape(count, 11, 8).clone()
    latent = latent.detach().clone()
    dense = method.startswith("dense_")
    if dense:
        torch.manual_seed(plan["seed"] + 21)
        parameters = [
            torch.randn(count, 11, 8, device=device) * 0.1,
            torch.zeros(count, 8, device=device),
            torch.randn(count, 8, device=device) * 0.1,
            torch.zeros(count, device=device),
        ]
        for parameter in parameters:
            parameter.requires_grad_()
        groups = [{"params": parameters, "lr": spec[method + "_learning_rate"]}]
    else:
        groups = [{"params": [numeric], "lr": spec["codebook_learning_rate"]}]
    if method == "vae_z":
        latent.requires_grad_()
        groups.append({"params": [latent], "lr": 0.03})
    optimizer_class = torch.optim.SGD if method == "dense_sgd" else torch.optim.Adam
    optimizer = optimizer_class(groups)

    def current():
        if dense:
            return parameters, indices
        if method == "vae_z":
            probabilities = index_probabilities(
                vae.decode(latent), straight_through=True
            )
            inverse = torch.argsort(mapping)
            probabilities = probabilities[..., inverse]
            weights = (probabilities[..., 1:] * numeric[:, None, None, :4]).sum(-1)
            current_indices = probabilities.detach().argmax(-1)
        else:
            codebook = F.pad(numeric[:, :4], (1, 0))
            weights = codebook.gather(1, indices.flatten(1)).reshape(-1, 11, 8)
            current_indices = indices
        return [
            weights,
            numeric[:, 4:5].expand(-1, 8),
            numeric[:, 5:6].expand(-1, 8),
            numeric[:, 6],
        ], current_indices

    best_loss = torch.full((count,), torch.inf, device=device)
    best = [parameter.detach().clone() for parameter in current()[0]]
    best_indices = indices.clone()
    best_numeric = numeric.detach().clone()
    best_latent = latent.detach().clone()
    best_step = torch.zeros(count, dtype=torch.long, device=device)
    history = []
    started = time.time()
    for step in range(spec["steps"] + 1):
        if step % spec["checkpoint_every"] == 0:
            with torch.no_grad():
                parameters_now, indices_now = current()
                query = scores(
                    predictions(parameters_now, inputs[split["query"]]),
                    target[:, split["query"]],
                )
                better = query["loss"] < best_loss
                best_loss[better] = query["loss"][better]
                for stored, parameter in zip(best, parameters_now):
                    stored[better] = parameter[better]
                best_indices[better] = indices_now[better]
                best_numeric[better] = numeric[better]
                best_latent[better] = latent[better]
                best_step[better] = step
        parameters_now, _ = current()
        training = scores(
            predictions(parameters_now, inputs[split["support"]]),
            target[:, split["support"]],
            balanced=True,
        )
        if step % spec["log_every"] == 0:
            history.append(
                {
                    "step": step,
                    "train": {
                        key: value.detach().cpu().tolist()
                        for key, value in training.items()
                    },
                    "query": {
                        key: value.cpu().tolist() for key, value in query.items()
                    },
                }
            )
            print(
                "FIT",
                method,
                step,
                "train",
                float(training["loss"].detach().mean()),
                "query",
                float(best_loss.mean()),
                flush=True,
            )
        if step == spec["steps"]:
            break
        objective = training["loss"].sum()
        if method == "vae_z":
            objective = objective + 0.0001 * latent.square().sum()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
    with torch.no_grad():
        result = {}
        for name, ids in [
            ("query", split["query"]),
            ("test", split["test"]),
            ("population", torch.arange(2048, device=device)),
        ]:
            values = scores(predictions(best, inputs[ids]), target[:, ids])
            result.update(
                {name + "_" + key: value.cpu().numpy() for key, value in values.items()}
            )
    result.update(
        {f"parameter_{i}": value.cpu().numpy() for i, value in enumerate(best)}
    )
    result.update(
        indices=best_indices.cpu().numpy(),
        numeric=best_numeric.cpu().numpy(),
        latent=best_latent.cpu().numpy(),
        step=best_step.cpu().numpy(),
    )
    np.savez_compressed(Path(plan["run"]) / (method + ".npz"), **result)
    save_json(Path(plan["run"]) / (method + "_history.json"), history)
    save_json(
        Path(plan["run"]) / (method + "_complete.json"),
        {"seconds": time.time() - started},
    )
    print(
        "COMPLETE",
        method,
        "population",
        float(result["population_accuracy"].mean()),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="pattern/autoregressive/protocol.json")
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    output = Path(plan["run"])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    files = [
        Path(args.protocol),
        Path(plan["source_protocol"]),
        *Path("pattern/autoregressive").glob("*.py"),
        *Path("pattern/codebook").glob("*.py"),
        Path("pattern/datasets.py"),
        Path(plan["source_run"]) / "bank.npz",
        Path(plan["source_run"]) / "extracted.npz",
        Path(plan["source_run"]) / "prior.pt",
    ]
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    save_json(
        output / "frozen_protocol.json",
        {
            **plan,
            "hashes": hashes,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
    )
    for path in Path("pattern/autoregressive").glob("*.py"):
        (output / ("snapshot_" + path.name)).write_bytes(path.read_bytes())
    started = time.time()
    model, train, frequency, mapping, consensus = train_prior(plan, output, device)
    source_plan = json.loads(Path(plan["source_protocol"]).read_text())
    patterns = source_plan["target_patterns"]
    count = len(patterns) * plan["fit"]["restarts"]
    proposals, vae, latent, label_mapping, _ = sample_proposals(
        plan, model, train, frequency, mapping, consensus, count, device
    )
    np.savez_compressed(
        output / "proposals.npz",
        **{key: value.cpu().numpy() for key, value in proposals.items()},
    )
    split = {key: value.to(device) for key, value in partitions(plan["seed"]).items()}
    np.savez_compressed(
        output / "splits.npz",
        **{key: value.cpu().numpy() for key, value in split.items()},
    )
    for method in plan["fit"]["methods"]:
        if method == "oracle":
            from pattern.codebook.fit import analytic_indices

            indices = analytic_indices(device).flatten().expand(count, -1)
        else:
            indices = proposals.get(
                method, torch.zeros(count, 88, dtype=torch.long, device=device)
            )
        fit_method(
            plan, method, indices, vae, latent, label_mapping, patterns, split, device
        )
    save_json(
        output / "complete.json",
        {"seconds": time.time() - started, "patterns": patterns},
    )


if __name__ == "__main__":
    main()
