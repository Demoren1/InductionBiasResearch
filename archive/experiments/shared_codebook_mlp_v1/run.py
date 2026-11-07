"""Transfer arbitrary weight sharing, inferred from source networks only."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from model import predict as network_predict
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "v2"
ACTIVATION = "relu"
NOISE = 0.0


class IndexVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(512 * 8, 128), nn.GELU())
        self.mu = nn.Linear(128, 16)
        self.logvar = nn.Linear(128, 16)
        self.decoder = nn.Sequential(
            nn.Linear(16, 128), nn.GELU(), nn.Linear(128, 512 * 32)
        )

    def forward(self, indices, sample=True):
        body = self.body(self.embedding(indices.flatten(1)))
        mu, logvar = self.mu(body), self.logvar(body).clamp(-12, 8)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return self.decoder(z).reshape(-1, 512, 32), mu, logvar


def prior(device):
    bank = np.load(OUT / "bank.npz")
    # No observations, numeric weights or teacher structures enter this stage.
    indices, quality = bank["aligned_indices"], bank["quality"]
    training, validation = [], []
    for task in range(6):
        ids = np.arange(task * 4, (task + 1) * 4)
        ids = ids[np.argsort(quality[ids])]
        training.append(ids[0])
        validation.append(ids[1])
    train, val = np.array(training), np.array(validation)
    consensus = np.stack([(indices[train] == k).sum(0) for k in range(32)]).argmax(0)
    np.savez_compressed(
        OUT / "structure_bank.npz",
        indices=indices,
        train=train,
        validation=val,
        consensus=consensus,
    )
    inputs = torch.tensor(indices, device=device)
    torch.manual_seed(282010)
    model = IndexVAE().to(device)
    torch.save(model.state_dict(), OUT / "untrained_prior.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, best = [], float("inf")
    for epoch in range(301):
        model.eval()
        with torch.no_grad():
            scores, _, _ = model(inputs[val], sample=False)
            validation_ce = F.cross_entropy(scores.flatten(0, 1), inputs[val].flatten())
            if float(validation_ce) < best:
                best = float(validation_ce)
                torch.save(model.state_dict(), OUT / "prior.pt")
        if epoch == 300:
            break
        model.train()
        for _ in range(4):
            ids = torch.tensor(train, device=device)[
                torch.randint(len(train), (16,), device=device)
            ]
            scores, mu, logvar = model(inputs[ids])
            ce = F.cross_entropy(scores.flatten(0, 1), inputs[ids].flatten())
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            loss = ce + 0.1 * min(1, (epoch + 1) / 50) * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "validation_ce": float(validation_ce),
                "training_ce": float(ce.detach()),
                "kl": float(kl.detach()),
            }
        )
    (OUT / "prior_history.json").write_text(json.dumps(history))
    (OUT / "prior_complete.json").write_text(
        json.dumps(
            {
                "validation_ce": best,
                "training_networks": train.tolist(),
                "validation_networks": val.tolist(),
                "stage_reads": [
                    "aligned indexators",
                    "source-fit quality for bank selection",
                ],
                "number_of_groups": None,
                "input_data_read": False,
                "decoder_cardinality_constraints": None,
            },
            indent=2,
        )
    )


def observations(seed, tasks, test_count, device):
    # This is only the benchmark generator; its teacher never enters the solver.
    teacher = np.load(OUT / "teacher.npz")["indexator"]
    torch.manual_seed(seed)
    c = torch.rand(tasks, 32, device=device) * 2 - 1
    x = torch.randn(128 + 1024 + test_count, 32, device=device)
    labels = network_predict(x, c[:, torch.tensor(teacher, device=device)], ACTIVATION)
    if NOISE:
        labels[:, :128] += (
            NOISE
            * labels[:, :128].std(1, unbiased=False)[:, None]
            * torch.randn_like(labels[:, :128])
        )
    return x, labels, c


def fit(x, y, method, rates, starts, steps, device, pretrained=None):
    tasks, count = len(y), len(y) * starts
    dense = method in ["dense_adam", "dense_sgd", "dense_transfer"]
    torch.manual_seed(282020)
    values = torch.randn(count, 512 if dense else 32, device=device) * 0.3
    if method == "dense_transfer":
        values = (
            pretrained[torch.arange(count, device=device) % len(pretrained)]
            .reshape(count, 512)
            .clone()
        )
    values.requires_grad_()
    optimizer = (
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
    labels = y.repeat_interleave(starts, 0)
    scale = labels[:, :128].var(1, unbiased=False).clamp_min(1e-8)
    bank = np.load(OUT / "structure_bank.npz")
    model, structural = None, None
    torch.manual_seed(282021)
    z = torch.randn(count, 16, device=device)
    if dense:
        indices = torch.arange(512, device=device)[None].expand(count, -1)
    elif method == "consensus":
        indices = torch.tensor(bank["consensus"].reshape(-1), device=device)[
            None
        ].expand(count, -1)
    elif method == "oracle":
        # Explicit diagnostic control; other structural solvers never read it.
        teacher_indices = np.load(OUT / "teacher.npz")["indexator"]
        indices = torch.tensor(teacher_indices.reshape(-1), device=device)[None].expand(
            count, -1
        )
    elif method == "bank_sample":
        choices = torch.tensor(bank["indices"][bank["train"]], device=device).flatten(1)
        indices = choices[torch.randint(len(choices), (count,), device=device)]
    elif method == "random":
        indices = torch.randint(32, (count, 512), device=device)
    elif method == "shuffled_consensus":
        base = torch.tensor(bank["consensus"].reshape(-1), device=device)
        indices = torch.stack(
            [base[torch.randperm(512, device=device)] for _ in range(count)]
        )
    else:
        model = IndexVAE().to(device)
        path = "untrained_prior.pt" if method == "untrained_vae_z" else "prior.pt"
        model.load_state_dict(
            torch.load(OUT / path, weights_only=True, map_location=device)
        )
        model.requires_grad_(False).eval()
        with torch.no_grad():
            indices = model.decoder(z).reshape(count, 512, 32).argmax(-1)
        if method in ["vae_z", "untrained_vae_z"]:
            z.requires_grad_()
            structural = torch.optim.Adam([z], lr=0.03)
    best_values, best_i, best_z = (
        values.detach().clone(),
        indices.clone(),
        z.detach().clone(),
    )
    best_loss = torch.full((count,), torch.inf, device=device)
    best_step = torch.zeros(count, dtype=torch.long, device=device)
    history = []

    def predict(batch, numbers, index, soft=None):
        weights = (
            numbers
            if dense
            else numbers.gather(1, index)
            if soft is None
            else (soft * numbers[:, None]).sum(-1)
        )
        return network_predict(batch, weights.reshape(count, 32, 16), ACTIVATION)

    for step in range(steps + 1):
        if step % 64 == 0:
            with torch.no_grad():
                if structural is not None:
                    indices = model.decoder(z).reshape(count, 512, 32).argmax(-1)
                error = torch.zeros(count, device=device)
                for ids in torch.arange(128, 1152, device=device).split(256):
                    error += (
                        (predict(x[ids], values, indices) - labels[:, ids])
                        .square()
                        .sum(1)
                        / 1024
                        / scale
                    )
                better = error < best_loss
                best_loss[better] = error[better]
                best_values[better] = values[better]
                best_i[better] = indices[better]
                best_z[better] = z[better]
                best_step[better] = step
                history.append({"step": step, "query_nmse": best_loss.cpu().tolist()})
        if step == steps:
            break
        soft = None
        update = structural is not None and step >= steps // 2 and step % 16 == 0
        if update:
            scores = model.decoder(z).reshape(count, 512, 32)
            indices = scores.detach().argmax(-1)
            probabilities = scores.softmax(-1)
            soft = (
                F.one_hot(indices, 32).float() + probabilities - probabilities.detach()
            )
        loss = (
            (predict(x[:128], values, indices, soft) - labels[:, :128]).square().mean(1)
            / scale
        ).sum()
        if update:
            loss = loss + 0.0001 * z.square().sum()
        optimizer.zero_grad(set_to_none=True)
        if structural is not None:
            structural.zero_grad(set_to_none=True)
        loss.backward()
        before = values.detach().clone()
        optimizer.step()
        with torch.no_grad():
            values.copy_(before + (values - before) * learning_rates[:, None])
        if update:
            structural.step()
    test_error = torch.zeros(count, device=device)
    if len(x) > 1152:
        with torch.no_grad():
            for ids in torch.arange(1152, len(x), device=device).split(256):
                test_error += (
                    (predict(x[ids], best_values, best_i) - labels[:, ids])
                    .square()
                    .sum(1)
                    / (len(x) - 1152)
                    / scale
                )
    return {
        "values": best_values.cpu().numpy(),
        "indices": best_i.cpu().numpy(),
        "z": best_z.cpu().numpy(),
        "step": best_step.cpu().numpy(),
        "query_nmse": best_loss.cpu().numpy(),
        "test_nmse": test_error.cpu().numpy(),
    }, history


def main():
    global OUT, ACTIVATION, NOISE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["prior", "calibrate", "evaluate"], required=True
    )
    parser.add_argument("--version", default="v2")
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--oracle-control", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.starts, args.tasks) <= 0:
        parser.error("--steps, --starts and --tasks must be positive")
    OUT = ROOT / "runs" / args.version
    source_protocol = json.loads((OUT / "protocol.json").read_text())
    ACTIVATION = source_protocol.get("activation", "relu")
    NOISE = source_protocol.get("noise_std_relative_to_clean_support", 0.0)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    shutil.copy2(__file__, OUT / ("run_" + args.stage + "_snapshot.py"))
    if args.stage == "prior":
        prior(device)
        return
    source = np.load(OUT / "bank.npz")
    best_source = [
        t * 4 + int(source["quality"][t * 4 : (t + 1) * 4].argmin()) for t in range(6)
    ]
    pretrained = torch.tensor(source["weights"][best_source], device=device)
    methods = [
        "consensus",
        "bank_sample",
        "vae_sample",
        "vae_z",
        "untrained_vae_z",
        "random",
        "shuffled_consensus",
        "dense_adam",
        "dense_sgd",
        "dense_transfer",
    ]
    if args.oracle_control:
        methods.append("oracle")
    if args.stage == "calibrate":
        x, y, c = observations(282030, 2, 0, device)
        np.savez_compressed(
            OUT / "calibration_data.npz",
            x=x.cpu().numpy(),
            y=y.cpu().numpy(),
            c=c.cpu().numpy(),
        )
        tuning = {}
        for method in ["consensus", "dense_adam", "dense_sgd", "dense_transfer"]:
            rates = (
                [0.01, 0.03, 0.1, 0.3]
                if method == "dense_sgd"
                else [0.001, 0.003, 0.01, 0.03]
            )
            repeated = [rate for rate in rates for _ in range(2)]
            data, history = fit(
                x, y, method, repeated, 8, args.steps, device, pretrained
            )
            scores = data["query_nmse"].reshape(2, 4, 2).min(-1).mean(0)
            tuning[method] = {
                "lr": rates[int(scores.argmin())],
                "source_query_grid": scores.tolist(),
            }
            np.savez_compressed(OUT / ("calibration_" + method + ".npz"), **data)
            (OUT / ("calibration_" + method + "_history.json")).write_text(
                json.dumps(history)
            )
            print("TUNE", method, tuning[method], flush=True)
        (OUT / "evaluation_protocol.json").write_text(
            json.dumps(
                {
                    "methods": methods,
                    "steps": args.steps,
                    "starts": args.starts,
                    "tasks": args.tasks,
                    "activation": ACTIVATION,
                    "noise_std_relative_to_clean_support": NOISE,
                    "support": 128,
                    "query": 1024,
                    "test": 8192,
                    "tuning": tuning,
                    "shared_teacher_layout": True,
                    "advantage_type": "transfer at small target support size, not proof of gradient local-minimum failure",
                },
                indent=2,
            )
        )
    else:
        spec = json.loads((OUT / "evaluation_protocol.json").read_text())
        methods = spec["methods"]
        frozen = {
            name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            for name in ["structure_bank.npz", "prior.pt", "evaluation_protocol.json"]
        }
        (OUT / "evaluation_frozen.json").write_text(json.dumps(frozen, indent=2))
        tasks, starts, steps = spec["tasks"], spec["starts"], spec["steps"]
        x, y, c = observations(282040, tasks, 8192, device)
        np.savez_compressed(
            OUT / "target_data.npz",
            x=x.cpu().numpy(),
            y=y.cpu().numpy(),
            c=c.cpu().numpy(),
        )
        for method in methods:
            if (OUT / (method + ".npz")).exists():
                continue
            reference = method if method.startswith("dense") else "consensus"
            rate = spec["tuning"][reference]["lr"]
            data, history = fit(x, y, method, [rate], starts, steps, device, pretrained)
            np.savez_compressed(OUT / (method + ".npz"), **data)
            (OUT / (method + "_history.json")).write_text(json.dumps(history))
            choice = data["query_nmse"].reshape(tasks, starts).argmin(1)
            selected = data["test_nmse"].reshape(tasks, starts)[
                np.arange(tasks), choice
            ]
            print(
                "RESULT",
                method,
                float(selected.mean()),
                int((data["test_nmse"] < 0.001).sum()),
                flush=True,
            )
        (OUT / "evaluate_complete.json").write_text(json.dumps({"complete": True}))


if __name__ == "__main__":
    main()
