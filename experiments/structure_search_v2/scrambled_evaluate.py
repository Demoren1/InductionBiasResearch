"""Held-out tasks for a learned large indexator, with source-calibrated controls."""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from nf_prior import NFVAE
from scrambled_pattern import teacher_labels
from scrambled_transfer import OUT, SOURCE, permutation_matrix


def numeric_fit(x, y, method, rates, steps, starts, device, pretrained=None):
    tasks, count = len(y), len(y) * starts
    dense = method in ["dense_adam", "dense_sgd", "dense_transfer"]
    torch.manual_seed(281330)
    if dense:
        parameters = [
            torch.randn(count, 64, 32, device=device) * 0.1,
            torch.zeros(count, 32, device=device),
            torch.randn(count, 32, device=device) * 0.1,
            torch.zeros(count, device=device),
        ]
    else:
        parameters = [
            torch.randn(count, 4, 8, device=device) * 0.5,
            torch.full((count, 4), -1.0, device=device),
            torch.ones(count, 4, device=device),
            torch.zeros(count, device=device),
        ]
    if method == "dense_transfer":
        source_ids = torch.arange(count, device=device) % 4
        parameters = [p[source_ids].clone() for p in pretrained]
        parameters[0] += torch.randn_like(parameters[0]) * 0.01
    parameters = [p.detach().requires_grad_() for p in parameters]
    labels = y.repeat_interleave(starts, 0)
    scale = labels[:, :8192].var(1, unbiased=False).clamp_min(1e-8)
    lr = (
        torch.tensor(rates, device=device).repeat(tasks)
        if len(rates) == starts
        else torch.full((count,), rates[0], device=device)
    )
    optimizer = (
        torch.optim.SGD(parameters, lr=1.0, momentum=0.9)
        if method == "dense_sgd"
        else torch.optim.Adam(parameters, lr=1.0)
    )
    torch.manual_seed(281331)
    z = torch.randn(count, 16, device=device)
    model, structural_optimizer = None, None
    if dense:
        indices = torch.arange(64, device=device)[None].expand(count, -1)
    elif method == "fixed_proposal":
        base = np.load(SOURCE / "moment_proposal.npz")["permutation"]
        indices = torch.tensor(base, device=device)[None].expand(count, -1)
    elif method == "random":
        indices = torch.stack([torch.randperm(64, device=device) for _ in range(count)])
    else:
        model = NFVAE(4, 64, 64 * 64).to(device)
        file = "untrained_prior.pt" if method == "untrained_vae_z" else "prior.pt"
        model.load_state_dict(torch.load(OUT / file, weights_only=True))
        model.requires_grad_(False).eval()
        with torch.no_grad():
            indices = permutation_matrix(model.decoder(z).reshape(-1, 64, 64)).argmax(
                -1
            )
        if method in ["vae_z", "untrained_vae_z"]:
            z.requires_grad_()
            structural_optimizer = torch.optim.Adam([z], lr=0.03)
    best = [p.detach().clone() for p in parameters]
    best_loss = torch.full((count,), torch.inf, device=device)
    best_indices, best_z = indices.clone(), z.detach().clone()
    best_step = torch.zeros(count, device=device, dtype=torch.long)
    history = []

    def predict(batch, values, indices, soft=None):
        w, b, a, c = values
        if dense:
            hidden = (batch @ w + b[:, None]).relu()
            return (hidden * a[:, None]).sum(-1) + c[:, None]
        grouped = (
            batch[:, indices].permute(1, 0, 2)
            if soft is None
            else torch.einsum("bi,msi->mbs", batch, soft)
        ).reshape(count, len(batch), 8, 8)
        hidden = (torch.einsum("mbgr,mkr->mbgk", grouped, w) + b[:, None, None]).relu()
        return (hidden * a[:, None, None]).sum((-1, -2)) / math.sqrt(32) + c[:, None]

    for step in range(steps + 1):
        if step % 128 == 0:
            with torch.no_grad():
                if model is not None and structural_optimizer is not None:
                    indices = permutation_matrix(
                        model.decoder(z).reshape(-1, 64, 64)
                    ).argmax(-1)
                error = torch.zeros(count, device=device)
                for ids in torch.arange(8192, 10240, device=device).split(256):
                    error += (
                        (predict(x[ids], parameters, indices) - labels[:, ids])
                        .square()
                        .sum(1)
                        / 2048
                        / scale
                    )
                better = error < best_loss
                best_loss[better] = error[better]
                for stored, current in zip(best, parameters):
                    stored[better] = current[better]
                best_indices[better] = indices[better]
                best_z[better] = z[better]
                best_step[better] = step
                history.append(
                    {
                        "step": step,
                        "query_nmse": error.cpu().tolist(),
                        "best_query_nmse": best_loss.cpu().tolist(),
                    }
                )
            if step % 1024 == 0:
                print(
                    method,
                    step,
                    float(best_loss.reshape(tasks, starts).min(1).values.mean()),
                    flush=True,
                )
        if step == steps:
            break
        ids = torch.randint(8192, (128,), device=device)
        soft = None
        structural_step = (
            structural_optimizer is not None and step >= steps // 2 and step % 16 == 0
        )
        if structural_step:
            soft = permutation_matrix(
                model.decoder(z).reshape(-1, 64, 64), straight_through=True
            )
            indices = soft.detach().argmax(-1)
        loss = (
            (predict(x[ids], parameters, indices, soft) - labels[:, ids])
            .square()
            .mean(1)
            / scale
        ).sum()
        if structural_step:
            loss = loss + 0.0001 * z.square().sum()
        optimizer.zero_grad(set_to_none=True)
        if structural_optimizer is not None:
            structural_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        old = [p.detach().clone() for p in parameters]
        optimizer.step()
        with torch.no_grad():
            for before, after in zip(old, parameters):
                shape = [count] + [1] * (after.ndim - 1)
                after.copy_(before + (after - before) * lr.reshape(shape))
        if structural_step:
            structural_optimizer.step()
    with torch.no_grad():
        test_error = torch.zeros(count, device=device)
        if len(x) > 10240:
            for ids in torch.arange(10240, len(x), device=device).split(256):
                test_error += (
                    (predict(x[ids], best, best_indices) - labels[:, ids])
                    .square()
                    .sum(1)
                    / (len(x) - 10240)
                    / scale
                )
    return {
        "query_nmse": best_loss.cpu().numpy(),
        "test_nmse": test_error.cpu().numpy(),
        "indices": best_indices.cpu().numpy(),
        "z": best_z.cpu().numpy(),
        "step": best_step.cpu().numpy(),
        **{f"parameter_{i}": p.detach().cpu().numpy() for i, p in enumerate(best)},
    }, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["calibrate", "evaluate"], required=True)
    args = parser.parse_args()
    snapshot = OUT / ("source_evaluation_" + args.stage + "_" + str(time.time_ns()))
    snapshot.mkdir()
    for file in ["scrambled_evaluate.py", "scrambled_transfer.py", "nf_prior.py"]:
        shutil.copy2(Path(__file__).parent / file, snapshot / file)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    if args.stage == "calibrate":
        old = json.loads((SOURCE / "controls_history.json").read_text())
        losses = np.array([r["all_nmse"] for r in old if r["method"] == "dense_adam"])
        scores = losses.min(0).reshape(4, 3, 4).min(-1).mean(0)
        adam_lr = [0.001, 0.003, 0.01][int(scores.argmin())]
        source = np.load(SOURCE / "source_data.npz")
        x, y = (
            torch.tensor(source["x"], device=device),
            torch.tensor(source["y"], device=device),
        )
        fit, history = numeric_fit(x, y, "dense_adam", [adam_lr], 6144, 8, device)
        np.savez_compressed(OUT / "dense_pretrained.npz", **fit)
        (OUT / "dense_pretrain_history.json").write_text(json.dumps(history))
        sgd_rates = [0.003, 0.01, 0.03, 0.1]
        fit, history = numeric_fit(
            x,
            y,
            "dense_sgd",
            [rate for rate in sgd_rates for _ in range(2)],
            6144,
            8,
            device,
        )
        scores = fit["query_nmse"].reshape(4, 4, 2).min(-1).mean(0)
        sgd_lr = sgd_rates[int(scores.argmin())]
        np.savez_compressed(OUT / "sgd_calibration.npz", **fit)
        (OUT / "sgd_calibration_history.json").write_text(json.dumps(history))
        (OUT / "calibration.json").write_text(
            json.dumps(
                {
                    "adam_lr": adam_lr,
                    "sgd_lr": sgd_lr,
                    "codebook_lr": 0.01,
                    "steps": 6144,
                    "restarts": 8,
                    "target_tasks": 8,
                    "success_nmse": 0.01,
                    "source_only": True,
                    "methods": [
                        "fixed_proposal",
                        "vae_sample",
                        "vae_z",
                        "untrained_vae_z",
                        "random",
                        "dense_adam",
                        "dense_sgd",
                        "dense_transfer",
                    ],
                },
                indent=2,
            )
        )
    else:
        spec = json.loads((OUT / "calibration.json").read_text())
        source_teacher = np.load(SOURCE / "hidden_teacher.npz")
        teacher_permutation = torch.tensor(source_teacher["permutation"], device=device)
        torch.manual_seed(281340)
        c = 0.75 + torch.rand(8, 4, 8, device=device)
        c *= torch.where(torch.rand_like(c) < 0.5, -1, 1)
        b = -(2 + torch.rand(8, 4, device=device))
        a = 0.7 + 0.6 * torch.rand(8, 4, device=device)
        x = torch.randn(8192 + 2048 + 8192, 64, device=device)
        y = teacher_labels(x, teacher_permutation, c, b, a)
        np.savez_compressed(
            OUT / "target_data.npz", x=x.cpu().numpy(), y=y.cpu().numpy()
        )
        np.savez_compressed(
            OUT / "target_teacher.npz",
            codebook=c.cpu().numpy(),
            bias=b.cpu().numpy(),
            readout=a.cpu().numpy(),
        )
        pretrained = np.load(OUT / "dense_pretrained.npz")
        selected = pretrained["query_nmse"].reshape(4, 8).argmin(1) + np.arange(4) * 8
        values = [
            torch.tensor(pretrained[f"parameter_{i}"][selected], device=device)
            for i in range(4)
        ]
        for method in spec["methods"]:
            if (OUT / (method + ".npz")).exists():
                continue
            rate = (
                spec["adam_lr"]
                if method in ["dense_adam", "dense_transfer"]
                else spec["sgd_lr"]
                if method == "dense_sgd"
                else spec["codebook_lr"]
            )
            fit, history = numeric_fit(
                x, y, method, [rate], spec["steps"], spec["restarts"], device, values
            )
            np.savez_compressed(OUT / (method + ".npz"), **fit)
            (OUT / (method + "_history.json")).write_text(json.dumps(history))
            print(
                "RESULT",
                method,
                "mean_test_nmse",
                float(fit["test_nmse"].mean()),
                "successes",
                int((fit["test_nmse"] < spec["success_nmse"]).sum()),
                flush=True,
            )
    (OUT / (args.stage + "_complete.json")).write_text(
        json.dumps({"stage": args.stage})
    )


if __name__ == "__main__":
    main()
