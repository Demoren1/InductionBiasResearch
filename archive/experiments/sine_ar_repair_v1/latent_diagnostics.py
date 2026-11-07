"""Describe archived VAE checkpoints; never use these scores for selection."""

import json

import torch

from experiments.sine_ar_repair_v1.models import Prior, configurations
from experiments.sine_ar_repair_v1.run import (
    OUTPUT,
    PLAN,
    dump,
    training_data,
)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    rows = []
    for config in configurations():
        if "vae" not in config["family"]:
            continue
        masks, features, weights = training_data(config, "cpu")
        for step in PLAN["checkpoints"]:
            for index in range(3):
                model = Prior(config)
                model.load_state_dict(
                    torch.load(
                        OUTPUT / config["id"] / f"seed{index}_step{step}.pt",
                        weights_only=True,
                        map_location="cpu",
                    )
                )
                mu, logvar = model.net.encoder(features)
                logits = model.net.decode(mu).flatten(-2)
                selected = logits.argsort(
                    dim=-1, descending=True, stable=True
                )[..., :2]
                prediction = torch.zeros_like(logits).scatter_(-1, selected, 1)
                correct = (prediction == masks.flatten(-2)).all(-1).float()
                zero_mu, _ = model.net.encoder(torch.zeros_like(features))
                zero_logits = model.net.decode(zero_mu).flatten(-2)
                zero_selected = zero_logits.argsort(
                    dim=-1, descending=True, stable=True
                )[..., :2]
                zero_prediction = torch.zeros_like(zero_logits).scatter_(
                    -1, zero_selected, 1
                )
                zero_correct = (
                    (zero_prediction == masks.flatten(-2)).all(-1).float()
                )
                kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1)
                row = {
                    "model": config["id"],
                    "step": step,
                    "seed_index": index,
                }
                for name, weight in zip(["train", "validation"], weights):
                    row[f"{name}_exact_reconstruction"] = float(
                        (correct * weight).sum()
                    )
                    row[f"{name}_zero_feature_reconstruction"] = float(
                        (zero_correct * weight).sum()
                    )
                    row[f"{name}_kl"] = float((kl * weight).sum())
                    row[f"{name}_posterior_variance"] = float(
                        (logvar.exp().mean(-1) * weight).sum()
                    )
                    mean = (mu * weight[..., None]).sum((0, 1))
                    variance = ((mu - mean).square() * weight[..., None]).sum(
                        (0, 1)
                    )
                    row[f"{name}_active_units"] = int((variance > 0.01).sum())
                rows.append(row)
        print(json.dumps({"diagnosed": config["id"]}), flush=True)
    dump(OUTPUT / "latent_diagnostics.json", rows)


if __name__ == "__main__":
    main()
