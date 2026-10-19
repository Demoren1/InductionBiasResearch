"""Separate learned Pattern prior from diagnostic column alignment."""

import json
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

from pattern.io import digest, save_json, save_torch
from pattern.masks import (
    align_masks,
    aligned_iou,
    analytical_mask,
    canonical_columns,
    exact_topk,
)
from pattern.models.task_model import Experiment

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
RUN = ROOT / "pattern/runs/nf_vae_v2"
OUT = HOME / "pattern"


def raw_iou(masks, reference):
    intersection = (masks * reference).sum((-2, -1))
    return intersection / (64 - intersection)


def describe(scores, reference):
    masks = exact_topk(scores)
    aligned = aligned_iou(masks, reference)
    raw = raw_iou(masks, reference)
    ordered = scores.flatten(1).sort(-1, descending=True).values
    return {
        "samples": len(scores),
        "mean_raw_iou": float(raw.mean()),
        "mean_aligned_iou": float(aligned.mean()),
        "raw_exact_fraction": float((raw == 1).float().mean()),
        "aligned_exact_fraction": float((aligned == 1).float().mean()),
        "unique_raw_masks": len(torch.unique(masks.flatten(1), dim=0)),
        "mean_top32_margin": float((ordered[:, 31] - ordered[:, 32]).mean()),
    }


def main():
    torch.set_num_threads(1)
    checkpoint_path = RUN / "checkpoints/best.pt"
    if (OUT / "metrics.json").exists():
        saved = json.loads((OUT / "metrics.json").read_text())
        if digest(checkpoint_path) != saved["checkpoint_sha256"]:
            raise ValueError("Pattern checkpoint changed")
        print("Reusing completed Pattern diagnostic", flush=True)
        return
    OUT.mkdir(exist_ok=False)
    checkpoint = torch.load(
        checkpoint_path, weights_only=True, map_location="cpu"
    )
    before = digest(checkpoint_path)
    config = checkpoint["config"]
    model = Experiment(config["tasks"], config["model"])
    model.load_state_dict(checkpoint["model_state"])
    decoder = model.decoder.eval().requires_grad_(False)
    reference = analytical_mask()
    generator = torch.Generator().manual_seed(17017)
    z = torch.randn(4096, config["model"]["latent_dim"], generator=generator)
    records = []
    with torch.no_grad():
        zero = decoder(torch.zeros_like(z[:1]))
        records.append({"kind": "trained_zero", **describe(zero, reference)})
        score_panels = {"trained_zero": zero[0], "reference": reference}
        for scale in (0.1, 1.0, 3.0, 10.0):
            scores = decoder(z * scale)
            records.append(
                {
                    "kind": "trained_prior",
                    "scale": scale,
                    "logit_std": float(scores.std(0).mean()),
                    **describe(scores, reference),
                }
            )
        bias = decoder.body[-1].bias.reshape(1, 11, 8)
        records.append({"kind": "last_bias_only", **describe(bias, reference)})
        score_panels["last_bias_only"] = bias[0]
    jacobian = torch.autograd.functional.jacobian(
        lambda value: decoder(value[None]).flatten(), z[0] * 0
    )
    for seed in (4113, 4114, 4115):
        torch.manual_seed(seed)
        untrained = Experiment(config["tasks"], config["model"]).decoder.eval()
        with torch.no_grad():
            scores = untrained(z[:1024])
        records.append(
            {
                "kind": "untrained_prior",
                "seed": seed,
                **describe(scores, reference),
            }
        )
        score_panels.setdefault(
            "untrained_zero", untrained(z[:1] * 0)[0].detach()
        )
    random_masks = exact_topk(torch.rand(4096, 11, 8, generator=generator))
    records.append(
        {"kind": "uniform_random_32", **describe(random_masks, reference)}
    )
    row_permuted = torch.stack(
        [
            reference[torch.randperm(11, generator=generator)]
            for _ in range(1024)
        ]
    )
    records.append(
        {
            "kind": "permuted_reference_rows",
            **describe(row_permuted, reference),
        }
    )
    column_permuted = torch.stack(
        [
            reference[:, torch.randperm(8, generator=generator)]
            for _ in range(1024)
        ]
    )
    records.append(
        {
            "kind": "permuted_reference_columns",
            **describe(column_permuted, reference),
        }
    )

    bank = Path(checkpoint["bank_reference"]["bank_path"])
    manifest_hash = digest(bank / "manifest.json")
    manifest = json.loads((bank / "manifest.json").read_text())
    sums = torch.zeros(2, 11, 8)
    counts = 0
    for task in tqdm(config["tasks"], desc="Pattern source-bank frequency"):
        for record in manifest["tasks"][task]:
            if record["split"] != "train":
                continue
            folder = bank / record["path"]
            mask = torch.load(
                folder / "imp_mask.pt", weights_only=True, map_location="cpu"
            )
            features = torch.load(
                folder / "nf_features.pt",
                weights_only=True,
                map_location="cpu",
            )
            order = canonical_columns(features["functional_map"])
            sums[0] += mask
            sums[1] += mask[:, order]
            counts += 1
    frequencies = sums / counts
    for name, values in zip(
        ("uncanonicalized_frequency", "canonical_frequency"), frequencies
    ):
        records.append(
            {
                "kind": name,
                "networks": counts,
                **describe(values[None], reference),
            }
        )
        score_panels[name] = values
    masks = exact_topk(torch.stack(list(score_panels.values())))
    aligned, orders = align_masks(masks, reference)
    save_torch(
        OUT / "panels.pt",
        {
            "names": list(score_panels),
            "scores": torch.stack(list(score_panels.values())),
            "masks": masks,
            "aligned": aligned,
            "orders": orders,
        },
    )
    probability = math.factorial(8) / math.comb(88, 32)
    save_json(
        OUT / "metrics.json",
        {
            "checkpoint_sha256": before,
            "bank_manifest_sha256": manifest_hash,
            "records": records,
            "jacobian_frobenius": float(jacobian.norm()),
            "uniform_random_exact_probability": probability,
            "uniform_random_reference_mean_iou": float(
                aligned_iou(random_masks).mean()
            ),
        },
    )
    assert digest(checkpoint_path) == before
    assert digest(bank / "manifest.json") == manifest_hash
    print(json.dumps(records, indent=2), flush=True)
    print(
        "Jacobian norm",
        float(jacobian.norm()),
        "chance",
        probability,
        flush=True,
    )


if __name__ == "__main__":
    main()
