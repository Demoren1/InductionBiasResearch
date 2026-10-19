"""Paired AGOP ablations on saved sparse teachers; never rebuild a bank."""

import argparse
import copy
import json
from pathlib import Path

import experiment as architecture
import torch
from torch import nn
from tqdm.auto import tqdm

from multimask.common import digest, load_bank, save_json, save_torch
from pattern.masks import aligned_iou, canonical_columns, exact_topk
from pattern.models.nf import NFLayer
from pattern.models.task_model import Experiment
from pattern.train import task_losses, validation_metrics

HOME = Path(__file__).resolve().parent
OUT = HOME / "agop"
ROOT = HOME.parents[1]
REPRESENTATIONS = ("functional", "agop", "diagonal", "zero")
BRANCHES = ("branch_agop", "branch_functional", "branch_zero")
PATTERN_BANK = ROOT / "data/pattern/banks/imp32_fullsupport_1000_20261009"


class DualEncoder(nn.Module):
    """Fuse invariant NF weight/gradient features with a full matrix MLP."""

    def __init__(self, rows, options):
        super().__init__()
        self.rows = rows
        self.edge_count = rows * 8 * 2
        width, channels = options["encoder_width"], options["nf_channels"]
        side_dim = max(rows * 8, rows * (rows + 1) // 2)
        self.nf = nn.Sequential(
            NFLayer(2, channels, rows),
            nn.GELU(),
            NFLayer(channels, channels, rows),
            nn.GELU(),
        )
        self.weight_projection = nn.Sequential(
            nn.Linear(rows * channels, width), nn.GELU()
        )
        self.matrix_projection = nn.Sequential(
            nn.Linear(side_dim, width), nn.GELU()
        )
        self.posterior = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 2 * options["latent_dim"]),
        )

    def forward(self, x):
        edges = x[:, : self.edge_count].reshape(-1, self.rows, 8, 2)
        nf = self.weight_projection(self.nf(edges).mean(2).flatten(1))
        matrix = self.matrix_projection(x[:, self.edge_count :])
        mu, logvar = self.posterior(torch.cat((nf, matrix), -1)).chunk(2, -1)
        return mu, logvar.clamp(-12, 8)


class BranchNetwork(nn.Module):
    """Task encoders and the unchanged shared Multimask decoder."""

    def __init__(self, kind, config):
        super().__init__()
        self.kind, self.task_ids = kind, architecture.tasks(config)
        options = config["model"]
        self.z_dim = options["latent_dim"]
        self.encoders = nn.ModuleList(
            DualEncoder(68, options) for _ in self.task_ids
        )
        width = options["decoder_width"]
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 68 * 8),
            nn.Unflatten(-1, (68, 8)),
        )

    def decode(self, z, task=0, family="A"):
        return self.decoder(z)

    def forward(self, task, x, sample=False):
        mu, logvar = self.encoders[task](x)
        z = mu + torch.randn_like(mu) * (logvar * 0.5).exp() if sample else mu
        return self.decode(z), mu, logvar


def read(path):
    return json.loads(path.read_text())


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


@torch.no_grad()
def channels(state, x):
    """Logit AGOP and its projection onto incoming weight directions."""
    weight, bias, readout = (state[key] for key in ("w", "b", "a"))
    gates = (x @ weight + bias[:, None] > 0).float()
    derivatives = gates * readout[:, None]
    jacobian = derivatives @ weight.transpose(1, 2)
    agop = jacobian.transpose(1, 2) @ jacobian / len(x)
    directions = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
    projected = agop @ directions
    diagonal = weight.square() * readout[:, None].square()
    diagonal = diagonal * gates.mean(1)[:, None]
    functional = weight.abs() * readout[:, None].abs() * gates.mean(1)[:, None]
    return {
        "agop": projected,
        "diagonal": diagonal,
        "functional": functional,
        "zero": torch.zeros_like(weight),
    }, agop


def check_derivative(state, x, agop):
    sample = x[:7].clone().requires_grad_()
    w, b, a = (state[k][:1] for k in ("w", "b", "a"))
    prediction = ((sample @ w[0] + b).relu() * a).sum(-1)
    gradient = torch.autograd.grad(prediction.sum(), sample)[0]
    full = (x @ w + b[:, None] > 0).float() * a[:, None]
    analytical = full @ w.transpose(1, 2)
    torch.testing.assert_close(gradient, analytical[0, :7])
    torch.testing.assert_close(
        agop[0], analytical[0].T @ analytical[0] / len(x)
    )
    eigenvalues = torch.linalg.eigvalsh(agop.double())
    if eigenvalues.min() < -1e-6 * agop.norm().clamp_min(1):
        raise ValueError("AGOP is not positive semidefinite")
    return float((gradient - analytical[0, :7]).abs().max())


def replace(card, representation):
    result = dict(card)
    if representation.startswith("branch_"):
        rows = card["features"].shape[1]
        side_dim = max(rows * 8, rows * (rows + 1) // 2)
        if representation == "branch_agop":
            indices = torch.triu_indices(rows, rows)
            side = card["full_agop"][:, indices[0], indices[1]]
        elif representation == "branch_functional":
            side = card["functional"].flatten(1)
        else:
            side = torch.zeros(len(card["features"]), side_dim)
        side = side / side.abs().amax(1, keepdim=True).clamp_min(1e-8)
        side = nn.functional.pad(side, (0, side_dim - side.shape[1]))
        result["features"] = torch.cat(
            (card["features"][..., :2].flatten(1), side), -1
        )
        return result
    result["features"] = card["features"].clone()
    values = card[representation]
    result["features"][..., 2] = values / values.abs().amax(
        (1, 2), keepdim=True
    ).clamp_min(1e-8)
    return result


def prepare_multimask():
    cache = OUT / "multimask_cards.pt"
    if cache.exists():
        return load(cache)
    config = read(architecture.CONFIG)
    _, manifest, cards, data = load_bank(architecture.BANK, config)
    audit, full = [], {}
    for task, card in tqdm(cards.items(), desc="Multimask AGOP"):
        state = {k: card[k] for k in ("w", "b", "a", "c")}
        state["w"] = card["raw_channels"][..., 0]
        alternatives, agop = channels(state, data["probe"])
        error = check_derivative(state, data["probe"], agop)
        difference = (
            alternatives["functional"] - card["raw_channels"][..., 2]
        ).abs()
        alternatives["functional"] = card["raw_channels"][..., 2]
        for name in REPRESENTATIONS:
            card[name] = alternatives[name].gather(
                2, card["column_orders"][:, None].expand_as(alternatives[name])
            )
        full[task] = agop
        audit.append(
            {
                "task": task,
                "jacobian_error": error,
                "functional_recompute_max_error": float(difference.max()),
                "functional_recompute_changed_fraction": float(
                    (difference > 1e-5).float().mean()
                ),
            }
        )
    save_torch(OUT / "full_agop.pt", full)
    save_json(
        OUT / "protocol.json",
        {
            "bank": str(architecture.BANK),
            "bank_sha256": digest(architecture.BANK / "manifest.json"),
            "bank_hashes": manifest["hashes"],
            "agop_output": "scalar pre-sigmoid logit, not loss/probability",
            "observations": "same unlabeled probe as functional channel",
            "projection": "A @ column_L2_normalized_W",
            "diagonal": "per-neuron E[(d(a_j ReLU_j)/dx_i)^2]",
            "ordering": "unchanged functional-centroid canonical targets",
            "normalization": "per-teacher/channel max absolute, floor 1e-8",
            "jacobian_checks": audit,
        },
    )
    save_torch(cache, (config, cards, data))
    return config, cards, data


def multimask(
    stage, representations, seeds, device, optimize=False, z_method=None
):
    config, cards, data = prepare_multimask()
    full = load(OUT / "full_agop.pt")
    for task in cards:
        cards[task]["full_agop"] = full[task]
    for representation in representations:
        # Functional models are exact controls already trained on this bank.
        architecture.HOME = (
            HOME if representation == "functional" else OUT / representation
        )
        architecture.HOME.joinpath("runs").mkdir(parents=True, exist_ok=True)
        modified = {
            task: replace(card, representation) for task, card in cards.items()
        }
        settings = copy.deepcopy(config)
        if stage == "transfer" and z_method is not None:
            settings["evaluation"]["z_method"] = z_method
        if representation != "functional":
            settings["input_representation"] = representation
        for seed in seeds:
            print(f"{stage}: {representation}/{seed}", flush=True)
            if stage == "train":
                if representation != "functional":
                    architecture.train_one(
                        "original",
                        seed,
                        settings,
                        modified,
                        device,
                        100,
                        reuse_baseline=False,
                        model_factory=BranchNetwork
                        if representation in BRANCHES
                        else architecture.Network,
                    )
            elif stage == "transfer":
                if representation == "functional" and seed == 4213:
                    continue  # Full cached Gumbel evaluation: 13 October.
                architecture.transfer_one(
                    "original",
                    seed,
                    settings,
                    modified,
                    data,
                    device,
                    optimize=optimize or seed == 4213,
                    model_factory=BranchNetwork
                    if representation in BRANCHES
                    else architecture.Network,
                )


def prepare_pattern():
    cache = OUT / "pattern_cards.pt"
    if cache.exists():
        existing = load(cache)
        if "full_agop" in next(iter(existing[1].values())):
            return existing
    config = read(ROOT / "pattern/configs/tuned.json")
    manifest = read(PATTERN_BANK / "manifest.json")
    x = load(PATTERN_BANK / "probe.pt")["x"]
    cards, maximum_error = {}, 0.0
    for task in tqdm(config["tasks"], desc="Pattern saved teachers"):
        rows = []
        for record in manifest["tasks"][task]:
            directory = PATTERN_BANK / record["path"]
            raw = load(directory / "nf_features.pt")
            saved = load(directory / "network_state.pt")["state_dict"]
            weight = raw["weights"]
            saved["w"] = weight
            state = {k: v[None] for k, v in saved.items()}
            alternatives, agop = channels(state, x)
            if not rows:
                maximum_error = max(
                    maximum_error, check_derivative(state, x, agop)
                )
            alternatives["functional"] = raw["functional_map"][None]
            order = canonical_columns(raw["functional_map"])
            features = torch.stack(
                [
                    raw[k]
                    for k in ("weights", "loss_gradient", "functional_map")
                ],
                -1,
            )[:, order]
            features /= features.abs().amax((0, 1)).clamp_min(1e-8)
            mask = load(directory / "imp_mask.pt").float()[:, order]
            rows.append(
                {
                    "features": features,
                    "masks": mask,
                    "full_agop": agop[0],
                    "split": torch.tensor(
                        {"train": 0, "validation": 1, "test": 2}[
                            record["split"]
                        ]
                    ),
                    **{
                        name: value[0, :, order]
                        for name, value in alternatives.items()
                    },
                }
            )
        cards[task] = {
            k: torch.stack([row[k] for row in rows]) for k in rows[0]
        }
    save_json(
        OUT / "pattern_protocol.json",
        {
            "bank": str(PATTERN_BANK),
            "bank_sha256": digest(PATTERN_BANK / "manifest.json"),
            "jacobian_max_error": maximum_error,
            "config": config,
        },
    )
    save_torch(cache, (config, cards))
    return config, cards


@torch.no_grad()
def pattern_metrics(model, cards, role, options, device):
    samples = {
        task: tuple(
            card[k][card["split"] == role].to(device)
            for k in ("features", "masks")
        )
        for task, card in cards.items()
    }
    values = validation_metrics(
        model, samples, options["beta"], options["hard_loss_weight"]
    )
    return {
        ("iou" if key == "aligned_iou" else key): sum(
            row[key] for row in values.values()
        )
        / len(values)
        for key in next(iter(values.values()))
    }


def pattern(representations, seeds, device):
    config, cards = prepare_pattern()
    options = config["training"]
    for representation in representations:
        modified = {t: replace(c, representation) for t, c in cards.items()}
        for seed in seeds:
            destination = OUT / "pattern" / f"{representation}_{seed}"
            if (destination / "COMPLETE.json").exists():
                continue
            destination.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(seed)
            model = Experiment(config["tasks"], config["model"]).to(device)
            if representation in BRANCHES:
                model.encoders = nn.ModuleDict(
                    {
                        task: DualEncoder(11, config["model"])
                        for task in config["tasks"]
                    }
                ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=options["lr"],
                weight_decay=options["weight_decay"],
            )
            training = {
                t: tuple(
                    c[k][c["split"] == 0].to(device)
                    for k in ("features", "masks")
                )
                for t, c in modified.items()
            }
            generator = torch.Generator().manual_seed(seed + 17)
            history, best, best_epoch = [], (-1, -float("inf")), 0
            for epoch in tqdm(
                range(1, 6), desc=f"Pattern {representation}/{seed}"
            ):
                model.train()
                orders = {
                    t: torch.randperm(len(x), generator=generator)
                    for t, (x, _) in training.items()
                }
                for start in range(
                    0,
                    len(next(iter(training.values()))[0]),
                    options["batch_size"],
                ):
                    samples = {
                        t: tuple(
                            v[orders[t][start : start + options["batch_size"]]]
                            for v in vs
                        )
                        for t, vs in training.items()
                    }
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = task_losses(
                        model,
                        samples,
                        options["beta"] * epoch / 30,
                        options["hard_loss_weight"],
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), 1, error_if_nonfinite=True
                    )
                    optimizer.step()
                model.eval()
                row = {"epoch": epoch}
                for role, split in ((0, "train"), (1, "validation")):
                    row.update(
                        {
                            f"{split}_{k}": v
                            for k, v in pattern_metrics(
                                model, modified, role, options, device
                            ).items()
                        }
                    )
                history.append(row)
                rank = row["validation_iou"], -row["validation_loss"]
                if rank > best:
                    best, best_epoch = rank, epoch
                    save_torch(
                        destination / "best.pt",
                        {
                            "model": {
                                k: v.detach().cpu()
                                for k, v in model.state_dict().items()
                            },
                            "epoch": epoch,
                        },
                    )
            model.load_state_dict(load(destination / "best.pt")["model"])
            model.eval().requires_grad_(False)
            with torch.no_grad():
                z = torch.randn(
                    4096, 16, generator=torch.Generator().manual_seed(17017)
                )
                scores = model.decoder(z.to(device)).cpu()
                iou = aligned_iou(exact_topk(scores))
                zero = model.decoder(torch.zeros(1, 16, device=device)).cpu()
            save_torch(
                destination / "prior.pt", {"scores": scores[:32], "zero": zero}
            )
            save_json(destination / "history.json", history)
            save_json(
                destination / "diagnostics.json",
                {
                    "best_epoch": best_epoch,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "bank_test": pattern_metrics(
                        model, modified, 2, options, device
                    ),
                    "prior_iou": float(iou.mean()),
                    "prior_exact": float((iou == 1).float().mean()),
                    "zero_iou": float(aligned_iou(exact_topk(zero))[0]),
                },
            )
            save_json(destination / "COMPLETE.json", {"seed": seed})
            print(f"DONE Pattern {representation}/{seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "train", "transfer", "pattern")
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS + BRANCHES,
        default=list(BRANCHES),
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--z-method", choices=("gumbel", "evolution"))
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run full z optimization for every selected seed",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    if args.stage == "pattern":
        pattern(
            args.representations, args.seeds or [4113, 4114, 4115], args.device
        )
    elif args.stage == "prepare":
        prepare_multimask()
    else:
        multimask(
            args.stage,
            args.representations,
            args.seeds or [4213, 4214, 4215],
            args.device,
            args.optimize,
            args.z_method,
        )


if __name__ == "__main__":
    main()
