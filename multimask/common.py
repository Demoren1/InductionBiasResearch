"""Shared data, batched MLPs and artifacts for four-reference tasks."""

import csv
import json
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from pattern.bank.imp import QueryCheckpoint, l2_penalty, mean_bce
from pattern.io import digest, save_json, save_torch
from pattern.models.nf import NFLayer
from pattern.reporting import progress, pyplot, save_figure

ROWS, HIDDEN, K = 68, 8, 32
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def tasks(config, split="train", family=None):
    return [
        f"{group}_{pattern}"
        for group in ([family] if family else "AB")
        for pattern in config[f"{split}_patterns"]
    ]


def load_config(path, args=None):
    config = json.loads(Path(path).read_text())
    for section, key, flag in (
        ("bank", "maps_per_task", "maps_per_task"),
        ("bank", "steps", "bank_steps"),
        ("bank", "batch_size", "bank_batch"),
        ("training", "epochs", "epochs"),
        ("training", "batch_size", "batch_size"),
        ("training", "lr", "lr"),
        ("training", "beta", "beta"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            config[section][key] = value
    for split in ("train", "test"):
        patterns = config[f"{split}_patterns"]
        if (
            not patterns
            or len(set(patterns)) != len(patterns)
            or any(len(p) != 4 or set(p) - set("01") for p in patterns)
        ):
            raise ValueError("patterns must be unique four-bit strings")
    if set(config["train_patterns"]) & set(config["test_patterns"]):
        raise ValueError("task splits overlap")
    for section, keys in (
        ("data", config["data"]),
        ("bank", ("maps_per_task", "steps", "batch_size", "select_every")),
        ("model", config["model"]),
        ("training", ("epochs", "batch_size")),
    ):
        if any(config[section][key] < 1 for key in keys):
            raise ValueError(f"{section} sizes must be positive")
    bank, train = config["bank"], config["training"]
    if (
        bank["maps_per_task"] < 4
        or not 0 < bank["prune_fraction"] < 1
        or bank["lr"] <= 0
        or bank["l2"] < 0
    ):
        raise ValueError("invalid bank protocol")
    if (
        train["lr"] <= 0
        or train["weight_decay"] < 0
        or train["beta"] < 0
        or train["warmup"] < 0
        or train["patience"] < 0
        or not 0 <= train["hard_weight"] <= 1
    ):
        raise ValueError("invalid training protocol")
    return config


def make_data(config):
    generator = torch.Generator().manual_seed(config["seed"])
    bits = (
        torch.randint(
            2, (sum(config["data"].values()), 2, 32), generator=generator
        ).float()
        * 2
        - 1
    )
    x = torch.cat((bits, bits[:, :, [0, 16]]), -1).flatten(1)
    return dict(zip(config["data"], x.split(list(config["data"].values()))))


def target(x, task):
    """Exact finite Bayes logit, evaluated only on the permitted task split."""
    family, pattern = task.split("_")
    offset = 34 * (family == "B")
    signs = x.new_tensor([2 * int(bit) - 1 for bit in pattern])
    gates = F.relu(
        (
            x[..., offset : offset + 32].reshape(*x.shape[:-1], 8, 4) * signs
        ).sum(-1)
        - 3
    )
    return (2 * (gates.sum(-1) - 0.5)).sigmoid()


def references(task):
    offset = 34 * (task[0] == "B")
    result = torch.zeros(4, ROWS, HIDDEN)
    for variant in range(4):
        rows = torch.arange(32)
        if variant & 1:
            rows[0] = 32
        if variant & 2:
            rows[16] = 33
        result[variant, rows + offset, torch.arange(32) // 4] = 1
    return result


def topk(scores, straight_through=False):
    flat = scores.flatten(-2)
    indices = (
        flat.detach()
        .cpu()
        .argsort(dim=-1, descending=True, stable=True)[..., :K]
        .to(scores.device)
    )
    hard = torch.zeros_like(flat).scatter(-1, indices, 1).reshape_as(scores)
    if straight_through:
        soft = scores.sigmoid()
        return hard + (soft - soft.detach())
    return hard


def initial(seeds, device):
    states = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        states.append(
            {
                "w": torch.randn(ROWS, HIDDEN, generator=generator) * 0.1,
                "b": torch.zeros(HIDDEN),
                "a": torch.randn(HIDDEN, generator=generator) * 0.1,
                "c": torch.zeros(()),
            }
        )
    return {
        key: torch.stack([state[key] for state in states]).to(device)
        for key in states[0]
    }


def logits(state, masks, x):
    hidden = F.relu(torch.matmul(x, state["w"] * masks) + state["b"][:, None])
    return (hidden * state["a"][:, None]).sum(-1) + state["c"][:, None]


def bce(state, masks, x, y):
    prediction = logits(state, masks, x)
    return mean_bce(prediction, y.expand_as(prediction))


def fit(
    original,
    masks,
    support,
    query,
    steps,
    lr,
    l2=0.0,
    select_every=25,
    enabled=False,
    log_every=25,
):
    """Independent Adam fits; query chooses each network's own checkpoint."""
    state = {
        key: value.detach().clone().requires_grad_()
        for key, value in original.items()
    }
    optimizer = torch.optim.Adam(state.values(), lr=lr)
    checkpoint = QueryCheckpoint(state, (len(masks),))
    history = []
    loop = progress(
        range(1, steps + 1),
        enabled,
        desc=f"Fresh MLPs ({len(masks)})",
        unit="step",
        leave=False,
    )
    for step in loop:
        loss = bce(state, masks, *support) + l2_penalty(state, masks, l2)
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
        with torch.no_grad():
            state["w"].mul_(masks)
            if step % select_every == 0 or step == steps:
                score = bce(state, masks, *query)
                if not torch.isfinite(score).all():
                    raise RuntimeError("nonfinite child BCE")
                checkpoint.update(state, score, step)
            if step == 1 or step % log_every == 0 or step == steps:
                values = (
                    torch.stack(
                        (
                            bce(state, masks, *support),
                            bce(state, masks, *query),
                        )
                    )
                    .mean(1)
                    .cpu()
                    .tolist()
                )
                history.append(
                    dict(step=step, support_bce=values[0], query_bce=values[1])
                )
                loop.set_postfix(bce=f"{values[1]:.5f}")
    return checkpoint.state, checkpoint.steps.cpu(), history


class Model(nn.Module):
    def __init__(self, task_ids, config):
        super().__init__()
        channels = config["nf_channels"]
        width = config["encoder_width"]
        z = config["latent_dim"]
        self.encoders = nn.ModuleDict(
            {
                task: nn.Sequential(
                    NFLayer(3, channels, ROWS),
                    nn.GELU(),
                    NFLayer(channels, channels, ROWS),
                    nn.GELU(),
                )
                for task in task_ids
            }
        )
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(ROWS * channels, width),
                    nn.GELU(),
                    nn.Linear(width, 2 * z),
                )
                for task in task_ids
            }
        )
        width = config["decoder_width"]
        self.decoder = nn.Sequential(
            nn.Linear(z, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, ROWS * HIDDEN),
            nn.Unflatten(-1, (ROWS, HIDDEN)),
        )

    def forward(self, task, x, sample=False):
        mu, logvar = self.heads[task](
            self.encoders[task](x).mean(2).flatten(1)
        ).chunk(2, -1)
        logvar = logvar.clamp(-12, 8)
        z = mu + torch.randn_like(mu) * (logvar * 0.5).exp() if sample else mu
        scores = self.decoder(z)
        return scores, mu, logvar


def vae_loss(output, masks, beta, hard_weight):
    scores, mu, logvar = output
    reconstruction = (1 - hard_weight) * F.binary_cross_entropy_with_logits(
        scores, masks, pos_weight=scores.new_tensor((ROWS * HIDDEN - K) / K)
    )
    reconstruction += hard_weight * F.mse_loss(topk(scores, True), masks)
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    return reconstruction + beta * kl, kl


def matches(masks, task):
    """Nearest of four references, modulo hidden-column permutations only."""
    masks = masks.detach().float().cpu()
    refs = references(task)
    overlaps = torch.einsum("brh,vrj->bvhj", masks, refs).numpy()
    values = []
    variants = []
    orders = []
    for candidates in overlaps:
        best = (-1, None, None)
        for variant, score in enumerate(candidates):
            source, destination = linear_sum_assignment(-score)
            intersection = float(score[source, destination].sum())
            if intersection > best[0]:
                order = torch.empty(HIDDEN, dtype=torch.long)
                order[torch.as_tensor(destination)] = torch.as_tensor(source)
                best = (intersection, variant, order)
        values.append(best[0] / (2 * K - best[0]))
        variants.append(best[1])
        orders.append(best[2])
    return torch.tensor(values), torch.tensor(variants), torch.stack(orders)


def save_maps(destination, name, scores, task, selected=0):
    scores = scores.detach().cpu()
    masks = topk(scores)
    iou, variants, orders = matches(masks, task)
    aligned = masks.gather(-1, orders[:, None, :].expand_as(masks))
    probabilities = scores.sigmoid()
    soft = probabilities.gather(
        -1, orders[:, None, :].expand_as(probabilities)
    )
    payload = dict(
        logits=scores,
        canonical=masks,
        probabilities=probabilities,
        aligned=aligned,
        probabilities_aligned=soft,
        orders=orders,
        references=references(task),
        iou=iou,
        variants=variants,
        selected_index=selected,
        alignment_scope=(
            "hidden columns for visualization only; "
            "raw masks used for child BCE"
        ),
    )
    save_torch(Path(destination) / "generated_masks" / f"{name}.pt", payload)
    panels = {
        "Nearest reference": references(task)[variants[selected]],
        "Generated mask": aligned[selected],
        "Decoder sigmoid": soft[selected],
        "Mask frequency": aligned.mean(0),
        "Mean sigmoid": soft.mean(0),
    }
    fig, axes = pyplot().subplots(
        1, len(panels), figsize=(15, 8), constrained_layout=True
    )
    for ax, (label, values) in zip(axes, panels.items()):
        image = ax.imshow(
            values.numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto"
        )
        ax.set_title(label)
        ax.set_xlabel("Hidden neuron")
        ax.set_yticks(range(0, ROWS, 4))
        ax.set_ylabel("Input coordinate")
    fig.colorbar(image, ax=list(axes), label="Value [0,1]", shrink=0.7)
    path = Path(destination) / "heatmaps" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, path)
    save_torch(path.with_name(name + "_values.pt"), panels)
    return payload


def save_history(destination, rows, x="epoch"):
    destination = Path(destination)
    save_json(destination / "history.json", rows)
    if not rows:
        return
    keys = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
            if isinstance(row[key], (int, float, str))
        )
    )
    with (destination / "losses.csv").open("w") as handle:
        writer = csv.DictWriter(handle, keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    groups = [
        ("Loss / BCE", [key for key in keys if "loss" in key or "bce" in key]),
        (
            "KL divergence",
            [key for key in keys if key.endswith("_kl") and "beta" not in key],
        ),
        ("Weighted KL", [key for key in keys if "beta_kl" in key]),
        (
            "Nearest-reference IoU",
            [key for key in keys if key.endswith("iou")],
        ),
    ]
    groups = [group for group in groups if group[1]]
    fig, axes = pyplot().subplots(
        len(groups),
        1,
        figsize=(7, 3 * len(groups)),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (label, fields) in zip(axes[:, 0], groups):
        for key in fields:
            ax.plot(
                [row[x] for row in rows],
                [row.get(key, float("nan")) for row in rows],
                label=key,
            )
        ax.set_ylabel(label)
        ax.set_xlabel(x)
        ax.grid(alpha=0.2)
        ax.legend()
    save_figure(fig, destination / "losses")


def load_bank(bank, config=None):
    bank = Path(bank).resolve()
    manifest = json.loads((bank / "manifest.json").read_text())
    if (
        manifest.get("schema") != "multimask.v1"
        or manifest["status"] != "complete"
    ):
        raise ValueError("bank is incomplete or incompatible")
    if config and any(
        manifest[key] != config[key]
        for key in ("seed", "data", "bank", "train_patterns")
    ):
        raise ValueError(
            "bank protocol differs from config; use another BANK path"
        )
    for name, sha in manifest["hashes"].items():
        if digest(bank / name) != sha:
            raise ValueError(f"bank artifact changed: {name}")
    cards = {
        task: torch.load(bank / f"{task}.pt", weights_only=True)
        for task in manifest["tasks"]
    }
    data = torch.load(bank / "data.pt", weights_only=True)
    return bank, manifest, cards, data
