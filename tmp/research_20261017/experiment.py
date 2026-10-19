"""Source-only architecture experiments; preserve production code and banks."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from multimask.common import (
    HIDDEN,
    ROWS,
    Model,
    digest,
    load_bank,
    matches,
    save_json,
    save_torch,
    tasks,
    topk,
    vae_loss,
)
from multimask.evaluate import fresh, search
from pattern.models.nf import NFLayer

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(__file__).resolve().parent
CONFIG = ROOT / "multimask/config.json"
BANK = ROOT / "data/multimask/banks/imp32_128_mlp_v3"
KINDS = ("original", "reverse", "shared", "cvae", "cvae_elbo")


class Network(nn.Module):
    """Swap parameter sharing; the condition contains only family identity."""

    def __init__(self, kind, config):
        super().__init__()
        self.kind = kind
        self.task_ids = tasks(config)
        options = config["model"]
        self.z_dim = options["latent_dim"]
        self.conditioned = kind.startswith("cvae")
        if kind == "original":
            self.original = Model(self.task_ids, options)
            return
        channels = options["nf_channels"]
        width = options["encoder_width"]
        extra = 2 * self.conditioned
        self.encoder = nn.Sequential(
            NFLayer(3, channels, ROWS),
            nn.GELU(),
            NFLayer(channels, channels, ROWS),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(ROWS * channels + extra, width),
            nn.GELU(),
            nn.Linear(width, 2 * self.z_dim),
        )
        width = options["decoder_width"]
        self.decoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.z_dim + extra, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, ROWS * HIDDEN),
                nn.Unflatten(-1, (ROWS, HIDDEN)),
            )
            for _ in range(len(self.task_ids) if kind == "reverse" else 1)
        )

    @staticmethod
    def condition(x, family):
        code = x.new_tensor([family == "A", family == "B"])
        return torch.cat((x, code.expand(len(x), 2)), -1)

    def decode(self, z, task=0, family="A"):
        if self.kind == "original":
            return self.original.decoder(z)
        if self.conditioned:
            z = self.condition(z, family)
        return self.decoders[task if self.kind == "reverse" else 0](z)

    def forward(self, task, x, sample=False):
        if self.kind == "original":
            return self.original(self.task_ids[task], x, sample)
        family = self.task_ids[task][0]
        encoded = self.encoder(x).mean(2).flatten(1)
        if self.conditioned:
            encoded = self.condition(encoded, family)
        mu, logvar = self.head(encoded).chunk(2, -1)
        logvar = logvar.clamp(-12, 8)
        z = mu + torch.randn_like(mu) * (logvar * 0.5).exp() if sample else mu
        return self.decode(z, task, family), mu, logvar


class BoundDecoder(nn.Module):
    """A frozen source-selected branch or fixed, known family condition."""

    def __init__(self, model, family, head=0):
        super().__init__()
        self.model, self.family, self.head = model, family, head

    def forward(self, z):
        return self.model.decode(z, self.head, self.family)


def prepare(device):
    config = json.loads(CONFIG.read_text())
    _, _, cards, data = load_bank(BANK, config)
    return config, cards, data


def objective(model, output, masks, config, beta=None):
    if model.kind != "cvae_elbo":
        return vae_loss(
            output,
            masks,
            config["training"]["beta"] if beta is None else beta,
            config["training"]["hard_weight"],
        )
    scores, mu, logvar = output
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    reconstruction = nn.functional.binary_cross_entropy_with_logits(
        scores, masks
    )
    # Negative Bernoulli ELBO per edge; beta is irrelevant in this control.
    return reconstruction + kl / (ROWS * HIDDEN), kl


@torch.no_grad()
def metrics(model, cards, config, role, device):
    rows = []
    for index, task in enumerate(tasks(config)):
        card = cards[task]
        chosen = card["split"] == role
        output = model(index, card["features"][chosen].to(device))
        loss, kl = objective(
            model, output, card["masks"][chosen].to(device), config
        )
        rows.append(
            {
                "task": task,
                "loss": float(loss),
                "kl": float(kl),
                "iou": float(matches(topk(output[0]), task)[0].mean()),
                "mu_variance": float(output[1].var(0, unbiased=False).mean()),
            }
        )
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
        if key != "task"
    }, rows


@torch.no_grad()
def prior_metrics(model, config, device):
    z = torch.randn(
        512, model.z_dim, generator=torch.Generator().manual_seed(71)
    )
    z = z.to(device)
    rows = []
    heads = range(len(tasks(config))) if model.kind == "reverse" else (0,)
    for head in heads:
        families = (
            (tasks(config)[head][0],) if model.kind == "reverse" else "AB"
        )
        for family in families:
            scores = model.decode(z, head, family)
            masks = topk(scores)
            iou, variants, _ = matches(masks, f"{family}_0000")
            zero = topk(model.decode(torch.zeros_like(z[:1]), head, family))
            counts = torch.bincount(variants[iou == 1], minlength=4)
            rows.append(
                {
                    "family": family,
                    "head": head,
                    "mean_iou": float(iou.mean()),
                    "exact_fraction": float((iou == 1).float().mean()),
                    "covered_variants": int((counts > 0).sum()),
                    "variant_counts": counts.tolist(),
                    "zero_iou": float(matches(zero, f"{family}_0000")[0][0]),
                    "mask_changes_from_zero": float(
                        (masks != zero).flatten(1).any(1).float().mean()
                    ),
                    "logit_std": float(scores.std(0).mean()),
                }
            )
    return rows


def train_one(
    kind,
    seed,
    config,
    cards,
    device,
    epochs,
    suffix="",
    reuse_baseline=True,
    model_factory=Network,
):
    destination = HOME / "runs" / f"{kind}_{seed}{suffix}"
    if (destination / "COMPLETE.json").exists():
        previous = json.loads((destination / "protocol.json").read_text())
        if previous["config"] != config or previous["epochs_limit"] != epochs:
            raise ValueError(
                "Completed run has different settings; use a new suffix"
            )
        print(f"Already complete: {destination.name}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = model_factory(kind, config).to(device)
    reference = {
        "kind": kind,
        "seed": seed,
        "config": config,
        "bank_sha256": digest(BANK / "manifest.json"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "epochs_limit": epochs,
        "objective": (
            "negative Bernoulli ELBO per edge, KL coefficient 1/544"
            if kind == "cvae_elbo"
            else "original weighted BCE/hard-MSE/beta-KL surrogate"
        ),
    }
    save_json(destination / "protocol.json", reference)
    # Reuse the existing matched seed, not a freshly observed held-out run.
    if kind == "original" and seed == 4213 and reuse_baseline:
        old = ROOT / "multimask/runs/nf_vae_v3"
        checkpoint = torch.load(
            old / "checkpoints/best.pt", map_location="cpu", weights_only=True
        )
        if checkpoint["bank"]["sha256"] != reference["bank_sha256"]:
            raise ValueError("Cached baseline belongs to another bank")
        model.original.load_state_dict(checkpoint["model"])
        history = json.loads((old / "history.json").read_text())
        best_epoch = checkpoint["epoch"]
        save_json(destination / "history.json", history)
        print("Reusing original trained seed 4213 / epoch 26", flush=True)
    else:
        options = config["training"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=options["lr"],
            weight_decay=options["weight_decay"],
        )
        inputs = [
            (
                card["features"][card["split"] == 0].to(device),
                card["masks"][card["split"] == 0].to(device),
            )
            for card in cards.values()
        ]
        generator = torch.Generator().manual_seed(config["seed"] + 17)
        history, best, best_epoch, weights = [], (-1, -float("inf")), 0, None
        loop = tqdm(range(1, epochs + 1), desc=f"{kind}/{seed}", unit="epoch")
        for epoch in loop:
            model.train()
            beta = (
                1 / (ROWS * HIDDEN)
                if kind == "cvae_elbo"
                else options["beta"] * min(1, epoch / options["warmup"])
            )
            orders = [
                torch.randperm(len(x), generator=generator) for x, _ in inputs
            ]
            for start in range(0, len(inputs[0][0]), options["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for index, (x, masks) in enumerate(inputs):
                    selected = orders[index][
                        start : start + options["batch_size"]
                    ]
                    losses.append(
                        objective(
                            model,
                            model(index, x[selected], True),
                            masks[selected],
                            config,
                            beta,
                        )[0]
                    )
                torch.stack(losses).mean().backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), 1, error_if_nonfinite=True
                )
                optimizer.step()
            model.eval()
            row = {"epoch": epoch, "optimization_beta": beta}
            for role, split in ((0, "train"), (1, "validation")):
                values, _ = metrics(model, cards, config, role, device)
                row.update({f"{split}_{k}": v for k, v in values.items()})
            history.append(row)
            rank = row["validation_iou"], -row["validation_loss"]
            if rank > best:
                best, best_epoch = rank, epoch
                weights = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                save_torch(
                    destination / "best.pt", {"model": weights, "epoch": epoch}
                )
            save_json(destination / "history.json", history)
            loop.set_postfix(
                iou=f"{rank[0]:.4f}", kl=f"{row['validation_kl']:.3f}"
            )
            if epoch - best_epoch >= options["patience"]:
                break
        model.load_state_dict(weights)
    model.eval()
    save_torch(
        destination / "best.pt",
        {
            "model": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "epoch": best_epoch,
        },
    )
    test_metrics, per_task = metrics(model, cards, config, 2, device)
    prior = prior_metrics(model, config, device)
    save_json(
        destination / "diagnostics.json",
        {
            "best_epoch": best_epoch,
            "bank_test": test_metrics,
            "per_task": per_task,
            "prior": prior,
        },
    )
    save_json(destination / "COMPLETE.json", {"epoch": best_epoch})
    print(f"DONE {kind}/{seed}: {test_metrics}", flush=True)


def load_model(kind, seed, config, device, model_factory=Network):
    model = model_factory(kind, config).to(device)
    checkpoint = torch.load(
        HOME / "runs" / f"{kind}_{seed}" / "best.pt",
        weights_only=True,
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval().requires_grad_(False)


def choose_head(model, source, config, data, options, device, routing="zero"):
    """Choose among all 12 frozen branches without knowing a held-out label."""
    with torch.no_grad():
        latent = torch.zeros(1, model.z_dim)
        if routing == "prior":
            generator = torch.Generator().manual_seed(
                config["seed"] + 9001 + (source[0][0] == "B")
            )
            latent = torch.randn(
                options["z_starts"], model.z_dim, generator=generator
            )
        latent = latent.to(device)
        scores = torch.cat(
            [
                model.decode(latent, index)
                for index in range(len(tasks(config)))
            ]
        )
        unique, inverse = torch.unique(
            topk(scores).cpu().flatten(1), dim=0, return_inverse=True
        )
    result = fresh(
        unique.reshape(-1, ROWS, HIDDEN),
        source,
        data,
        options,
        config["seed"] + 710009,
        device,
        False,
        mlp=config["bank"],
    )
    risks = result["bce"].mean((1, 2))[inverse.cpu()]
    risks = risks.reshape(len(tasks(config)), len(latent)).min(1).values
    return int(risks.argmin()), risks.tolist()


def transfer_one(
    kind,
    seed,
    config,
    cards,
    data,
    device,
    optimize,
    routing="zero",
    model_factory=Network,
):
    stage = (
        f"transfer_{config['evaluation'].get('z_method', 'gumbel')}"
        if optimize
        else "transfer_prior"
    )
    if kind == "reverse" and routing == "prior":
        stage += "_prior_head"
    destination = HOME / "runs" / f"{kind}_{seed}" / stage
    if (destination / "COMPLETE.json").exists():
        return
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=False)
    model = load_model(kind, seed, config, device, model_factory)
    frozen = {
        key: value.cpu().clone() for key, value in model.state_dict().items()
    }
    options = config["evaluation"]
    reports = {}
    for family in "AB":
        output = destination / family
        output.mkdir()
        source = tasks(config, family=family)
        head, head_risks = (0, [])
        if kind == "reverse":
            head, head_risks = choose_head(
                model, source, config, data, options, device, routing
            )
        decoder = BoundDecoder(model, family, head)
        print(f"TRANSFER {kind}/{seed}/{family}, head={head}", flush=True)
        if optimize:
            scores, initial, latent, steps = search(
                decoder, source, data, config, options, output, device, True
            )
        else:
            generator = torch.Generator().manual_seed(
                config["seed"] + 9001 + (family == "B")
            )
            latent = torch.randn(
                options["z_starts"], model.z_dim, generator=generator
            )
            with torch.no_grad():
                initial = scores = decoder(latent.to(device)).cpu()
            steps = torch.zeros(len(latent), dtype=torch.long)
        masks = topk(scores)
        zero = topk(decoder(torch.zeros(1, model.z_dim, device=device))).cpu()
        frequency = topk(
            torch.cat(
                [
                    card["masks"][card["split"] == 0]
                    for task, card in cards.items()
                    if task[0] == family
                ]
            ).mean(0, keepdim=True)
        )
        all_masks = torch.cat((masks, topk(initial), zero, frequency))
        unique, inverse = torch.unique(
            all_masks.flatten(1), dim=0, return_inverse=True
        )
        selection = fresh(
            unique.reshape(-1, ROWS, HIDDEN),
            source,
            data,
            options,
            config["seed"] + 710009,
            device,
            False,
            mlp=config["bank"],
        )
        risks = selection["bce"].mean((1, 2))[inverse]
        selected = int(risks[: len(masks)].argmin())
        random = int(risks[len(masks) : 2 * len(masks)].argmin())
        with torch.no_grad():
            swap = topk(
                model.decode(
                    latent[selected : selected + 1].to(device),
                    head,
                    "B" if family == "A" else "A",
                )
            ).cpu()
        candidates = torch.cat(
            (
                masks[selected : selected + 1],
                topk(initial)[random : random + 1],
                zero,
                frequency,
                swap,
            )
        )
        names = (
            "optimized",
            "best_prior",
            "zero",
            "condition_only_frequency",
            "swapped_condition",
        )
        # This disk write precedes every access to held-out task targets.
        save_torch(
            output / "sealed_masks.pt",
            {
                "masks": candidates,
                "names": names,
                "selected": selected,
                "z": latent,
                "source_validation": risks,
                "head": head,
            },
        )
        result = fresh(
            candidates,
            tasks(config, "test", family),
            data,
            options,
            config["seed"] + 910019,
            device,
            True,
            final=True,
            mlp=config["bank"],
        )
        iou, variants, orders = matches(candidates, source[0])
        save_torch(
            output / "results.pt",
            {
                **result,
                "iou": iou,
                "variants": variants,
                "orders": orders,
                "probabilities": scores[selected].sigmoid(),
            },
        )
        reports[family] = {
            "head": head,
            "head_source_validation": head_risks,
            "selected_search_step": int(steps[selected]),
            "methods": {
                name: {
                    "iou": float(iou[index]),
                    **{
                        key: float(result[key][index].mean())
                        for key in ("bce", "excess_bce", "accuracy")
                    },
                }
                for index, name in enumerate(names)
            },
        }
        print(
            f"RESULT {kind}/{seed}/{family}: {reports[family]['methods']}",
            flush=True,
        )
    assert all(
        torch.equal(value.cpu(), frozen[key])
        for key, value in model.state_dict().items()
    )
    save_json(destination / "metrics.json", reports)
    save_json(
        destination / "COMPLETE.json",
        {
            "decoder_frozen": True,
            "optimized": optimize,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "transfer"))
    parser.add_argument(
        "--kinds", nargs="+", default=list(KINDS), choices=KINDS
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[4213, 4214, 4215]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--routing", choices=("zero", "prior"), default="zero")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config, cards, data = prepare(args.device)
    if args.patience is not None:
        config["training"]["patience"] = args.patience
    HOME.joinpath("runs").mkdir(exist_ok=True)
    started = time.monotonic()
    for kind in args.kinds:
        for seed in args.seeds:
            if args.stage == "train":
                train_one(
                    kind,
                    seed,
                    config,
                    cards,
                    args.device,
                    args.epochs,
                    args.suffix,
                )
            else:
                transfer_one(
                    kind,
                    seed,
                    config,
                    cards,
                    data,
                    args.device,
                    args.optimize,
                    args.routing,
                )
    print(f"Completed in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
