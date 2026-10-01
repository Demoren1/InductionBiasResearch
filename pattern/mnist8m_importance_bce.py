"""MNIST8m digit-vs-rest agreement using VAEs trained on importance maps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from deepsets_z.mnist8m.meta_u import ImageSum, load_pool
from pattern.models.cvae import CVAE


CHECKPOINT = Path("deepsets_z/mnist8m/outputs/meta_u/full_v1/learned_seed42_best.pt")
DATA = Path("datasets/mnist8m")
FEATURES, HIDDEN, K = 100, 30, 600
LATENT, WIDTH = 16, 256
PAIRS = ((0, 1), (3, 8), (4, 9), (5, 6))
COEFFICIENTS = (0.0, 1.0, 10.0)


def save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def exact_masks(n: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    random = torch.rand(n, FEATURES * HIDDEN, generator=generator, device=device)
    result = torch.zeros_like(random)
    result.scatter_(1, random.topk(K, dim=1).indices, 1.)
    return result.reshape(n, FEATURES, HIDDEN)


def topk(logits: torch.Tensor) -> torch.Tensor:
    shape = logits.shape
    flat = logits.reshape(-1, FEATURES * HIDDEN)
    mask = torch.zeros_like(flat)
    mask.scatter_(-1, flat.topk(K, dim=-1).indices, 1.)
    return mask.reshape(shape)


def features(out: Path, device: torch.device, pair: tuple[int, int]) -> dict:
    path = out / "features.pt"
    if path.exists():
        result = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(result["pair"]) != pair:
            raise ValueError(f"different digit pair in {path}")
        return result
    model = ImageSum(learn_u=True, rank=16, seed=42).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    model.eval()
    result = {"pair": list(pair)}
    for split, count in (("train", 500), ("validation", 200), ("test", 200)):
        pixels, digits = load_pool(DATA, seed=42, per_digit=count,
                                   split=split, device=device)
        with torch.no_grad():
            vectors = torch.cat([model.encode(x.float().div(255)).cpu()
                                 for x in pixels.split(512)])
        result[split] = (vectors, digits.cpu())
    save(result, path)
    return result


def balanced_batch(pool: tuple[torch.Tensor, torch.Tensor], digit: int,
                   count: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x, labels = pool
    positive = torch.nonzero(labels == digit).flatten()
    negative = torch.nonzero(labels != digit).flatten()
    p = positive[torch.randint(len(positive), (count // 2,), generator=generator,
                               device=x.device)]
    n = negative[torch.randint(len(negative), (count - len(p),), generator=generator,
                               device=x.device)]
    ids = torch.cat((p, n))
    ids = ids[torch.randperm(count, generator=generator, device=x.device)]
    return x[ids], (labels[ids] == digit).float()


def predict(x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor,
            bias: torch.Tensor, readout: torch.Tensor,
            offset: torch.Tensor) -> torch.Tensor:
    hidden = torch.tanh(torch.einsum("bf,...fh->...bh", x, weight * mask) +
                        bias[..., None, :])
    return (hidden * readout[..., None, :]).sum(-1) + offset[..., None]


def balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels.expand_as(logits),
                                              reduction="none")
    return .5 * (loss[..., labels.bool()].mean(-1) +
                 loss[..., ~labels.bool()].mean(-1))


def balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    predictions = logits > 0
    positive = labels.bool()
    return .5 * (predictions[..., positive].float().mean(-1) +
                 (~predictions[..., ~positive]).float().mean(-1))


def build_bank(out: Path, data: dict, device: torch.device, *, candidates: int,
               max_steps: int, task_indices: tuple[int, ...] = (0, 1)) -> None:
    train = tuple(t.to(device) for t in data["train"])
    validation = tuple(t.to(device) for t in data["validation"])
    for task, digit in enumerate(data["pair"]):
        if task not in task_indices:
            continue
        path = out / f"bank_task{task}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (payload["settings"]["candidates"] != candidates or
                    payload["settings"]["max_steps"] != max_steps or
                    payload["settings"]["top_fraction"] != .1):
                raise ValueError(f"bank protocol mismatch: {path}")
            continue
        vg = torch.Generator(device=device).manual_seed(9300 + digit)
        vx, vy = balanced_batch(validation, digit, 512, vg)
        maps, masks, losses = [], [], []
        for group in tqdm(range(0, candidates, 32), desc=f"bank digit {digit}",
                          unit="group", mininterval=2):
            size = min(32, candidates - group)
            mask = exact_masks(size, 100000 + digit * 10000 + group, device)
            generator = torch.Generator(device=device).manual_seed(
                200000 + digit * 10000 + group)
            weight = torch.nn.Parameter(torch.randn(size, FEATURES, HIDDEN,
                                                     generator=generator,
                                                     device=device) * .08)
            bias = torch.nn.Parameter(torch.zeros(size, HIDDEN, device=device))
            readout = torch.nn.Parameter(torch.randn(size, HIDDEN,
                                                      generator=generator,
                                                      device=device) * .08)
            offset = torch.nn.Parameter(torch.zeros(size, device=device))
            optimizer = torch.optim.Adam((weight, bias, readout, offset), lr=.005)
            best = torch.full((size,), float("inf"), device=device)
            best_weight = torch.zeros_like(weight)
            stale = 0
            for step in range(1, max_steps + 1):
                x, y = balanced_batch(train, digit, 64, generator)
                logits = predict(x, mask, weight, bias, readout, offset)
                loss = F.binary_cross_entropy_with_logits(
                    logits, y.expand_as(logits), reduction="none").mean(-1)
                optimizer.zero_grad(set_to_none=True)
                loss.sum().backward()
                optimizer.step()
                if step % 100 == 0:
                    with torch.no_grad():
                        logits = predict(vx, mask, weight, bias, readout, offset)
                        val = balanced_bce(logits, vy)
                        improved = val < best - 1e-4
                        best = torch.where(improved, val, best)
                        best_weight = torch.where(improved[:, None, None],
                                                  weight.detach(), best_weight)
                        stale = 0 if improved.any() else stale + 1
                    if stale >= 5:
                        break
            importance = (best_weight * mask).abs()
            importance /= importance.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            maps.append(importance.cpu())
            masks.append(mask.cpu())
            losses.append(best.cpu())
        maps = torch.cat(maps)
        masks = torch.cat(masks)
        losses = torch.cat(losses)
        selected = losses.argsort()[:math.ceil(candidates * .1)]
        save({"importance": maps[selected], "masks": masks[selected],
              "all_validation_bce": losses, "selected_indices": selected,
              "digit": digit,
              "settings": {"candidates": candidates, "max_steps": max_steps,
                           "top_fraction": .1, "batch": 64,
                           "early_stop": "5 validation checks of 100 steps",
                           "importance": "abs(W * mask) / max(abs(W * mask))"}}, path)


def train_vaes(out: Path, device: torch.device, *, epochs: int,
               task_indices: tuple[int, ...] = (0, 1)) -> None:
    for task in task_indices:
        path = out / f"vae_task{task}.pt"
        if path.exists():
            continue
        maps = torch.load(out / f"bank_task{task}.pt", map_location="cpu",
                          weights_only=True)["importance"].flatten(1)
        generator = torch.Generator().manual_seed(3130 + task)
        order = torch.randperm(len(maps), generator=generator)
        n_val = max(32, round(.15 * len(maps)))
        train = maps[order[n_val:]].to(device)
        validation = maps[order[:n_val]].to(device)
        torch.manual_seed(9173 + task)
        model = CVAE(FEATURES * HIDDEN, LATENT, WIDTH, cond_dim=0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        best, stale, best_state, history = float("inf"), 0, None, []
        for epoch in tqdm(range(1, epochs + 1), desc=f"vae task {task}",
                          unit="epoch", mininterval=2):
            model.train()
            for block in train[torch.randperm(len(train), device=device)].split(64):
                mu, logvar = model.encode(block, block.new_zeros(len(block), 0))
                z = model.reparameterize(mu, logvar)
                logits = model.decode(z, block.new_zeros(len(block), 0))
                reconstruction = F.binary_cross_entropy_with_logits(
                    logits, block, reduction="none").sum(-1).mean()
                kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
                loss = reconstruction + .1 * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                mu, logvar = model.encode(validation,
                                           validation.new_zeros(len(validation), 0))
                logits = model.decode(mu, validation.new_zeros(len(validation), 0))
                reconstruction = F.binary_cross_entropy_with_logits(
                    logits, validation, reduction="none").sum(-1).mean()
                kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
                score = float(reconstruction + .1 * kl)
            history.append((epoch, score))
            if score < best - .01:
                best, stale = score, 0
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            else:
                stale += 1
            if stale >= 25:
                break
        save({"model": best_state, "best_loss": best,
              "history": torch.tensor(history),
              "config": {"mask_dim": FEATURES * HIDDEN, "latent_dim": LATENT,
                         "hidden": WIDTH, "cond_dim": 0}}, path)


def load_vaes(out: Path, device: torch.device) -> list[CVAE]:
    result = []
    for task in range(2):
        payload = torch.load(out / f"vae_task{task}.pt", map_location="cpu",
                             weights_only=True)
        model = CVAE(**payload["config"]).to(device)
        model.load_state_dict(payload["model"])
        model.eval().requires_grad_(False)
        result.append(model)
    return result


def decode(models: list[CVAE], z: torch.Tensor) -> torch.Tensor:
    return torch.stack([model.decode(z[i], z.new_zeros(z.shape[1], 0)).reshape(
        z.shape[1], FEATURES, HIDDEN) for i, model in enumerate(models)])


def search(out: Path, data: dict, device: torch.device, *, coefficient: float,
           starts: int, max_steps: int) -> None:
    path = out / f"search_lambda{coefficient:g}.pt"
    if path.exists():
        return
    models = load_vaes(out, device)
    train = tuple(t.to(device) for t in data["train"])
    validation = tuple(t.to(device) for t in data["validation"])
    digits = data["pair"]
    torch.manual_seed(50000 + int(coefficient * 100))
    z = torch.nn.Parameter(torch.randn(2, starts, LATENT, device=device))
    with torch.no_grad():
        initial = decode(models, z)
        orders = torch.arange(HIDDEN, device=device).repeat(2, starts, 1)
        for start in range(starts):
            cost = torch.cdist(initial[0, start].T,
                               initial[1, start].T).square().cpu().numpy()
            rows, columns = linear_sum_assignment(cost)
            orders[1, start, torch.as_tensor(rows, device=device)] = torch.as_tensor(
                columns, device=device)

    def aligned() -> torch.Tensor:
        raw = decode(models, z)
        return torch.gather(raw, -1, orders[:, :, None, :].expand_as(raw))

    shape = (2, starts)
    weight = torch.nn.Parameter(torch.randn(*shape, FEATURES, HIDDEN,
                                            device=device) * .08)
    bias = torch.nn.Parameter(torch.zeros(*shape, HIDDEN, device=device))
    readout = torch.nn.Parameter(torch.randn(*shape, HIDDEN, device=device) * .08)
    offset = torch.nn.Parameter(torch.zeros(shape, device=device))
    optimizer = torch.optim.Adam([{"params": [z], "lr": .02},
                                  {"params": [weight, bias, readout, offset],
                                   "lr": .003}])
    train_gen = torch.Generator(device=device).manual_seed(51000)
    val_gen = torch.Generator(device=device).manual_seed(52000)
    validation_batches = [balanced_batch(validation, digit, 512, val_gen)
                          for digit in digits]
    best = torch.full((starts,), float("inf"), device=device)
    best_z = z.detach().clone()
    stale, history = 0, []
    for step in tqdm(range(1, max_steps + 1), desc=f"search λ={coefficient:g}",
                     unit="step", mininterval=2):
        raw = aligned()
        flat = raw.flatten(-2)
        hard = topk(raw)
        kth = flat.topk(K, dim=-1).values[..., -1:].detach()
        soft = torch.sigmoid((flat - kth) / .25).reshape_as(raw)
        mask = hard + soft - soft.detach()
        batches = [balanced_batch(train, digit, 64, train_gen)
                   for digit in digits]
        task_losses = []
        for task, (x, y) in enumerate(batches):
            logits = predict(x, mask[task], weight[task], bias[task],
                             readout[task], offset[task])
            task_losses.append(F.binary_cross_entropy_with_logits(
                logits, y.expand_as(logits), reduction="none").mean(-1))
        task_loss = torch.stack(task_losses).mean(0)
        agreement = (raw - raw.mean(0, keepdim=True)).square().mean((0, 2, 3))
        loss = (task_loss + coefficient * agreement).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            z *= (10 / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
        if step % 250 == 0:
            with torch.no_grad():
                raw = aligned()
                val_losses = []
                for task, (x, y) in enumerate(validation_batches):
                    logits = predict(x, topk(raw[task]), weight[task], bias[task],
                                     readout[task], offset[task])
                    val_losses.append(balanced_bce(logits, y))
                val_bce = torch.stack(val_losses).mean(0)
                agreement = (raw - raw.mean(0, keepdim=True)).square().mean((0, 2, 3))
                score = val_bce + coefficient * agreement
                improved = score < best - .0001
                best = torch.where(improved, score, best)
                best_z = torch.where(improved[None, :, None], z.detach(), best_z)
                stale = 0 if improved.any() else stale + 1
                history.append((step, float(best.min()), float(val_bce.mean()),
                                float(agreement.mean())))
                if stale >= 12:
                    break
    with torch.no_grad():
        raw = torch.gather(decode(models, best_z), -1,
                           orders[:, :, None, :].expand(2, starts, FEATURES, HIDDEN))
    save({"z": best_z.cpu(), "logits": raw.cpu(), "chosen_start": int(best.argmin()),
          "history": torch.tensor(history), "coefficient": coefficient,
          "steps": step, "plateau": stale >= 12,
          "alignment": "fixed Hungarian from initial raw decoder logits"}, path)


def evaluate(out: Path, data: dict, device: torch.device, *, max_steps: int) -> dict:
    path = out / "evaluation.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    train = tuple(t.to(device) for t in data["train"])
    validation = tuple(t.to(device) for t in data["validation"])
    test = tuple(t.to(device) for t in data["test"])
    digits = data["pair"]
    names, masks = [], []
    for coefficient in COEFFICIENTS:
        result = torch.load(out / f"search_lambda{coefficient:g}.pt",
                            map_location="cpu", weights_only=True)
        raw = result["logits"][:, result["chosen_start"]].to(device)
        names.append(f"consensus_lambda{coefficient:g}")
        masks.append(topk(raw.mean(0)))
        if coefficient in (0.0, 10.0):
            for task in range(2):
                names.append(f"vae{task}_lambda{coefficient:g}")
                masks.append(topk(raw[task]))
    for task in range(2):
        bank = torch.load(out / f"bank_task{task}.pt", map_location="cpu",
                          weights_only=True)
        names.append(f"bank_task{task}_best")
        masks.append(bank["masks"][0].to(device))
    models = load_vaes(out, device)
    for task, model in enumerate(models):
        with torch.no_grad():
            prior = model.decode(torch.zeros(1, LATENT, device=device),
                                 torch.zeros(1, 0, device=device)).reshape(FEATURES, HIDDEN)
        names.append(f"vae{task}_z0")
        masks.append(topk(prior))
    names.extend(f"random_{i:02d}" for i in range(16))
    masks.extend(exact_masks(16, 20261099, device))
    names.append("dense")
    masks.append(torch.ones(FEATURES, HIDDEN, device=device))
    masks = torch.stack(masks)
    repeats = 2
    expanded = masks.repeat_interleave(repeats, dim=0)
    count = len(expanded)
    torch.manual_seed(20261101)
    weight = torch.nn.Parameter(torch.randn(2, count, FEATURES, HIDDEN,
                                            device=device) * .08)
    bias = torch.nn.Parameter(torch.zeros(2, count, HIDDEN, device=device))
    readout = torch.nn.Parameter(torch.randn(2, count, HIDDEN, device=device) * .08)
    offset = torch.nn.Parameter(torch.zeros(2, count, device=device))
    optimizer = torch.optim.Adam((weight, bias, readout, offset), lr=.003)
    train_gen = torch.Generator(device=device).manual_seed(20261102)
    val_gen = torch.Generator(device=device).manual_seed(20261103)
    val_batches = [balanced_batch(validation, digit, 512, val_gen)
                   for digit in digits]
    best = torch.full((2, count), float("inf"), device=device)
    best_values = None
    stale = 0
    for step in tqdm(range(1, max_steps + 1), desc="evaluate masks",
                     unit="step", mininterval=2):
        losses = []
        for task, digit in enumerate(digits):
            x, y = balanced_batch(train, digit, 128, train_gen)
            logits = predict(x, expanded, weight[task], bias[task],
                             readout[task], offset[task])
            losses.append(F.binary_cross_entropy_with_logits(
                logits, y.expand_as(logits), reduction="none").mean(-1))
        loss = torch.stack(losses).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            with torch.no_grad():
                val = torch.stack([
                    balanced_bce(predict(x, expanded, weight[t], bias[t],
                                         readout[t], offset[t]), y)
                    for t, (x, y) in enumerate(val_batches)])
                improved = val < best - .0001
                best = torch.where(improved, val, best)
                values = (weight.detach(), bias.detach(), readout.detach(),
                          offset.detach())
                if best_values is None:
                    best_values = tuple(value.clone() for value in values)
                else:
                    best_values = tuple(torch.where(
                        improved.reshape(*improved.shape,
                                         *((1,) * (value.ndim - improved.ndim))),
                        value, prior)
                        for value, prior in zip(values, best_values))
                stale = 0 if improved.any() else stale + 1
                if stale >= 15:
                    break
    assert best_values is not None
    metrics = {"bce": [], "balanced_accuracy": []}
    for task, digit in enumerate(digits):
        x, labels = test
        target = (labels == digit).float()
        with torch.no_grad():
            logits = predict(x, expanded,
                             *(value[task] for value in best_values))
            metrics["bce"].append(balanced_bce(logits, target).cpu().reshape(
                len(masks), repeats))
            metrics["balanced_accuracy"].append(
                balanced_accuracy(logits, target).cpu().reshape(len(masks), repeats))
    payload = {"names": names, "masks": masks.cpu(),
               "metrics": {key: torch.stack(value) for key, value in metrics.items()},
               "validation_bce": best.cpu().reshape(2, len(masks), repeats),
               "pair": list(digits),
               "settings": {"max_steps": max_steps, "actual_steps": step,
                            "repeats": repeats, "test_images_per_digit": 200}}
    save(payload, path)
    return payload


def report(out: Path, evaluation: dict) -> None:
    names = evaluation["names"]
    val = evaluation["validation_bce"].numpy()
    random = [i for i, name in enumerate(names) if name.startswith("random_")]
    best_random = random[int(val[:, random].mean((0, 2)).argmin())]
    rows = []
    for key in ("balanced_accuracy", "bce"):
        metric = evaluation["metrics"][key].numpy()
        row = {"metric": key}
        for name in ("consensus_lambda0", "consensus_lambda1",
                     "consensus_lambda10", "dense"):
            row[name] = float(metric[:, names.index(name)].mean())
        row["random"] = float(metric[:, random].mean())
        row["random_best16"] = float(metric[:, best_random].mean())
        row["best_bank_per_task"] = float(np.mean([
            metric[task, names.index(f"bank_task{task}_best")].mean()
            for task in range(2)]))
        row["single_optimized_per_task"] = float(np.mean([
            metric[task, names.index(f"vae{task}_lambda0")].mean()
            for task in range(2)]))
        rows.append(row)
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    keys = ("random_best16", "best_bank_per_task", "single_optimized_per_task",
            "consensus_lambda0", "consensus_lambda1", "consensus_lambda10", "dense")
    labels = ("Случайная", "Банк", "Один VAE", "λ=0", "λ=1", "λ=10", "Плотный")
    for ax, row in zip(axes, rows):
        ax.bar(range(len(keys)), [row[key] for key in keys])
        ax.set_xticks(range(len(keys)), labels, rotation=40, ha="right")
        ax.set_title("Сбалансированная точность ↑" if row["metric"] ==
                     "balanced_accuracy" else "BCE ↓")
    fig.tight_layout()
    fig.savefig(out / "quality.png", dpi=170)
    plt.close(fig)
    pair = evaluation["pair"]
    bank_count = torch.load(out / "bank_task0.pt", map_location="cpu",
                            weights_only=True)["settings"]["candidates"]
    lines = [f"# MNIST8m: agreement VAE для цифр {pair[0]} и {pair[1]}", "",
             f"Для каждой цифры обучены {bank_count} masked MLP на задаче «эта цифра "
             "против остальных». Из 10% MLP с минимальной validation BCE "
             "взяты карты |W·M|/max(|W·M|); на картах каждой цифры обучен "
             "отдельный VAE. При поиске VAE заморожены; обучаются два z и "
             "две головы по BCE своих задач и MSE выровненных логитов. "
             "После поиска каждая маска проверена с новыми головами на "
             "изображениях отдельного тестового блока.", "",
             "| Маска | Сбалансированная точность ↑ | BCE ↓ |",
             "|---|---:|---:|"]
    display = (("random_best16", "Лучшая из 16 случайных"),
               ("best_bank_per_task", "Лучшая маска банка"),
               ("single_optimized_per_task", "Один VAE на задачу"),
               ("consensus_lambda0", "Общая без agreement"),
               ("consensus_lambda1", "Agreement λ=1"),
               ("consensus_lambda10", "Agreement λ=10"),
               ("dense", "Плотный слой"))
    for key, label in display:
        lines.append(f"| {label} | {rows[0][key]:.4f} | {rows[1][key]:.4f} |")
    lines.extend(["", "Метрики усреднены по двум цифрам и двум новым обучениям "
                  "головы. Сбалансированная точность придаёт одинаковый вес "
                  "положительному и отрицательному классу. Случайная маска "
                  "выбрана по валидации. Плотный слой имеет 3000 связей, "
                  "остальные — 600.", "", "![Качество](quality.png)", ""])
    (out / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("pattern/outputs/mnist8m_importance_bce/pair01"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pair-index", type=int, choices=range(len(PAIRS)), default=0)
    parser.add_argument("--bank-candidates", type=int, default=4096)
    parser.add_argument("--bank-steps", type=int, default=1200)
    parser.add_argument("--vae-epochs", type=int, default=200)
    parser.add_argument("--search-steps", type=int, default=6000)
    parser.add_argument("--evaluation-steps", type=int, default=2500)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--stage", choices=("all", "features", "bank", "vae",
                                            "search", "evaluate"), default="all")
    parser.add_argument("--task-index", type=int, choices=(0, 1))
    parser.add_argument("--coefficient", type=float, choices=COEFFICIENTS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(args.out, torch.load(args.out / "evaluation.pt", map_location="cpu",
                                    weights_only=True))
        return
    device = torch.device(args.device)
    data = features(args.out, device, PAIRS[args.pair_index])
    if args.stage == "features":
        return
    tasks = (args.task_index,) if args.task_index is not None else (0, 1)
    if args.stage == "bank":
        build_bank(args.out, data, device, candidates=args.bank_candidates,
                   max_steps=args.bank_steps, task_indices=tasks)
        return
    if args.stage == "vae":
        train_vaes(args.out, device, epochs=args.vae_epochs, task_indices=tasks)
        return
    if args.stage == "search":
        coefficients = (args.coefficient,) if args.coefficient is not None else COEFFICIENTS
        for coefficient in coefficients:
            search(args.out, data, device, coefficient=coefficient,
                   starts=8, max_steps=args.search_steps)
        return
    if args.stage == "evaluate":
        evaluation = evaluate(args.out, data, device, max_steps=args.evaluation_steps)
        report(args.out, evaluation)
        print(args.out / "report.md", flush=True)
        return
    build_bank(args.out, data, device, candidates=args.bank_candidates,
               max_steps=args.bank_steps)
    train_vaes(args.out, device, epochs=args.vae_epochs)
    for coefficient in COEFFICIENTS:
        search(args.out, data, device, coefficient=coefficient,
               starts=8, max_steps=args.search_steps)
    evaluation = evaluate(args.out, data, device, max_steps=args.evaluation_steps)
    report(args.out, evaluation)
    print(args.out / "report.md", flush=True)


if __name__ == "__main__":
    main()
