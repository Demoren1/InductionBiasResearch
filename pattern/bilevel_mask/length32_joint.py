"""Joint structure/weight search for length-32, length-4 pattern tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pattern.evaluation.decoder_agreement import align_columns, hard_topk, soft_topk


TRAIN_PATTERNS = ("0000", "1111", "0001", "1000", "1110", "0111", "0010", "0100", "1101", "1011")
VALIDATION_PATTERNS = ("0110", "1001")
TEST_PATTERNS = ("0011", "1100", "0101", "1010")


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


@dataclass(frozen=True)
class Length32Config:
    seed: int = 42
    seq_len: int = 32
    pattern_len: int = 4
    hidden: int = 32
    latent_dim: int = 4
    generator_width: int = 16
    k_active: int = 128
    restarts: int = 64
    outer_steps: int = 100
    inner_max_steps: int = 300
    eval_inner_max_steps: int = 1000
    inner_patience: int = 15
    inner_min_delta: float = 1e-5
    weight_steps_per_z: int = 5
    frozen_pretrain_steps: int = 300
    eval_weight_steps: int = 2000
    weight_lr: float = 0.001
    z_lr: float = 0.1
    generator_lr: float = 0.01
    z_radius: float = 8.0
    temperature_start: float = 1.0
    temperature_end: float = 0.05
    binary_penalty: float = 0.01
    support_per_class: int = 512
    validation_per_class: int = 256
    query_per_class: int = 2048

    def __post_init__(self) -> None:
        if self.k_active != self.pattern_len * self.hidden:
            raise ValueError("k_active must equal pattern_len * hidden for the analytic control")
        if self.seq_len < self.pattern_len or self.hidden < self.seq_len - self.pattern_len + 1:
            raise ValueError("dimensions cannot represent every analytic window")
        integer_names = (
            "seq_len", "pattern_len", "hidden", "latent_dim", "generator_width", "k_active", "restarts",
            "outer_steps", "inner_max_steps", "eval_inner_max_steps", "inner_patience", "weight_steps_per_z", "frozen_pretrain_steps",
            "eval_weight_steps", "support_per_class", "validation_per_class", "query_per_class",
        )
        if any(getattr(self, name) <= 0 for name in integer_names):
            raise ValueError("integer configuration values must be positive")

    @property
    def mask_dim(self) -> int:
        return self.seq_len * self.hidden

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Generator(nn.Module):
    def __init__(self, config: Length32Config) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.network = nn.Sequential(
                nn.Linear(config.latent_dim, config.generator_width), nn.Tanh(),
                nn.Linear(config.generator_width, config.generator_width), nn.Tanh(),
                nn.Linear(config.generator_width, config.mask_dim),
            )
            nn.init.normal_(self.network[-1].weight, std=0.01)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class Weights(nn.Module):
    def __init__(self, tasks: int, restarts: int, config: Length32Config, device: torch.device, seed: int) -> None:
        super().__init__()
        rng = torch.Generator(device="cpu").manual_seed(seed)
        self.w1 = nn.Parameter((0.1 * torch.randn(tasks, restarts, config.seq_len, config.hidden, generator=rng)).to(device))
        self.b1 = nn.Parameter(torch.zeros(tasks, restarts, config.hidden, device=device))
        self.w2 = nn.Parameter((0.1 * torch.randn(tasks, restarts, config.hidden, generator=rng)).to(device))
        self.b2 = nn.Parameter(torch.zeros(tasks, restarts, device=device))

    def values(self, detach: bool = False) -> tuple[torch.Tensor, ...]:
        values = (self.w1, self.b1, self.w2, self.b2)
        return tuple(value.detach() for value in values) if detach else values

    def copy_values_(self, values: tuple[torch.Tensor, ...]) -> None:
        with torch.no_grad():
            for target, source in zip(self.values(), values):
                target.copy_(source)


@dataclass
class Data:
    patterns: tuple[str, ...]
    support_x: torch.Tensor
    support_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor


def labels(x01: torch.Tensor, pattern: str) -> torch.Tensor:
    bits = torch.tensor([int(bit) for bit in pattern], device=x01.device, dtype=x01.dtype)
    return (x01.unfold(-1, len(pattern), 1) == bits).all(-1).any(-1)


def _sample_balanced(pattern: str, per_class: int, config: Length32Config, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []
    n_positive = n_negative = 0
    while n_positive < per_class or n_negative < per_class:
        x01 = torch.randint(0, 2, (4096, config.seq_len), generator=rng, dtype=torch.float32)
        y = labels(x01, pattern)
        if n_positive < per_class:
            selected = x01[y][:per_class - n_positive]
            positives.append(selected)
            n_positive += selected.shape[0]
        if n_negative < per_class:
            selected = x01[~y][:per_class - n_negative]
            negatives.append(selected)
            n_negative += selected.shape[0]
    positive = torch.cat(positives)[:per_class]
    negative = torch.cat(negatives)[:per_class]
    x = torch.cat((positive, negative))
    y = torch.cat((torch.ones(per_class), torch.zeros(per_class)))
    order = torch.randperm(x.shape[0], generator=rng)
    return x[order].mul(2).sub(1), y[order]


def make_data(patterns: Sequence[str], config: Length32Config, device: torch.device) -> Data:
    """Generate one large matrix per task, then slice it into three roles."""
    counts = (config.support_per_class, config.validation_per_class, config.query_per_class)
    total = sum(counts)
    fields_x: list[list[torch.Tensor]] = [[], [], []]
    fields_y: list[list[torch.Tensor]] = [[], [], []]
    for pattern in patterns:
        x, y = _sample_balanced(pattern, total, config, _seed(config.seed, "data", pattern))
        positive = x[y == 1]
        negative = x[y == 0]
        offset = 0
        for index, count in enumerate(counts):
            part_x = torch.cat((positive[offset:offset + count], negative[offset:offset + count]))
            part_y = torch.cat((torch.ones(count), torch.zeros(count)))
            rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, "shuffle", pattern, index))
            order = torch.randperm(part_x.shape[0], generator=rng)
            fields_x[index].append(part_x[order])
            fields_y[index].append(part_y[order])
            offset += count
    xs = [torch.stack(value).to(device) for value in fields_x]
    ys = [torch.stack(value).to(device) for value in fields_y]
    return Data(tuple(patterns), xs[0], ys[0], xs[1], ys[1], xs[2], ys[2])


def analytic_mask(config: Length32Config, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(config.seq_len, config.hidden, device=device)
    windows = config.seq_len - config.pattern_len + 1
    for column in range(config.hidden):
        start = column % windows
        mask[start:start + config.pattern_len, column] = 1
    return mask


def forward(x: torch.Tensor, weights: Weights | tuple[torch.Tensor, ...], masks: torch.Tensor) -> torch.Tensor:
    w1, b1, w2, b2 = weights.values() if isinstance(weights, Weights) else weights
    hidden = F.relu(torch.einsum("tbi,trih->trbh", x, w1 * masks) + b1[:, :, None, :])
    return torch.einsum("trbh,trh->trb", hidden, w2) + b2[:, :, None]


def losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets[:, None].expand_as(logits), reduction="none").mean(-1)


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return hard

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, gradient


def masks(generator: Generator, z: torch.Tensor, config: Length32Config, temperature: float, mode: str) -> torch.Tensor:
    logits = generator(z)
    soft = soft_topk(logits, config.k_active, temperature)
    if mode == "soft":
        flat = soft
    elif mode == "hard":
        flat = hard_topk(logits, config.k_active)
    elif mode == "ste":
        flat = _STE.apply(hard_topk(logits, config.k_active), soft)
    else:
        raise ValueError(mode)
    return flat.reshape(*z.shape[:-1], config.seq_len, config.hidden)


def _project(z: torch.Tensor, radius: float) -> None:
    with torch.no_grad():
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        z.mul_(torch.clamp(radius / norm, max=1.0))


def _fit_weights(weights: Weights, optimizer: torch.optim.Optimizer, mask: torch.Tensor,
                 x: torch.Tensor, y: torch.Tensor, steps: int) -> float:
    value = math.nan
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = losses(forward(x, weights, mask), y).mean()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
    return value


def _freeze_generator(generator: Generator, frozen: bool) -> None:
    for parameter in generator.parameters():
        parameter.requires_grad_(not frozen)


def joint_search(
    generator: Generator, z: nn.Parameter, weights: Weights, data: Data,
    config: Length32Config, temperature: float,
) -> dict[str, float | int]:
    """Alternate weight fitting and one latent step until validation convergence."""
    _freeze_generator(generator, True)
    weight_optimizer = torch.optim.Adam(weights.parameters(), lr=config.weight_lr)
    z_optimizer = torch.optim.Adam([z], lr=config.z_lr)
    best = math.inf
    best_z = z.detach().clone()
    best_weights = tuple(value.detach().clone() for value in weights.values())
    stale = 0
    for step in range(1, config.inner_max_steps + 1):
        with torch.no_grad():
            hard = masks(generator, z, config, temperature, "hard")
        _fit_weights(weights, weight_optimizer, hard, data.support_x, data.support_y, config.weight_steps_per_z)

        z_optimizer.zero_grad(set_to_none=True)
        soft = masks(generator, z, config, temperature, "soft")
        value_tensor = losses(forward(data.validation_x, weights.values(detach=True), soft), data.validation_y).mean()
        value = float(value_tensor.detach())
        improved = value < best - config.inner_min_delta
        if value < best:
            best = value
            best_z.copy_(z.detach())
            best_weights = tuple(item.detach().clone() for item in weights.values())
        stale = 0 if improved else stale + 1
        value_tensor.backward()
        z_optimizer.step()
        _project(z, config.z_radius)
        if stale >= config.inner_patience:
            break
    with torch.no_grad():
        z.copy_(best_z)
    weights.copy_values_(best_weights)
    _freeze_generator(generator, False)
    return {"steps": step, "validation_bce": best, "converged": int(step < config.inner_max_steps)}


def frozen_search(
    generator: Generator, z: nn.Parameter, weights: Weights, data: Data,
    config: Length32Config, temperature: float, dense: bool,
) -> dict[str, float | int]:
    """Control: fit weights once, then optimize z while weights stay frozen."""
    _freeze_generator(generator, True)
    optimizer_w = torch.optim.Adam(weights.parameters(), lr=config.weight_lr)
    if dense:
        initial_mask = torch.ones(*z.shape[:2], config.seq_len, config.hidden, device=z.device)
    else:
        with torch.no_grad():
            initial_mask = masks(generator, z, config, temperature, "hard")
    _fit_weights(weights, optimizer_w, initial_mask, data.support_x, data.support_y, config.frozen_pretrain_steps)
    optimizer_z = torch.optim.Adam([z], lr=config.z_lr)
    best = math.inf
    best_z = z.detach().clone()
    stale = 0
    for step in range(1, config.inner_max_steps + 1):
        optimizer_z.zero_grad(set_to_none=True)
        soft = masks(generator, z, config, temperature, "soft")
        value_tensor = losses(forward(data.validation_x, weights.values(detach=True), soft), data.validation_y).mean()
        value = float(value_tensor.detach())
        improved = value < best - config.inner_min_delta
        if value < best:
            best = value
            best_z.copy_(z.detach())
        stale = 0 if improved else stale + 1
        value_tensor.backward()
        optimizer_z.step()
        _project(z, config.z_radius)
        if stale >= config.inner_patience:
            break
    with torch.no_grad():
        z.copy_(best_z)
    _freeze_generator(generator, False)
    return {"steps": step, "validation_bce": best, "converged": int(step < config.inner_max_steps)}


def _new_state(tasks: int, config: Length32Config, device: torch.device, tag: str) -> tuple[nn.Parameter, Weights]:
    rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, "z", tag))
    z = nn.Parameter(torch.randn(tasks, config.restarts, config.latent_dim, generator=rng).to(device))
    _project(z, config.z_radius)
    weights = Weights(tasks, config.restarts, config, device, _seed(config.seed, "w", tag))
    return z, weights


def _temperature(config: Length32Config, step: int) -> float:
    if config.outer_steps == 1:
        return config.temperature_end
    ratio = (step - 1) / (config.outer_steps - 1)
    return config.temperature_start * (config.temperature_end / config.temperature_start) ** ratio


def _gather(value: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    task = torch.arange(value.shape[0], device=value.device)
    return value[task, selected][:, None]


def train_generator(config: Length32Config, device: torch.device) -> tuple[Generator, list[dict[str, Any]]]:
    data = make_data(TRAIN_PATTERNS, config, device)
    generator = Generator(config).to(device)
    z, weights = _new_state(len(TRAIN_PATTERNS), config, device, "train")
    optimizer = torch.optim.Adam(generator.parameters(), lr=config.generator_lr)
    history: list[dict[str, Any]] = []
    for outer in range(1, config.outer_steps + 1):
        temperature = _temperature(config, outer)
        info = joint_search(generator, z, weights, data, config, temperature)
        optimizer.zero_grad(set_to_none=True)
        candidate = masks(generator, z.detach(), config, temperature, "ste")
        with torch.no_grad():
            val = losses(forward(data.validation_x, weights.values(detach=True), candidate.detach()), data.validation_y)
            selected = val.argmin(1)
        chosen_mask = _gather(candidate, selected)
        chosen_weights = tuple(_gather(value.detach(), selected) for value in weights.values())
        query = losses(forward(data.query_x, chosen_weights, chosen_mask), data.query_y).mean()
        logits = generator(z.detach())
        soft = soft_topk(logits, config.k_active, temperature)
        penalty = config.binary_penalty * (soft * (1 - soft)).mean()
        (query + penalty).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
        optimizer.step()
        row = {"outer_step": outer, "query_bce": float(query.detach()), "inner_validation_bce": info["validation_bce"],
               "inner_steps": info["steps"], "inner_converged": info["converged"], "temperature": temperature,
               "generator_gradient_norm": float(grad_norm)}
        history.append(row)
        if outer == 1 or outer % 5 == 0 or outer == config.outer_steps:
            print(f"L32 seed={config.seed} outer={outer}/{config.outer_steps} query={float(query):.6f} "
                  f"inner_val={float(info['validation_bce']):.6f} inner_steps={info['steps']}", flush=True)
    return generator, history


def _iou(mask: torch.Tensor, gold: torch.Tensor) -> float:
    aligned = align_columns(gold, mask)
    return float((aligned.bool() & gold.bool()).sum() / (aligned.bool() | gold.bool()).sum())


def fit_and_evaluate(candidate_masks: torch.Tensor, data: Data, config: Length32Config,
                     device: torch.device, method: str) -> dict[str, Any]:
    tasks, restarts = candidate_masks.shape[:2]
    weights = Weights(tasks, restarts, config, device, _seed(config.seed, "fair-eval-weights"))
    optimizer = torch.optim.Adam(weights.parameters(), lr=config.weight_lr)
    _fit_weights(weights, optimizer, candidate_masks, data.support_x, data.support_y, config.eval_weight_steps)
    with torch.no_grad():
        val = losses(forward(data.validation_x, weights, candidate_masks), data.validation_y)
        selected = val.argmin(1)
        chosen_masks = _gather(candidate_masks, selected)
        chosen_weights = tuple(_gather(value.detach(), selected) for value in weights.values())
        logits = forward(data.query_x, chosen_weights, chosen_masks).squeeze(1)
        query = F.binary_cross_entropy_with_logits(logits, data.query_y, reduction="none").mean(1)
        accuracy = ((logits > 0) == data.query_y.bool()).float().mean(1)
    gold = analytic_mask(config, device)
    ious = [_iou(mask, gold) for mask in chosen_masks.squeeze(1)]
    return {"method": method, "mean_query_bce": float(query.mean()), "mean_query_accuracy": float(accuracy.mean()),
            "mean_iou": sum(ious) / len(ious), "chosen_masks": chosen_masks.squeeze(1).cpu().tolist(),
            "tasks": [{"pattern": pattern, "query_bce": float(query[i]), "accuracy": float(accuracy[i]),
                       "iou": ious[i], "selected_restart": int(selected[i])}
                      for i, pattern in enumerate(data.patterns)]}


def evaluate(generator: Generator, patterns: Sequence[str], config: Length32Config,
             device: torch.device, split: str) -> dict[str, Any]:
    data = make_data(patterns, config, device)
    tasks = len(patterns)
    search_config = replace(config, inner_max_steps=config.eval_inner_max_steps)
    results: dict[str, Any] = {}
    convergence: dict[str, Any] = {}
    for strategy in ("joint", "frozen", "dense"):
        # Paired control: every strategy starts from identical z and weights.
        z, weights = _new_state(tasks, config, device, f"{split}-search")
        if strategy == "joint":
            info = joint_search(generator, z, weights, data, search_config, config.temperature_end)
        else:
            info = frozen_search(generator, z, weights, data, search_config, config.temperature_end, dense=strategy == "dense")
        with torch.no_grad():
            candidate = masks(generator, z, config, config.temperature_end, "hard")
        results[strategy] = fit_and_evaluate(candidate, data, config, device, strategy)
        convergence[strategy] = info
    rng = torch.Generator(device="cpu").manual_seed(_seed(config.seed, "random", split))
    scores = torch.rand(tasks, config.restarts, config.mask_dim, generator=rng).to(device)
    random_masks = hard_topk(scores, config.k_active).reshape(tasks, config.restarts, config.seq_len, config.hidden)
    results["random"] = fit_and_evaluate(random_masks, data, config, device, "random best-of-64")
    gold = analytic_mask(config, device)
    gold_masks = gold[None, None].expand(tasks, config.restarts, -1, -1).clone()
    results["analytic"] = fit_and_evaluate(gold_masks, data, config, device, "analytic")
    return {"patterns": list(patterns), "strategies": results, "convergence": convergence}


def run(config: Length32Config, output: str | Path, device: str | torch.device = "cuda") -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    torch.manual_seed(config.seed)
    started = time.monotonic()
    generator, history = train_generator(config, device)
    torch.save({"config": config.to_dict(), "generator": generator.state_dict(), "history": history}, output / "training.pt")
    evaluations = {
        "validation": evaluate(generator, VALIDATION_PATTERNS, config, device, "validation"),
        "test": evaluate(generator, TEST_PATTERNS, config, device, "test"),
    }
    summary = {"config": config.to_dict(), "history": history, "evaluations": evaluations,
               "generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
               "parallel_train_mlps": len(TRAIN_PATTERNS) * config.restarts, "seconds": time.monotonic() - started}
    with (output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    _report(summary, output / "RESULTS.md")
    _plot(summary, output / "summary.png")
    return output


def reevaluate(checkpoint: str | Path, output: str | Path, device: str | torch.device = "cuda",
               eval_inner_max_steps: int = 1000) -> Path:
    """Reuse a trained generator and rerun only the expensive search controls."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = replace(Length32Config(**payload["config"]), eval_inner_max_steps=eval_inner_max_steps)
    device = torch.device(device)
    generator = Generator(config).to(device)
    generator.load_state_dict(payload["generator"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    evaluations = {
        "validation": evaluate(generator, VALIDATION_PATTERNS, config, device, "validation"),
        "test": evaluate(generator, TEST_PATTERNS, config, device, "test"),
    }
    summary = {"config": config.to_dict(), "history": payload["history"], "evaluations": evaluations,
               "generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
               "parallel_train_mlps": len(TRAIN_PATTERNS) * config.restarts,
               "seconds": time.monotonic() - started, "reevaluated_from": str(checkpoint)}
    with (output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    _report(summary, output / "RESULTS.md")
    _plot(summary, output / "summary.png")
    return output


def _report(summary: dict[str, Any], path: Path) -> None:
    lines = ["# Length 32 / pattern length 4: joint structure search", "",
             f"Seed `{summary['config']['seed']}`; `{summary['parallel_train_mlps']}` parallel MLPs; "
             f"generator parameters `{summary['generator_parameters']}`.", "",
             "| Task split | Search | Query BCE | Accuracy | IoU to analytic |", "|---|---|---:|---:|---:|"]
    for split in ("validation", "test"):
        for method in ("joint", "frozen", "dense", "random", "analytic"):
            row = summary["evaluations"][split]["strategies"][method]
            lines.append(f"| {split} | {row['method']} | {row['mean_query_bce']:.6f} | "
                         f"{row['mean_query_accuracy']:.4f} | {row['mean_iou']:.4f} |")
    lines += ["", "| Search | Test z steps | Converged |", "|---|---:|---:|"]
    for method in ("joint", "frozen", "dense"):
        row = summary["evaluations"]["test"]["convergence"][method]
        lines.append(f"| {method} | {row['steps']} | {row['converged']} |")
    lines += ["", f"Wall time: `{summary['seconds']:.1f}` seconds."]
    path.write_text("\n".join(lines) + "\n")


def _plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    methods = ("joint", "frozen", "dense", "random", "analytic")
    test = summary["evaluations"]["test"]["strategies"]
    masks_to_plot = torch.tensor(test["joint"]["chosen_masks"])
    gold = analytic_mask(Length32Config(**summary["config"]), torch.device("cpu"))
    figure, axes = plt.subplots(2, 5, figsize=(14, 6.5))
    axes[0, 0].plot([row["outer_step"] for row in summary["history"]], [row["query_bce"] for row in summary["history"]])
    axes[0, 0].set(xlabel="Outer step", ylabel="Query BCE", title="Generator training")
    axes[0, 1].bar(methods, [test[name]["mean_query_bce"] for name in methods])
    axes[0, 1].set_yscale("log"); axes[0, 1].set_title("Held-out BCE"); axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 2].bar(methods, [test[name]["mean_iou"] for name in methods])
    axes[0, 2].set_ylim(0, 1.03); axes[0, 2].set_title("Held-out IoU"); axes[0, 2].tick_params(axis="x", rotation=25)
    axes[0, 3].plot([row["outer_step"] for row in summary["history"]], [row["inner_steps"] for row in summary["history"]])
    axes[0, 3].set(xlabel="Outer step", ylabel="Steps", title="Joint convergence")
    axes[0, 4].axis("off")
    for index in range(4):
        axes[1, index].imshow(align_columns(gold, masks_to_plot[index]), cmap="Greys", vmin=0, vmax=1)
        axes[1, index].set_title(f"Joint: {summary['evaluations']['test']['patterns'][index]}")
        axes[1, index].set(xticks=[], yticks=[])
    axes[1, 4].imshow(gold, cmap="Greys", vmin=0, vmax=1)
    axes[1, 4].set_title("Analytic"); axes[1, 4].set(xticks=[], yticks=[])
    figure.tight_layout(); figure.savefig(path, dpi=170); plt.close(figure)
