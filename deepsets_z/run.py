"""Generated parameter sharing on the Deep Sets text-digit-sum task.

This is a controlled, four-bit version of the text experiment in Zaheer et al.
The set is supplied only with its sum label. G_psi(z, slot, bit) generates a
per-slot permutation U of four shared weights, and W = U v. The inner problem
fits v by ridge regression on support sets; the outer problem trains G_psi on
separately sampled query sets. Validation chooses a checkpoint and one candidate U.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("learned_z", "fixed_z", "no_z_one", "no_z_16", "oracle")
TEST_LENGTHS = (5, 10, 20, 50, 100)
PERMUTATIONS = F.one_hot(
    torch.tensor(list(itertools.permutations(range(4)))), 4,
).float()


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = tuple(range(42, 50))
    steps: int = 500
    candidates: int = 16
    train_sets: int = 12000
    val_sets: int = 2000
    test_sets_per_length: int = 2000
    support_batch: int = 256
    query_batch: int = 256
    validate_every: int = 25
    generator_lr: float = 0.01
    latent_lr: float = 0.03
    ridge: float = 1e-6
    temperature: float = 1.0
    train_max_length: int = 10
    max_length: int = 100
    width: int = 16
    latent_dim: int = 4
    placement: str = "prefix"
    head_init_std: float = 0.01
    position_scale: float = 1.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sets(n: int, max_length: int, rng: torch.Generator,
              device: torch.device, fixed_length: int | None = None,
              slot_capacity: int | None = None,
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent digit sets; train and validation contain 1..10 elements."""
    capacity = slot_capacity or max_length
    lengths = (torch.full((n,), fixed_length, dtype=torch.long) if fixed_length
               else torch.randint(1, max_length + 1, (n,), generator=rng))
    digits = torch.randint(0, 10, (n, capacity), generator=rng)
    bits = ((digits[..., None] >> torch.arange(4)) & 1).float()
    if slot_capacity is None:
        mask = torch.arange(capacity)[None, :] < lengths[:, None]
    else:
        # Every coordinate can participate in training despite small set size.
        scores = torch.rand(n, capacity, generator=rng)
        ranks = scores.argsort(dim=1).argsort(dim=1)
        mask = ranks < lengths[:, None]
    bits *= mask[..., None]
    target = (digits * mask).sum(-1).float()
    return bits.to(device), target.to(device), lengths.to(device)


def coordinate_grid(slots: int, train_slots: int, device: torch.device,
                    position_scale: float = 1.0) -> torch.Tensor:
    # Training positions span [-1, 1]; unseen positions require extrapolation.
    position = (torch.arange(slots, device=device).float()
                / (train_slots - 1) * 2 - 1) * position_scale
    bit = torch.arange(4, device=device).float() / 3 * 2 - 1
    return torch.stack(torch.broadcast_tensors(position[:, None], bit[None, :]), -1)


class LatentGenerator(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(cfg.latent_dim + 2, cfg.width), nn.Tanh(),
            nn.Linear(cfg.width, cfg.width), nn.Tanh(),
            nn.Linear(cfg.width, 4),
        )
        nn.init.normal_(self.network[-1].weight, std=cfg.head_init_std)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, grid: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        candidates, latent_dim = z.shape
        slots = grid.shape[0]
        coordinates = grid[None].expand(candidates, -1, -1, -1)
        latent = z[:, None, None].expand(candidates, slots, 4, latent_dim)
        return self.network(torch.cat((latent, coordinates), -1))


class IndependentGenerators(nn.Module):
    """Vectorized independent generators; their first-layer z is folded away."""
    def __init__(self, cfg: Config, seeds: list[int], initial_z: torch.Tensor) -> None:
        super().__init__()
        parameters: dict[str, list[torch.Tensor]] = {
            name: [] for name in ("w1", "b1", "w2", "b2", "w3", "b3")
        }
        for seed, z in zip(seeds, initial_z, strict=True):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                net = LatentGenerator(cfg).network
            first, second, last = net[0], net[2], net[4]
            values = {
                "w1": first.weight[:, cfg.latent_dim:].detach(),
                "b1": (first.bias + first.weight[:, :cfg.latent_dim] @ z.cpu()).detach(),
                "w2": second.weight.detach(), "b2": second.bias.detach(),
                "w3": last.weight.detach(), "b3": last.bias.detach(),
            }
            for name in parameters:
                parameters[name].append(values[name])
        for name, values in parameters.items():
            setattr(self, name, nn.Parameter(torch.stack(values)))

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(torch.einsum("pdi,chi->cpdh", grid, self.w1)
                            + self.b1[:, None, None])
        hidden = torch.tanh(torch.einsum("cpdi,chi->cpdh", hidden, self.w2)
                            + self.b2[:, None, None])
        return torch.einsum("cpdi,chi->cpdh", hidden, self.w3) + self.b3[:, None, None]


def assignment(logits: torch.Tensor, *, temperature: float, straight_through: bool,
               ) -> torch.Tensor:
    permutations = PERMUTATIONS.to(logits.device)
    scores = torch.einsum("cpdq,kdq->cpk", logits, permutations)
    hard = permutations[scores.argmax(-1)]
    if not straight_through:
        return hard
    soft = (logits / temperature).softmax(-1)
    return hard + soft - soft.detach()


def design(bits: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bpd,cpdq->cbq", bits, u)


def fit_weights(h: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor,
                ridge: float) -> torch.Tensor:
    # Normalize by set size so short and long sets have the same influence.
    x = h / lengths[None, :, None]
    y = target / lengths
    gram = torch.einsum("cbq,cbr->cqr", x, x) / x.shape[1]
    cross = torch.einsum("cbq,b->cq", x, y) / x.shape[1]
    eye = torch.eye(4, device=h.device)[None]
    return torch.linalg.solve(gram + ridge * eye, cross[..., None]).squeeze(-1)


@torch.no_grad()
def assess(u: torch.Tensor, train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
           cases: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
           cfg: Config) -> tuple[list[dict], torch.Tensor]:
    # Fit the task weights afresh for every candidate; never use validation/test labels.
    train_h = design(train[0], u[:, :train[0].shape[1]])
    v = fit_weights(train_h, train[1], train[2], cfg.ridge)
    rows = []
    for name, (bits, target, lengths) in cases.items():
        h = design(bits, u[:, :bits.shape[1]])
        prediction = torch.einsum("cbq,cq->cb", h, v)
        error = prediction - target
        rows.append({
            "split": name,
            "mae": error.abs().mean(-1).cpu().tolist(),
            "rmse": error.square().mean(-1).sqrt().cpu().tolist(),
            "exact_round_accuracy": (prediction.round() == target[None]).float().mean(-1).cpu().tolist(),
            "mean_normalized_squared_error":
                (error / lengths[None]).square().mean(-1).cpu().tolist(),
        })
    return rows, v


def invariance_error(u: torch.Tensor, v: torch.Tensor,
                     bits: torch.Tensor) -> list[float]:
    """Mean absolute change after reversing element order in each set."""
    h = design(bits, u[:, :bits.shape[1]])
    reversed_h = design(bits.flip(1), u[:, :bits.shape[1]])
    prediction = torch.einsum("cbq,cq->cb", h, v)
    reversed_prediction = torch.einsum("cbq,cq->cb", reversed_h, v)
    return (prediction - reversed_prediction).abs().mean(-1).cpu().tolist()


def make_data(cfg: Config, seed: int, device: torch.device):
    rng = torch.Generator().manual_seed(seed + 1_000_000)
    capacity = cfg.max_length if cfg.placement == "random" else None
    train = make_sets(cfg.train_sets, cfg.train_max_length, rng, device,
                      slot_capacity=capacity)
    val = make_sets(cfg.val_sets, cfg.train_max_length, rng, device,
                    slot_capacity=capacity)
    val_20 = make_sets(cfg.val_sets, 20, rng, device, fixed_length=20,
                       slot_capacity=capacity)
    tests = {f"test_{length}": make_sets(cfg.test_sets_per_length, length, rng,
                                         device, fixed_length=length,
                                         slot_capacity=capacity)
             for length in TEST_LENGTHS}
    return train, val, val_20, tests


def run_arm(cfg: Config, seed: int, arm: str, device: torch.device,
            train: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            val: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            val_20: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            tests: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            ) -> dict:
    candidates = 1 if arm == "no_z_one" or arm == "oracle" else cfg.candidates
    if arm == "oracle":
        u = torch.eye(4, device=device)[None, None].expand(1, cfg.max_length, -1, -1)
        rows, v = assess(u, train, {"validation": val, "validation_20": val_20,
                                   **tests}, cfg)
        return {"arm": arm, "seed": seed, "candidates": 1, "training_seconds": 0.0,
                "selected_candidate": 0, "selected_step": 0,
                "shared_weights": v[0].tolist(),
                "metrics": {row["split"]: {key: value[0] for key, value in row.items()
                                           if key != "split"} for row in rows},
                "permutation_error_10": invariance_error(u, v, tests["test_10"][0])[0],
                "slot_assignments": u[0].argmax(-1).cpu().tolist(),
                "history": []}
    set_seed(seed + 1_000)
    initial_z = torch.randn(cfg.candidates, cfg.latent_dim)
    if arm in ("learned_z", "fixed_z"):
        set_seed(seed + 2_000)
        generator: nn.Module = LatentGenerator(cfg).to(device)
        z = nn.Parameter(initial_z.to(device), requires_grad=arm == "learned_z")
        groups = [{"params": generator.parameters(), "lr": cfg.generator_lr}]
        if arm == "learned_z":
            groups.append({"params": [z], "lr": cfg.latent_lr})

        def logits_for(grid: torch.Tensor) -> torch.Tensor:
            return generator(grid, z)
    else:
        init_seeds = [seed + 2_000 + 10_000 * index for index in range(candidates)]
        generator = IndependentGenerators(cfg, init_seeds, initial_z[:candidates]).to(device)
        groups = [{"params": generator.parameters(), "lr": cfg.generator_lr}]

        def logits_for(grid: torch.Tensor) -> torch.Tensor:
            return generator(grid)

    optimizer = torch.optim.Adam(groups)
    training_slots = cfg.max_length if cfg.placement == "random" else cfg.train_max_length
    grid = coordinate_grid(cfg.max_length, training_slots, device, cfg.position_scale)
    rng = torch.Generator().manual_seed(seed + 3_000_000)
    all_indices = torch.randint(0, cfg.train_sets,
                                (cfg.steps, cfg.support_batch + cfg.query_batch),
                                generator=rng, device="cpu").to(device)
    selection = {
        "in_distribution": {"score": math.inf, "u": None, "step": 0, "candidate": 0},
        "length_20": {"score": math.inf, "u": None, "step": 0, "candidate": 0},
    }
    history: list[dict] = []
    started = time.monotonic()
    for step in range(1, cfg.steps + 1):
        indices = all_indices[step - 1]
        support_idx = indices[:cfg.support_batch]
        query_idx = indices[cfg.support_batch:]
        logits = logits_for(grid[:training_slots])
        u = assignment(logits, temperature=cfg.temperature, straight_through=True)
        support_h = design(train[0][support_idx], u)
        v = fit_weights(support_h, train[1][support_idx], train[2][support_idx], cfg.ridge)
        query_h = design(train[0][query_idx], u)
        prediction = torch.einsum("cbq,cq->cb", query_h, v)
        error = (prediction - train[1][query_idx][None]) / train[2][query_idx][None]
        loss = error.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % cfg.validate_every == 0 or step == cfg.steps:
            with torch.no_grad():
                complete_u = assignment(logits_for(grid), temperature=cfg.temperature,
                                        straight_through=False)
                rows, _ = assess(complete_u, train,
                                 {"validation": val, "validation_20": val_20}, cfg)
                scores_by_policy = {
                    "in_distribution": rows[0]["mean_normalized_squared_error"],
                    "length_20": rows[1]["mean_normalized_squared_error"],
                }
                for policy, scores in scores_by_policy.items():
                    candidate = int(np.argmin(scores))
                    score = scores[candidate]
                    if score < selection[policy]["score"]:
                        selection[policy] = {"score": score, "step": step,
                                             "candidate": candidate,
                                             "u": complete_u[candidate:candidate + 1].clone()}
                history.append({"step": step, "train_loss": float(loss.detach()),
                                "best_validation_normalized_mse":
                                    min(scores_by_policy["in_distribution"]),
                                "best_validation_20_normalized_mse":
                                    min(scores_by_policy["length_20"]),
                                "elapsed_seconds": time.monotonic() - started})
    training_seconds = time.monotonic() - started
    selected_results = {}
    for policy, chosen in selection.items():
        best_u = chosen["u"]
        assert isinstance(best_u, torch.Tensor)
        rows, v = assess(best_u, train, {"validation": val,
                                         "validation_20": val_20, **tests}, cfg)
        selected_results[policy] = {
            "selected_candidate": chosen["candidate"],
            "selected_step": chosen["step"],
            "selected_validation_normalized_mse": chosen["score"],
            "shared_weights": v[0].tolist(),
            "metrics": {row["split"]: {key: value[0] for key, value in row.items()
                                         if key != "split"} for row in rows},
            "permutation_error_10": invariance_error(best_u, v,
                                                      tests["test_10"][0])[0],
            "slot_assignments": best_u[0].argmax(-1).cpu().tolist(),
        }
    return {"arm": arm, "seed": seed, "candidates": candidates,
            "training_seconds": training_seconds,
            "selections": selected_results,
            "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("deepsets_z/results"))
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 50)))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--train-sets", type=int, default=12000)
    parser.add_argument("--val-sets", type=int, default=2000)
    parser.add_argument("--test-sets-per-length", type=int, default=2000)
    parser.add_argument("--placement", choices=("prefix", "random"), default="prefix")
    parser.add_argument("--head-init-std", type=float, default=0.01)
    parser.add_argument("--position-scale", type=float, default=1.0)
    args = parser.parse_args()
    cfg = Config(seeds=tuple(args.seeds), steps=args.steps, train_sets=args.train_sets,
                 val_sets=args.val_sets, test_sets_per_length=args.test_sets_per_length,
                 placement=args.placement, head_init_std=args.head_init_std,
                 position_scale=args.position_scale)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    for seed in cfg.seeds:
        train, val, val_20, tests = make_data(cfg, seed, device)
        for arm in args.arms:
            path = args.output / f"{arm}_seed{seed}.json"
            if path.exists():
                print(f"SKIP {path}", flush=True)
                continue
            result = run_arm(cfg, seed, arm, device, train, val, val_20, tests)
            path.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
            chosen = (result if arm == "oracle"
                      else result["selections"]["in_distribution"])
            print(f"DONE seed={seed} arm={arm} step={chosen['selected_step']} "
                  f"val_mae={chosen['metrics']['validation']['mae']:.4f} "
                  f"test100_mae={chosen['metrics']['test_100']['mae']:.4f} "
                  f"seconds={result['training_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
