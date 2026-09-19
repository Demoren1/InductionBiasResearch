"""Position-free generated sharing for the four-bit digit-sum experiment.

G_psi(z, bit_id) emits one 4x4 permutation U, shared by every set element.
The digit bits enter only the predictor: sum_i x_i U v.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .run_position_dependent import (
    ARMS, TEST_LENGTHS, PERMUTATIONS, fit_weights, make_sets, set_seed,
)


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
    width: int = 16
    latent_dim: int = 4
    head_init_std: float = 0.01


def make_data(cfg: Config, seed: int, device: torch.device):
    rng = torch.Generator().manual_seed(seed + 1_000_000)
    train = make_sets(cfg.train_sets, cfg.train_max_length, rng, device)
    val = make_sets(cfg.val_sets, cfg.train_max_length, rng, device)
    val_20 = make_sets(cfg.val_sets, 20, rng, device, fixed_length=20)
    tests = {f"test_{length}": make_sets(cfg.test_sets_per_length, length, rng,
                                         device, fixed_length=length)
             for length in TEST_LENGTHS}
    return train, val, val_20, tests


def bit_coordinates(device: torch.device) -> torch.Tensor:
    return torch.arange(4, device=device).float() / 3 * 2 - 1


class BitGenerator(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(cfg.latent_dim + 1, cfg.width), nn.Tanh(),
            nn.Linear(cfg.width, cfg.width), nn.Tanh(),
            nn.Linear(cfg.width, 4),
        )
        nn.init.normal_(self.network[-1].weight, std=cfg.head_init_std)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        candidates, latent_dim = z.shape
        bit = bit_coordinates(z.device)[None, :, None].expand(candidates, -1, -1)
        latent = z[:, None, :].expand(candidates, 4, latent_dim)
        return self.network(torch.cat((latent, bit), -1))


class IndependentBitGenerators(nn.Module):
    """Independent networks with the initial z contribution folded into bias."""

    def __init__(self, cfg: Config, seeds: list[int], initial_z: torch.Tensor) -> None:
        super().__init__()
        parameters: dict[str, list[torch.Tensor]] = {
            name: [] for name in ("w1", "b1", "w2", "b2", "w3", "b3")
        }
        for seed, z in zip(seeds, initial_z, strict=True):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                net = BitGenerator(cfg).network
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

    def forward(self, device: torch.device) -> torch.Tensor:
        bit = bit_coordinates(device)[:, None]
        hidden = torch.tanh(torch.einsum("di,chi->cdh", bit, self.w1)
                            + self.b1[:, None])
        hidden = torch.tanh(torch.einsum("cdi,chi->cdh", hidden, self.w2)
                            + self.b2[:, None])
        return torch.einsum("cdi,chi->cdh", hidden, self.w3) + self.b3[:, None]


def assignment(logits: torch.Tensor, *, temperature: float,
               straight_through: bool) -> torch.Tensor:
    permutations = PERMUTATIONS.to(logits.device)
    scores = torch.einsum("cdq,kdq->ck", logits, permutations)
    hard = permutations[scores.argmax(-1)]
    if not straight_through:
        return hard
    soft = (logits / temperature).softmax(-1)
    return hard + soft - soft.detach()


def design(bits: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Aggregate all positions first; each U is shared over the whole set."""
    counts = bits.sum(dim=1)
    return torch.einsum("bd,cdq->cbq", counts, u)


@torch.no_grad()
def assess(u: torch.Tensor, train: tuple, cases: dict[str, tuple], cfg: Config):
    v = fit_weights(design(train[0], u), train[1], train[2], cfg.ridge)
    rows = []
    for name, (bits, target, lengths) in cases.items():
        prediction = torch.einsum("cbq,cq->cb", design(bits, u), v)
        error = prediction - target[None]
        rows.append({
            "split": name,
            "mae": error.abs().mean(-1).cpu().tolist(),
            "rmse": error.square().mean(-1).sqrt().cpu().tolist(),
            "exact_round_accuracy": (prediction.round() == target[None]).float().mean(-1).cpu().tolist(),
            "mean_normalized_squared_error":
                (error / lengths[None]).square().mean(-1).cpu().tolist(),
        })
    return rows, v


def run_arm(cfg: Config, seed: int, arm: str, device: torch.device,
            train: tuple, val: tuple, val_20: tuple, tests: dict[str, tuple]) -> dict:
    candidates = 1 if arm in ("no_z_one", "oracle") else cfg.candidates
    if arm == "oracle":
        u = torch.eye(4, device=device)[None]
        rows, v = assess(u, train, {"validation": val, "validation_20": val_20,
                                   **tests}, cfg)
        return {"arm": arm, "seed": seed, "candidates": 1,
                "training_seconds": 0.0, "selected_candidate": 0,
                "selected_step": 0, "shared_weights": v[0].tolist(),
                "metrics": {row["split"]: {key: value[0] for key, value in row.items()
                                           if key != "split"} for row in rows},
                "bit_assignments": u[0].argmax(-1).cpu().tolist()}

    set_seed(seed + 1_000)
    initial_z = torch.randn(cfg.candidates, cfg.latent_dim)
    if arm in ("learned_z", "fixed_z"):
        set_seed(seed + 2_000)
        generator: nn.Module = BitGenerator(cfg).to(device)
        z = nn.Parameter(initial_z.to(device), requires_grad=arm == "learned_z")
        groups = [{"params": generator.parameters(), "lr": cfg.generator_lr}]
        if arm == "learned_z":
            groups.append({"params": [z], "lr": cfg.latent_lr})

        def get_logits() -> torch.Tensor:
            return generator(z)
    else:
        init_seeds = [seed + 2_000 + 10_000 * index for index in range(candidates)]
        generator = IndependentBitGenerators(cfg, init_seeds,
                                             initial_z[:candidates]).to(device)
        groups = [{"params": generator.parameters(), "lr": cfg.generator_lr}]

        def get_logits() -> torch.Tensor:
            return generator(device)

    optimizer = torch.optim.Adam(groups)
    rng = torch.Generator().manual_seed(seed + 3_000_000)
    all_indices = torch.randint(0, cfg.train_sets,
                                (cfg.steps, cfg.support_batch + cfg.query_batch),
                                generator=rng).to(device)
    selected = {"score": math.inf, "u": None, "step": 0, "candidate": 0}
    history = []
    started = time.monotonic()
    for step in range(1, cfg.steps + 1):
        indices = all_indices[step - 1]
        support_idx = indices[:cfg.support_batch]
        query_idx = indices[cfg.support_batch:]
        u = assignment(get_logits(), temperature=cfg.temperature,
                       straight_through=True)
        v = fit_weights(design(train[0][support_idx], u),
                        train[1][support_idx], train[2][support_idx], cfg.ridge)
        prediction = torch.einsum("cbq,cq->cb", design(train[0][query_idx], u), v)
        error = (prediction - train[1][query_idx][None]) / train[2][query_idx][None]
        loss = error.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % cfg.validate_every == 0 or step == cfg.steps:
            with torch.no_grad():
                hard_u = assignment(get_logits(), temperature=cfg.temperature,
                                    straight_through=False)
                rows, _ = assess(hard_u, train, {"validation": val}, cfg)
                scores = rows[0]["mean_normalized_squared_error"]
                candidate = int(np.argmin(scores))
                if scores[candidate] < selected["score"]:
                    selected = {"score": scores[candidate], "step": step,
                                "candidate": candidate,
                                "u": hard_u[candidate:candidate + 1].clone()}
                history.append({"step": step, "train_loss": float(loss.detach()),
                                "best_validation_normalized_mse": min(scores)})
    training_seconds = time.monotonic() - started
    best_u = selected["u"]
    assert isinstance(best_u, torch.Tensor)
    rows, v = assess(best_u, train, {"validation": val, "validation_20": val_20,
                                    **tests}, cfg)
    return {"arm": arm, "seed": seed, "candidates": candidates,
            "training_seconds": training_seconds,
            "selected_candidate": selected["candidate"],
            "selected_step": selected["step"],
            "shared_weights": v[0].tolist(),
            "metrics": {row["split"]: {key: value[0] for key, value in row.items()
                                       if key != "split"} for row in rows},
            "bit_assignments": best_u[0].argmax(-1).cpu().tolist(),
            "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("deepsets_z/results_position_free"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 50)))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--train-sets", type=int, default=12000)
    parser.add_argument("--val-sets", type=int, default=2000)
    parser.add_argument("--test-sets-per-length", type=int, default=2000)
    args = parser.parse_args()
    cfg = Config(seeds=tuple(args.seeds), steps=args.steps, train_sets=args.train_sets,
                 val_sets=args.val_sets, test_sets_per_length=args.test_sets_per_length)
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
            print(f"DONE seed={seed} arm={arm} step={result['selected_step']} "
                  f"val_mae={result['metrics']['validation']['mae']:.4f} "
                  f"test100_mae={result['metrics']['test_100']['mae']:.4f} "
                  f"seconds={result['training_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
