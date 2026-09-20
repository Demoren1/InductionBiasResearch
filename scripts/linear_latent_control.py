"""Exact-inner-solve control for the necessity of a variable structure code.

Tasks are isotropic Gaussian linear regression problems with population weights
w = U_c v. A single rank-k subspace cannot contain both deliberately disjoint
condition subspaces. The same coordinate generator is trained with either one
shared code or one code per condition; task-specific v is solved exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


DIM = 16
RANK = 4
LATENT_DIM = 4
CONDITIONS = 2


class CoordinateBasis(nn.Module):
    def __init__(self, variable_code: bool) -> None:
        super().__init__()
        self.variable_code = variable_code
        self.network = nn.Sequential(
            nn.Linear(LATENT_DIM + 1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, RANK),
        )
        initial = torch.zeros(CONDITIONS if variable_code else 1, LATENT_DIM)
        if variable_code:
            initial[0, 0], initial[1, 0] = -1, 1
        self.codes = nn.Parameter(initial)
        self.register_buffer("positions", torch.linspace(-1, 1, DIM)[:, None])

    def forward(self, condition: int) -> torch.Tensor:
        code = self.codes[condition if self.variable_code else 0]
        z = code[None].expand(DIM, -1)
        return self.network(torch.cat((z, self.positions), dim=-1))


def true_basis(varying_truth: bool, device: torch.device) -> torch.Tensor:
    truth = torch.zeros(CONDITIONS, DIM, RANK, device=device)
    truth[0, :RANK, :] = torch.eye(RANK, device=device)
    truth[1, (RANK if varying_truth else 0):(2 * RANK if varying_truth else RANK), :] = (
        torch.eye(RANK, device=device)
    )
    return truth


def task_weights(truth: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    coeffs = torch.randn(CONDITIONS, RANK, count, generator=rng).to(truth.device)
    return truth @ coeffs


def exact_task_loss(candidate: torch.Tensor, targets: torch.Tensor, ridge: float = 1e-5) -> torch.Tensor:
    gram = candidate.T @ candidate + ridge * torch.eye(RANK, device=candidate.device)
    coefficients = torch.linalg.solve(gram, candidate.T @ targets)
    residual = candidate @ coefficients - targets
    return residual.square().sum(dim=0).mean()


def projector_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    q1, _ = torch.linalg.qr(first)
    q2, _ = torch.linalg.qr(second)
    return float(((2 * RANK - 2 * (q1.T @ q2).square().sum()).clamp_min(0) / (2 * RANK)).sqrt())


def one_run(seed: int, varying_truth: bool, variable_code: bool, *, device: torch.device,
            steps: int) -> dict:
    torch.manual_seed(seed)
    truth = true_basis(varying_truth, device)
    train = task_weights(truth, 32, seed + 1000)
    validation = task_weights(truth, 64, seed + 2000)
    test = task_weights(truth, 1024, seed + 3000)
    model = CoordinateBasis(variable_code).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best = float("inf")
    best_state = None
    best_step = 0
    history = []
    for step in range(steps + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([exact_task_loss(model(c), train[c]) for c in range(CONDITIONS)]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                val = float(torch.stack([
                    exact_task_loss(model(c), validation[c]) for c in range(CONDITIONS)
                ]).mean())
            history.append({"step": step, "validation_excess_mse": val})
            if val < best:
                best, best_step = val, step
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    with torch.no_grad():
        per_condition = [float(exact_task_loss(model(c), test[c])) for c in range(CONDITIONS)]
        learned_distance = projector_distance(model(0), model(1))
        true_distance = projector_distance(truth[0], truth[1])
        code_distance = float((model.codes[0] - model.codes[-1]).norm())
        # Best direct shared-U rank-k benchmark on these held-out targets.
        pooled = torch.cat((test[0], test[1]), dim=1)
        singular = torch.linalg.svdvals(pooled)
        direct_shared_optimum = float(singular[RANK:].square().sum() / pooled.shape[1])
    return {"seed": seed, "varying_truth": varying_truth, "variable_code": variable_code,
            "selected_step": best_step, "selected_validation_excess_mse": best,
            "test_excess_mse": sum(per_condition) / CONDITIONS,
            "test_excess_mse_by_condition": per_condition,
            "true_projector_distance": true_distance,
            "learned_projector_distance": learned_distance,
            "learned_code_distance": code_distance,
            "direct_shared_test_optimum": direct_shared_optimum,
            "direct_per_condition_test_optimum": 0.0,
            "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    rows = []
    for varying_truth in (False, True):
        for variable_code in (False, True):
            for seed in args.seeds:
                result = one_run(seed, varying_truth, variable_code, device=device, steps=args.steps)
                rows.append(result)
                print(f"LINEAR varying_truth={varying_truth} variable_code={variable_code} "
                      f"seed={seed} test={result['test_excess_mse']:.6f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"protocol": {"input_covariance": "identity",
                                               "train_task_heads_per_condition": 32,
                                               "validation_task_heads_per_condition": 64,
                                               "test_task_heads_per_condition": 1024,
                                               "true_union_rank_by_case": {"same_truth": 4,
                                                                           "varying_truth": 8},
                                               "candidate_rank": RANK,
                                               "checkpoint_selection": "validation task population risk",
                                               "inner_solve": "ridge 1e-5 closed form"},
                                    "device": str(device), "steps": args.steps, "rows": rows},
                                   indent=2) + "\n")


if __name__ == "__main__":
    main()
