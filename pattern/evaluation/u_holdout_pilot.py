"""Train one shared binary U on 12 patterns, then adapt only task heads on 4 held-out patterns.

The held-out labels never update U. Validation BCE selects checkpoints within
each stage. Toeplitz scores and test labels are evaluated only after both stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pattern.evaluation.u_reparameterization_pilot import (
    SharedUModel, evaluate, make_data,
)
from pattern.evaluation.u_toeplitz_audit import audit


def split(fold: int) -> tuple[list[int], list[int]]:
    order = torch.randperm(16, generator=torch.Generator().manual_seed(2026))
    held_out = order.reshape(4, 4)[fold].sort().values.tolist()
    train = [index for index in range(16) if index not in held_out]
    return train, held_out


def weight_toeplitz_scores(weights: np.ndarray, permutation: list[int]) -> list[float]:
    aligned = weights[:, :, permutation]
    aligned = aligned / np.maximum(np.linalg.norm(aligned, axis=1, keepdims=True), 1e-12)
    offsets = np.arange(11)[:, None] - np.arange(8)[None, :]
    projected = np.zeros_like(aligned)
    for tap in range(4):
        positions = offsets == tap
        projected[:, positions] = aligned[:, positions].mean(axis=1)[:, None]
    return (np.square(projected).sum(axis=(1, 2)) /
            np.maximum(np.square(aligned).sum(axis=(1, 2)), 1e-12)).tolist()


def fit_stage(model: SharedUModel, optimizer: torch.optim.Optimizer,
              pool_x: torch.Tensor, pool_y: torch.Tensor,
              val_x: torch.Tensor, val_y: torch.Tensor,
              tasks: list[int], steps: int, eval_every: int,
              batch_size: int, eval_batch_size: int,
              generator: torch.Generator) -> tuple[dict, int, float]:
    device = pool_x.device
    selected = torch.tensor(tasks, device=device)
    best_val = float("inf")
    best_state = None
    best_step = 0
    for step in range(1, steps + 1):
        indices = torch.randint(pool_x.size(1), (len(tasks), batch_size),
                                generator=generator, device=device)
        x = pool_x[selected[:, None], indices]
        y = pool_y[selected[:, None], indices]
        # The shared model stores 16 task-specific heads. The selected heads
        # alone enter this loss, so held-out labels cannot update U in stage 1.
        weights = model.first_layer()[selected]
        hidden = F.relu(torch.einsum("tbi,tih->tbh", x, weights) +
                        model.b1[selected, None, :])
        logits = (torch.einsum("tbh,th->tb", hidden, model.w2[selected]) +
                  model.b2[selected, None])
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                val_bce, _ = evaluate(model, val_x, val_y, eval_batch_size)
                score = float(val_bce[selected].mean())
            if score < best_val:
                best_val = score
                best_step = step
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return best_state, best_step, best_val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("learned_binary", "learned_binary_40",
                                              "random_binary", "random_binary_40",
                                              "analytic_binary"), required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--adapt-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-pool-size", type=int, default=32768)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_tasks, held_out_tasks = split(args.fold)
    model = SharedUModel(args.method, args.seed, 11).to(device)
    pool_x, pool_y = make_data(args.train_pool_size, 100000, device, 11)
    val_x, val_y = make_data(args.eval_samples, 200000, device, 11)
    test_x, test_y = make_data(args.eval_samples, 300000, device, 11)

    stage1_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _, train_step, train_val = fit_stage(
        model, stage1_optimizer, pool_x, pool_y, val_x, val_y,
        train_tasks, args.steps, args.eval_every, args.batch_size,
        args.eval_batch_size,
        torch.Generator(device=device).manual_seed(args.seed * 1000 + args.fold))
    frozen_u = model.assignment_matrix().detach().cpu().clone()
    frozen_u_parameters = {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
        if name not in ("v", "b1", "w2", "b2")}

    # The four new tasks can learn their coefficients, biases, and output heads.
    # U is omitted from the optimizer and verified unchanged after adaptation.
    stage2_optimizer = torch.optim.Adam([model.v, model.b1, model.w2, model.b2],
                                        lr=args.lr)
    best_state, adapt_step, adapt_val = fit_stage(
        model, stage2_optimizer, pool_x, pool_y, val_x, val_y,
        held_out_tasks, args.adapt_steps, args.eval_every, args.batch_size,
        args.eval_batch_size,
        torch.Generator(device=device).manual_seed(args.seed * 1000 + args.fold + 100000))
    assert torch.equal(frozen_u, model.assignment_matrix().detach().cpu())
    assert all(torch.equal(value, dict(model.named_parameters())[name].detach().cpu())
               for name, value in frozen_u_parameters.items())

    with torch.no_grad():
        test_bce, test_acc = evaluate(model, test_x, test_y, args.eval_batch_size)
    summary = {
        "method": args.method,
        "seq_len": 11,
        "seed": args.seed,
        "fold": args.fold,
        "train_tasks": train_tasks,
        "held_out_tasks": held_out_tasks,
        "steps": args.steps,
        "adapt_steps": args.adapt_steps,
        "best_train_step": train_step,
        "best_adapt_step": adapt_step,
        "best_train_val_bce": train_val,
        "best_held_out_val_bce": adapt_val,
        "train_test_bce": float(test_bce[train_tasks].mean()),
        "held_out_test_bce": float(test_bce[held_out_tasks].mean()),
        "train_test_acc": float(test_acc[train_tasks].mean()),
        "held_out_test_acc": float(test_acc[held_out_tasks].mean()),
        "mean_test_acc": float(test_acc.mean()),
        "u_frozen_during_adaptation": True,
        "global_active_edges": int((frozen_u.argmax(1) != 0).sum()),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"state_dict": best_state,
                "protocol": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
                "train_tasks": train_tasks, "held_out_tasks": held_out_tasks},
               args.out_dir / "best.pt")
    scores = audit(args.out_dir)
    weights = model.first_layer().detach().cpu().numpy()
    per_task_energy = weight_toeplitz_scores(
        weights, scores["hidden_permutation_new_to_old"])
    scores.update({
        "train_weight_toeplitz_energy": float(np.mean(
            [per_task_energy[index] for index in train_tasks])),
        "held_out_weight_toeplitz_energy": float(np.mean(
            [per_task_energy[index] for index in held_out_tasks])),
        "weight_toeplitz_energy_by_task": per_task_energy,
    })
    (args.out_dir / "toeplitz_audit.json").write_text(json.dumps(scores, indent=2) + "\n")
    print(args.method, "fold", args.fold, "seed", args.seed,
          "heldout BCE", round(summary["held_out_test_bce"], 3),
          "heldout W Toeplitz", round(scores["held_out_weight_toeplitz_energy"], 3),
          "U support IoU", round(scores["toeplitz_support_iou"], 3), flush=True)


if __name__ == "__main__":
    main()
