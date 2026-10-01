"""First-order bilevel pilot for binary parameter-sharing U on pattern tasks.

Task coefficients and heads train on T while the shared one-hot assignment U
is updated on V. After selecting U by validation loss, U is frozen and task
coefficients/heads are retrained on T+V. This alternates inner and outer steps
as a practical first-order approximation to Yeh et al.'s bilevel objective;
it does not differentiate through all inner optimization steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from pattern import config
from pattern.evaluation.u_reparameterization_pilot import SharedUModel, TASKS, make_data, evaluate


def batch_from_pool(x: torch.Tensor, y: torch.Tensor, size: int,
                    generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(x.size(1), (TASKS, size), device=x.device,
                            generator=generator)
    tasks = torch.arange(TASKS, device=x.device)[:, None]
    return x[tasks, indices], y[tasks, indices]


def fit(args: argparse.Namespace) -> dict:
    torch.set_num_threads(2)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = SharedUModel("learned_binary", args.seed).to(device)
    train_x, train_y = make_data(args.train_pool_size, 100000, device)
    val_x, val_y = make_data(args.eval_samples, 200000, device)
    test_x, test_y = make_data(args.eval_samples, 300000, device)
    generator = torch.Generator(device=device).manual_seed(args.seed * 1000)
    task_params = [model.v, model.b1, model.w2, model.b2]
    inner_optimizer = torch.optim.Adam(task_params, lr=args.inner_lr)
    outer_optimizer = torch.optim.Adam([model.assignment_logits], lr=args.outer_lr)
    best_val = float("inf")
    best_outer_step = 0
    best_state = None
    for outer_step in range(1, args.outer_steps + 1):
        model.assignment_logits.requires_grad_(False)
        for parameter in task_params:
            parameter.requires_grad_(True)
        for _ in range(args.inner_steps):
            x, y = batch_from_pool(train_x, train_y, args.batch_size, generator)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            inner_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            inner_optimizer.step()
        model.assignment_logits.requires_grad_(True)
        for parameter in task_params:
            parameter.requires_grad_(False)
        x, y = batch_from_pool(val_x, val_y, args.batch_size, generator)
        prediction_loss = F.binary_cross_entropy_with_logits(model(x), y)
        probabilities = torch.softmax(model.assignment_logits, dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1).mean()
        outer_loss = prediction_loss + args.entropy_weight * entropy
        outer_optimizer.zero_grad(set_to_none=True)
        outer_loss.backward()
        outer_optimizer.step()
        if outer_step % args.eval_every == 0 or outer_step == args.outer_steps:
            model.eval()
            with torch.no_grad():
                val_bce, _ = evaluate(model, val_x, val_y, args.eval_batch_size)
            mean_val = float(val_bce.mean())
            if mean_val < best_val:
                best_val = mean_val
                best_outer_step = outer_step
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            model.train()
    assert best_state is not None
    model.load_state_dict(best_state)
    model.assignment_logits.requires_grad_(False)
    with torch.no_grad():
        reset = torch.Generator(device=device).manual_seed(args.seed + 10000)
        model.v.copy_(torch.randn(model.v.shape, generator=reset, device=device) * 0.1)
        model.w2.copy_(torch.randn(model.w2.shape, generator=reset, device=device) * 0.1)
        model.b1.zero_()
        model.b2.zero_()
    for parameter in task_params:
        parameter.requires_grad_(True)
    final_optimizer = torch.optim.Adam(task_params, lr=args.inner_lr)
    full_x = torch.cat((train_x, val_x), dim=1)
    full_y = torch.cat((train_y, val_y), dim=1)
    for _ in range(args.final_steps):
        x, y = batch_from_pool(full_x, full_y, args.batch_size, generator)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        final_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        final_optimizer.step()
    model.eval()
    with torch.no_grad():
        test_bce, test_acc = evaluate(model, test_x, test_y, args.eval_batch_size)
        assignment = model.assignment_matrix().argmax(1).reshape(8, 8).cpu()
        W = model.first_layer().cpu()
    output = {
        "method": "learned_binary_first_order_bilevel",
        "seed": args.seed,
        "outer_steps": args.outer_steps,
        "inner_steps_per_outer": args.inner_steps,
        "final_steps": args.final_steps,
        "best_outer_step": best_outer_step,
        "best_discovery_val_bce": best_val,
        "mean_test_acc": float(test_acc.mean()),
        "mean_test_bce": float(test_bce.mean()),
        "test_acc_by_pattern": dict(zip(config.PATTERNS, test_acc.cpu().tolist())),
        "active_weight_entries": int((assignment != 0).sum()),
        "assignment_codes": assignment.tolist(),
        "used_codes": assignment.unique().tolist(),
        "protocol": {
            "split": "task coefficients on T; U on V; fixed U then task coefficients on T+V",
            "outer_gradient": "first-order: does not differentiate through inner optimizer",
            "binary_U": "one-hot rows, code 0 has fixed coefficient zero, exactly 32 nonzero rows",
            "VAE_trained": False,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    torch.save({"state_dict": {name: value.detach().cpu()
                                for name, value in model.state_dict().items()},
                "first_layer": W, "assignment": assignment,
                "patterns": list(config.PATTERNS)}, args.out_dir / "best.pt")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-steps", type=int, default=250)
    parser.add_argument("--inner-steps", type=int, default=20)
    parser.add_argument("--final-steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-pool-size", type=int, default=32768)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--outer-lr", type=float, default=1e-2)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = fit(args)
    print("[u-bilevel] acc", result["mean_test_acc"],
          "best_outer_step", result["best_outer_step"], flush=True)


if __name__ == "__main__":
    main()
