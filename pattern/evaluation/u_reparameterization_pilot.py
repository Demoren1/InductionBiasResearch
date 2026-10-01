"""Pattern pilot for W = Uv, including Kronecker and binary assignment U.

The structure U is shared by all 16 length-four pattern tasks. The active
length-11 protocol has W of shape 11x8 and a global budget of 32 active edges;
no per-hidden-unit quota is imposed. Binary direct U is a one-hot assignment
over a fixed zero code and four task-specific shared coefficients. The script
also retains length-8 and diagnostic quota/free-cardinality modes. This tests
representations and Toeplitz geometry; it does not train a VAE or test new task
families.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from data.generate import make_dataset  # noqa: E402

METHODS = ("direct_continuous", "kronecker_continuous", "kronecker_binary",
           "learned_binary", "random_binary", "analytic_binary",
           "learned_binary_40", "random_binary_40", "outer_assignment",
           "rank3_assignment", "offset_assignment", "free_assignment",
           "rank3_tied", "offset_tied")
TASKS = len(config.PATTERNS)
K = 25
BINARY_K = 5  # fixed zero code plus four shared tap values


def analytic_assignment(seq_len: int = 8) -> torch.Tensor:
    windows = seq_len - config.PATTERN_LEN + 1
    categories = torch.zeros(seq_len, 8, dtype=torch.long)
    for hidden in range(8):
        start = hidden % windows
        for tap in range(4):
            categories[start + tap, hidden] = tap + 1
    assert int((categories != 0).sum()) == 32
    return F.one_hot(categories.flatten(), num_classes=BINARY_K).float()


def random_assignment(seed: int, n_codes: int = BINARY_K,
                      seq_len: int = 8, column_quota: bool = False,
                      free_cardinality: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    n_edges = seq_len * 8
    categories = torch.zeros(n_edges, dtype=torch.long)
    if free_cardinality:
        active = torch.where(torch.rand(n_edges, generator=generator) < 32 / n_edges)[0]
    elif column_quota:
        active = torch.cat([8 * torch.randperm(seq_len, generator=generator)[:4] + h
                            for h in range(8)])
    else:
        active = torch.randperm(n_edges, generator=generator)[:32]
    categories[active] = torch.randint(1, n_codes, (len(active),), generator=generator)
    return F.one_hot(categories, num_classes=n_codes).float()


class SharedUModel(nn.Module):
    def __init__(self, method: str, seed: int, seq_len: int = 8,
                 column_quota: bool = False, free_cardinality: bool = False):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.seq_len = seq_len
        self.n_edges = seq_len * 8
        self.column_quota = column_quota
        self.free_cardinality = free_cardinality
        generator = torch.Generator().manual_seed(seed)
        self.n_codes = (seq_len + 8 if method.endswith("_tied") else
                        41 if method.endswith("_40") else BINARY_K)
        coefficient_count = (seq_len * 5 if method == "kronecker_binary" else
                             self.n_codes if "binary" in method or
                             method.endswith(("_assignment", "_tied")) else K)
        self.v = nn.Parameter(torch.randn(TASKS, coefficient_count,
                                          generator=generator) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(TASKS, 8))
        self.w2 = nn.Parameter(torch.randn(TASKS, 8, generator=generator) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(TASKS))
        if method == "direct_continuous":
            self.u = nn.Parameter(torch.randn(self.n_edges, K, generator=generator) * 0.1)
        elif method == "kronecker_continuous":
            q1 = torch.linalg.qr(torch.randn(seq_len, 5, generator=generator)).Q
            q2 = torch.linalg.qr(torch.randn(8, 5, generator=generator)).Q
            self.u1 = nn.Parameter(q1.clone())
            self.u2 = nn.Parameter(q2.clone())
        elif method == "kronecker_binary":
            identity = torch.eye(seq_len) * 1.0
            self.u1_logits = nn.Parameter(identity +
                                          torch.randn(seq_len, seq_len, generator=generator) * 0.1)
            self.u2_logits = nn.Parameter(torch.randn(8, 5, generator=generator) * 0.1)
            self.gate_logits = nn.Parameter(torch.randn(self.n_edges, generator=generator) * 0.1)
        elif method in ("learned_binary", "learned_binary_40"):
            self.assignment_logits = nn.Parameter(
                torch.randn(self.n_edges, self.n_codes, generator=generator) * 0.1)
        elif method in ("random_binary", "random_binary_40"):
            self.register_buffer("assignment", random_assignment(seed, self.n_codes,
                                                                  seq_len, column_quota,
                                                                  free_cardinality))
        elif method == "analytic_binary":
            self.register_buffer("assignment", analytic_assignment(seq_len))
        elif method.endswith("_assignment"):
            self.code_logits = nn.Parameter(torch.randn(self.n_edges, 4,
                                                        generator=generator) * 0.1)
            if method == "outer_assignment":
                self.gate_left = nn.Parameter(torch.randn(seq_len, generator=generator) * 0.1)
                self.gate_right = nn.Parameter(torch.randn(8, generator=generator) * 0.1)
                self.gate_bias = nn.Parameter(torch.zeros(()))
            elif method == "rank3_assignment":
                self.gate_left = nn.Parameter(torch.randn(seq_len, 3, generator=generator) * 0.1)
                self.gate_right = nn.Parameter(torch.randn(8, 3, generator=generator) * 0.1)
                self.gate_bias = nn.Parameter(torch.zeros(()))
            elif method == "offset_assignment":
                self.offset_logits = nn.Parameter(torch.randn(seq_len + 7,
                                                              generator=generator) * 0.1)
            else:
                self.gate_logits = nn.Parameter(torch.randn(self.n_edges,
                                                            generator=generator) * 0.1)
        elif method.endswith("_tied"):
            if method == "rank3_tied":
                self.gate_left = nn.Parameter(torch.randn(seq_len, 3, generator=generator) * 0.1)
                self.gate_right = nn.Parameter(torch.randn(8, 3, generator=generator) * 0.1)
                self.gate_bias = nn.Parameter(torch.zeros(()))
            else:
                self.offset_logits = nn.Parameter(torch.randn(seq_len + 7,
                                                              generator=generator) * 0.1)

    def gate_scores(self) -> torch.Tensor:
        if self.method in ("outer_assignment", "rank3_assignment", "rank3_tied"):
            if self.method == "outer_assignment":
                scores = self.gate_left[:, None] * self.gate_right[None, :]
            else:
                scores = self.gate_left @ self.gate_right.T
            return (scores + self.gate_bias).flatten()
        if self.method in ("offset_assignment", "offset_tied"):
            input_index = torch.arange(self.seq_len, device=self.offset_logits.device)[:, None]
            hidden_index = torch.arange(8, device=self.offset_logits.device)[None, :]
            return self.offset_logits[input_index - hidden_index + 7].flatten()
        return self.gate_logits

    def hard_active(self, scores: torch.Tensor) -> torch.Tensor:
        if self.free_cardinality:
            return (scores > 0).float()
        if self.column_quota:
            matrix = scores.reshape(self.seq_len, 8)
            active = torch.zeros_like(matrix)
            active.scatter_(0, matrix.topk(4, dim=0).indices, 1.0)
            return active.flatten()
        active = torch.zeros_like(scores)
        active[scores.topk(32).indices] = 1.0
        return active

    def active_probability(self) -> torch.Tensor:
        if self.method in ("learned_binary", "learned_binary_40"):
            return 1 - torch.softmax(self.assignment_logits, dim=1)[:, 0]
        if self.method == "kronecker_binary":
            return torch.sigmoid(self.gate_logits)
        if self.method.endswith(("_assignment", "_tied")):
            return torch.sigmoid(self.gate_scores())
        return torch.zeros(self.n_edges, device=self.v.device)

    def assignment_matrix(self) -> torch.Tensor:
        if self.method.endswith("_tied"):
            scores = self.gate_scores()
            hard_gate = self.hard_active(scores)
            soft_gate = torch.sigmoid(scores)
            gate = hard_gate + soft_gate - soft_gate.detach()
            input_index = torch.arange(self.seq_len, device=scores.device)[:, None]
            hidden_index = torch.arange(8, device=scores.device)[None, :]
            offset = (input_index - hidden_index + 7).flatten()
            code = F.one_hot(offset, self.seq_len + 7).float()
            return torch.cat((1 - gate[:, None], gate[:, None] * code), dim=1)
        if self.method.endswith("_assignment"):
            scores = self.gate_scores()
            hard_gate = self.hard_active(scores)
            soft_gate = torch.sigmoid(scores)
            gate = hard_gate + soft_gate - soft_gate.detach()
            hard_code = F.one_hot(self.code_logits.argmax(1), 4).float()
            soft_code = torch.softmax(self.code_logits, dim=1)
            code = hard_code + soft_code - soft_code.detach()
            return torch.cat((1 - gate[:, None], gate[:, None] * code), dim=1)
        if self.method not in ("learned_binary", "learned_binary_40"):
            return self.assignment
        logits = self.assignment_logits
        active_score = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
        if self.free_cardinality:
            active = torch.where(active_score > math.log(self.n_codes - 1) + 0.04)[0]
        else:
            active = torch.where(self.hard_active(active_score) > 0)[0]
        category = torch.zeros(self.n_edges, dtype=torch.long, device=logits.device)
        category[active] = logits[active, 1:].argmax(1) + 1
        hard = F.one_hot(category, num_classes=self.n_codes).float()
        soft = torch.softmax(logits, dim=1)
        return hard + soft - soft.detach()

    def first_layer(self) -> torch.Tensor:
        if self.method == "kronecker_continuous":
            V = self.v.reshape(TASKS, 5, 5)
            return torch.einsum("ia,tab,jb->tij", self.u1, V, self.u2)
        if self.method == "kronecker_binary":
            def one_hot_st(logits: torch.Tensor) -> torch.Tensor:
                hard = F.one_hot(logits.argmax(1), logits.size(1)).float()
                soft = torch.softmax(logits, dim=1)
                return hard + soft - soft.detach()
            u1 = one_hot_st(self.u1_logits)
            u2 = one_hot_st(self.u2_logits)
            scores = self.gate_logits
            hard_gate = self.hard_active(scores)
            soft_gate = torch.sigmoid(scores)
            gate = (hard_gate + soft_gate - soft_gate.detach()).reshape(self.seq_len, 8)
            V = self.v.reshape(TASKS, self.seq_len, 5)
            return torch.einsum("ia,tab,jb->tij", u1, V, u2) * gate
        if self.method in ("direct_continuous",):
            U = self.u
            v = self.v
        else:
            U = self.assignment_matrix()
            v = torch.cat((torch.zeros_like(self.v[:, :1]), self.v[:, 1:]), dim=1)
        return (U @ v.T).T.reshape(TASKS, self.seq_len, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (task, batch, input); output: (task, batch)."""
        W = self.first_layer()
        h = F.relu(torch.einsum("tbi,tih->tbh", x, W) + self.b1[:, None, :])
        return torch.einsum("tbh,th->tb", h, self.w2) + self.b2[:, None]


def make_data(sample_count: int, base_seed: int, device: torch.device,
              seq_len: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len == 8:
        datasets = [make_dataset(pattern, sample_count, base_seed + index,
                                 pos_fraction=config.POS_FRACTION)
                    for index, pattern in enumerate(config.PATTERNS)]
        x = torch.stack([item["x"] for item in datasets]).to(device)
        y = torch.stack([item["y"] for item in datasets]).to(device)
        return x, y
    xs, ys = [], []
    for index, pattern in enumerate(config.PATTERNS):
        generator = torch.Generator().manual_seed(base_seed + index)
        bits = torch.randint(0, 2, (sample_count, seq_len), generator=generator)
        target = torch.tensor([int(value) for value in pattern])
        matches = lambda b: (b.unfold(1, 4, 1) == target).all(-1).any(-1)
        labels = matches(bits)
        missing = max(0, round(sample_count * config.POS_FRACTION) - int(labels.sum()))
        negative = torch.where(~labels)[0]
        chosen = negative[torch.randperm(len(negative), generator=generator)[:missing]]
        starts = torch.randint(seq_len - 3, (len(chosen),), generator=generator)
        bits[chosen[:, None], starts[:, None] + torch.arange(4)] = target
        xs.append(2.0 * bits.float() - 1.0)
        ys.append(matches(bits).float())
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


@torch.no_grad()
def evaluate(model: SharedUModel, x: torch.Tensor, y: torch.Tensor,
             batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    loss = torch.zeros(TASKS, device=x.device)
    correct = torch.zeros_like(loss)
    for start in range(0, x.size(1), batch_size):
        logits = model(x[:, start:start + batch_size])
        target = y[:, start:start + batch_size]
        loss += F.binary_cross_entropy_with_logits(logits, target,
                                                   reduction="none").sum(1)
        correct += ((logits > 0) == (target > 0.5)).sum(1)
    return loss / x.size(1), correct / x.size(1)


def train(args: argparse.Namespace) -> dict:
    torch.set_num_threads(2)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = SharedUModel(args.method, args.seed, args.seq_len,
                         args.column_quota, args.free_cardinality).to(device)
    if args.importance_init is not None:
        if args.method != "learned_binary" or args.seq_len != 11:
            raise ValueError("importance initialization requires learned_binary, seq_len=11")
        prior = torch.load(args.importance_init, map_location="cpu", weights_only=True)
        importance = prior["consensus"][:, prior["canonical_permutation"]].float()
        importance = (importance - importance.min()) / (importance.max() - importance.min())
        with torch.no_grad():
            model.assignment_logits[:, 0].copy_(-4 * importance.flatten().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pool_x, pool_y = make_data(args.train_pool_size, 100000, device, args.seq_len)
    val_x, val_y = make_data(args.eval_samples, 200000, device, args.seq_len)
    test_x, test_y = make_data(args.eval_samples, 300000, device, args.seq_len)
    generator = torch.Generator(device=device).manual_seed(args.seed * 1000)
    best_val = float("inf")
    best_step = 0
    best_state = None
    stability = {"max_train_bce": 0.0, "min_train_bce": float("inf"),
                 "max_grad_norm": 0.0, "nonfinite_steps": 0,
                 "bce_at_steps": []} if args.trace_stability else None
    for step in range(1, args.steps + 1):
        indices = torch.randint(pool_x.size(1), (TASKS, args.batch_size),
                                generator=generator, device=device)
        task_index = torch.arange(TASKS, device=device)[:, None]
        x, y = pool_x[task_index, indices], pool_y[task_index, indices]
        logits = model(x)
        loss = (F.binary_cross_entropy_with_logits(logits, y) +
                args.sparsity_weight * model.active_probability().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if stability is not None:
            bce_value = float(loss.detach())
            grad_norm = float(sum(parameter.grad.detach().square().sum()
                                  for parameter in model.parameters()
                                  if parameter.grad is not None).sqrt())
            if not math.isfinite(bce_value) or not math.isfinite(grad_norm):
                stability["nonfinite_steps"] += 1
                raise FloatingPointError(f"nonfinite BCE/gradient at step {step}")
            stability["max_train_bce"] = max(stability["max_train_bce"], bce_value)
            stability["min_train_bce"] = min(stability["min_train_bce"], bce_value)
            stability["max_grad_norm"] = max(stability["max_grad_norm"], grad_norm)
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                stability["bce_at_steps"].append([step, bce_value, grad_norm])
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            val_bce, _ = evaluate(model, val_x, val_y, args.eval_batch_size)
            mean_val = float(val_bce.mean() +
                             args.sparsity_weight * model.active_probability().mean())
            if mean_val < best_val:
                best_val = mean_val
                best_step = step
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
            model.train()
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    val_bce, val_acc = evaluate(model, val_x, val_y, args.eval_batch_size)
    test_bce, test_acc = evaluate(model, test_x, test_y, args.eval_batch_size)
    W = model.first_layer().detach().cpu()
    output = {
        "method": args.method,
        "seq_len": args.seq_len,
        "column_quota": args.column_quota,
        "free_cardinality": args.free_cardinality,
        "sparsity_weight": args.sparsity_weight,
        "importance_init": str(args.importance_init) if args.importance_init else None,
        "seed": args.seed,
        "steps": args.steps,
        "best_step": best_step,
        "best_val_objective": best_val,
        "best_mean_val_bce": float(val_bce.mean()),
        "mean_test_acc": float(test_acc.mean()),
        "mean_test_bce": float(test_bce.mean()),
        "mean_val_acc": float(val_acc.mean()),
        "test_acc_by_pattern": dict(zip(config.PATTERNS, test_acc.cpu().tolist())),
        "test_bce_by_pattern": dict(zip(config.PATTERNS, test_bce.cpu().tolist())),
        "active_weight_entries_mean": float((W != 0).float().sum((1, 2)).mean()),
        "first_layer_rank_mean": float(torch.linalg.matrix_rank(W).float().mean()),
        "coefficient_count_per_task": (args.seq_len * 5 if args.method == "kronecker_binary" else
                                       model.n_codes - 1 if "binary" in args.method or
                                       args.method.endswith(("_assignment", "_tied")) else K),
        "shared_u_parameter_count": sum(p.numel() for name, p in model.named_parameters()
                                        if name in ("u", "u1", "u2", "assignment_logits",
                                                    "u1_logits", "u2_logits", "gate_logits",
                                                    "code_logits", "gate_left", "gate_right",
                                                    "offset_logits", "gate_bias")),
    }
    if args.method in ("learned_binary", "random_binary", "analytic_binary",
                       "learned_binary_40", "random_binary_40") or args.method.endswith(("_assignment", "_tied")):
        assignment = model.assignment_matrix().detach().cpu()
        output["binary_assignment_active_rows"] = int((assignment.argmax(1) != 0).sum())
        output["binary_assignment_used_codes"] = int(assignment.argmax(1).unique().numel())
    if args.method == "kronecker_binary":
        input_code = model.u1_logits.argmax(1).detach().cpu()
        hidden_code = model.u2_logits.argmax(1).detach().cpu()
        gate = model.hard_active(model.gate_logits.detach().cpu()).bool()
        code = 1 + input_code[:, None] * 5 + hidden_code[None, :]
        code = torch.where(gate.reshape(args.seq_len, 8), code, 0)
        output["binary_assignment_active_rows"] = int(gate.sum())
        output["binary_assignment_used_codes"] = int(code.unique().numel())
    if stability is not None:
        output["stability_trace"] = stability
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    torch.save({"state_dict": best_state, "first_layer": W,
                "patterns": list(config.PATTERNS),
                "protocol": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()}},
               args.out_dir / "best.pt")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--column-quota", action="store_true")
    parser.add_argument("--free-cardinality", action="store_true")
    parser.add_argument("--sparsity-weight", type=float, default=0.0)
    parser.add_argument("--importance-init", type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-pool-size", type=int, default=32768)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--trace-stability", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.train_pool_size < args.batch_size or args.seq_len < 4 or (
        args.column_quota and args.free_cardinality
    ) or args.sparsity_weight < 0:
        parser.error("invalid training steps or pool size")
    result = train(args)
    print("[u-pilot]", args.method, result["mean_test_acc"],
          "best_step", result["best_step"], flush=True)


if __name__ == "__main__":
    main()
