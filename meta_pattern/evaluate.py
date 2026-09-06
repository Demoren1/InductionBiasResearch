"""Freeze U, fit fresh task v, evaluate held-out patterns on untouched inputs."""
import argparse
from pathlib import Path

import torch

from .common import dataset, make_model, metrics, seed_for, setup, source_hashes, task_splits, write_json
from .config import Config
from .models import adapt_v, forward_with_u


def evaluate(checkpoint, out, device="cuda", steps=(20, 100, 500), repeats=3,
             tasks_per_length=0, test_size=2048, support_size=None):
    if not steps or any(s < 0 for s in steps) or repeats < 1 or tasks_per_length < 0 or test_size < 1:
        raise ValueError("Invalid evaluation budget")
    if support_size is not None and support_size < 1:
        raise ValueError("support_size must be positive")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {out}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    c = Config(**state["config"])
    support_size = c.support_size if support_size is None else support_size
    setup(c.seed, device)
    model = make_model(c, device)
    model.load_state_dict(state["model"])
    model.eval()
    rows, skipped = [], []
    for k in c.lengths:
        if c.method == "table" and k not in c.train_lengths:
            skipped.append({"length": k, "reason": "No table entry for a held-out length"})
            continue
        tasks = [t for t in task_splits(c)["test"] if t.length == k]
        if tasks_per_length:
            tasks = tasks[:tasks_per_length]
        u = tuple(t.detach() for t in model(k))
        singular = [torch.linalg.svdvals(t).cpu().tolist() for t in u]
        for task in tasks:
            for repeat in range(repeats):
                base = seed_for("final_evaluation", c.seed, task.pattern, repeat)
                support = dataset(c, task, support_size, seed_for(base, "support"), "support", device)
                tests = {kind: dataset(c, task, test_size, seed_for(base, kind), "test", device,
                                      balanced=(kind == "balanced")) for kind in ("balanced", "natural")}
                for budget in steps:
                    v = adapt_v(u, support["x"], support["y"], steps=budget, lr=c.inner_lr,
                                seed=seed_for(base, "v"), create_graph=False, batch_size=c.batch_size,
                                optimizer=c.inner_optimizer, init_scale=c.init_scale)
                    for kind, data in tests.items():
                        with torch.no_grad():
                            score = metrics(forward_with_u(data["x"], u, v, seq_len=c.seq_len, hidden=c.hidden), data["y"])
                        rows.append({"pattern": task.pattern, "length": k, "repeat": repeat, "steps": budget,
                                     "distribution": kind, "length_seen": k in c.train_lengths,
                                     "regime": ("seen_length" if k in c.train_lengths else
                                                "interpolation" if min(c.train_lengths) < k < max(c.train_lengths)
                                                else "extrapolation"),
                                     "u_singular_values": singular, **score})
            print(f"EVAL method={c.method} seed={c.seed} length={k} pattern={task.pattern}", flush=True)
    aggregates = []
    for budget in steps:
        for kind in ("balanced", "natural"):
            for k in sorted({r["length"] for r in rows}):
                group = [r for r in rows if r["steps"] == budget and r["distribution"] == kind and r["length"] == k]
                aggregates.append({"length": k, "steps": budget, "distribution": kind, "n": len(group),
                                   "regime": group[0]["regime"],
                                   **{key: sum(r[key] for r in group) / len(group) for key in ("bce", "accuracy")}})
    result = {"checkpoint": str(Path(checkpoint).resolve()), "checkpoint_step": state["step"],
              "config": c.to_dict(), "steps": list(steps), "repeats": repeats,
              "tasks_per_length": tasks_per_length, "test_size": test_size,
              "support_size": support_size, "skipped": skipped,
              "training_source_sha256": state["source_sha256"], "evaluation_source_sha256": source_hashes(),
              "aggregates": aggregates, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", nargs="+", type=int, default=[20, 100, 500])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tasks-per-length", type=int, default=0, help="0 evaluates every held-out pattern")
    p.add_argument("--test-size", type=int, default=2048)
    p.add_argument("--support-size", type=int)
    evaluate(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
