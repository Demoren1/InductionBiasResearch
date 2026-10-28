"""Fit full source ReLU networks; infer sharing only from their saved weights."""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4096)
    args = parser.parse_args()
    out = ROOT / "runs" / "v1"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / "source_snapshot.py")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.manual_seed(282000)
    d, h, k, tasks, starts = 32, 16, 32, 6, 4
    hidden_i = torch.randperm(d * h, device=device).reshape(d, h) % k
    values = torch.stack(
        [
            torch.linspace(-1, 1, k, device=device)[torch.randperm(k, device=device)]
            for _ in range(tasks)
        ]
    )
    x = torch.randn(8192 + 2048, d, device=device)
    teacher = values[:, hidden_i]
    y = (torch.einsum("bd,tdh->tbh", x, teacher).relu()).sum(-1) / math.sqrt(h)
    np.savez_compressed(
        out / "teacher.npz",
        indexator=hidden_i.cpu().numpy(),
        values=values.cpu().numpy(),
    )
    np.savez_compressed(out / "source_data.npz", x=x.cpu().numpy(), y=y.cpu().numpy())
    torch.manual_seed(282001)
    weights = (torch.randn(tasks * starts, d, h, device=device) * 0.3).requires_grad_()
    optimizer = torch.optim.Adam([weights], lr=0.01)
    labels = y.repeat_interleave(starts, 0)
    scale = labels[:, :8192].var(1, unbiased=False)
    best = weights.detach().clone()
    best_loss = torch.full((tasks * starts,), torch.inf, device=device)
    history = []
    for step in range(args.steps + 1):
        if step % 128 == 0:
            with torch.no_grad():
                error = torch.zeros(tasks * starts, device=device)
                for ids in torch.arange(8192, 10240, device=device).split(256):
                    pred = (torch.einsum("bd,mdh->mbh", x[ids], weights).relu()).sum(
                        -1
                    ) / math.sqrt(h)
                    error += (pred - labels[:, ids]).square().sum(1) / 2048 / scale
                good = error < best_loss
                best_loss[good] = error[good]
                best[good] = weights[good]
                history.append({"step": step, "query_nmse": best_loss.cpu().tolist()})
            if step % 512 == 0:
                print(
                    step,
                    float(best_loss.mean()),
                    float(best_loss.reshape(tasks, starts).min(1).values.mean()),
                    flush=True,
                )
        if step == args.steps:
            break
        ids = torch.randint(8192, (256,), device=device)
        pred = (torch.einsum("bd,mdh->mbh", x[ids], weights).relu()).sum(
            -1
        ) / math.sqrt(h)
        loss = ((pred - labels[:, ids]).square().mean(1) / scale).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    saved = best.cpu().numpy()
    quality = best_loss.cpu().numpy()
    # Generic scalar quantization, with K equal to the declared codebook budget.
    raw = []
    for w in saved:
        raw.append(
            KMeans(n_clusters=k, n_init=5, random_state=282002)
            .fit_predict(w.reshape(-1, 1))
            .reshape(d, h)
        )
    raw = np.array(raw)
    # Label-invariant within-column sharing signatures align hidden units.
    reference = int(quality.argmin())
    ref = raw[reference]
    ref_sign = (ref[:, None, :] == ref[None, :, :]).reshape(d * d, h)
    aligned = []
    column_orders = []
    for matrix in raw:
        signature = (matrix[:, None, :] == matrix[None, :, :]).reshape(d * d, h)
        cost = np.not_equal(signature[:, :, None], ref_sign[:, None, :]).sum(0)
        old, new = linear_sum_assignment(cost)
        order = np.empty(h, dtype=int)
        order[new] = old
        matrix = matrix[:, order]
        votes = np.zeros((k, k), dtype=int)
        np.add.at(votes, (matrix.flatten(), ref.flatten()), 1)
        a, b = linear_sum_assignment(-votes)
        labels_map = np.empty(k, dtype=int)
        labels_map[a] = b
        aligned.append(labels_map[matrix])
        column_orders.append(order)
    aligned = np.array(aligned)
    selected = []
    for task in range(tasks):
        ids = np.arange(task * starts, (task + 1) * starts)
        selected.extend(ids[np.argsort(quality[ids])[:2]])
    selected = np.array(selected)
    votes = np.stack([(aligned[selected] == label).sum(0) for label in range(k)])
    consensus = votes.argmax(0)
    np.savez_compressed(
        out / "bank.npz",
        weights=saved,
        quality=quality,
        raw_indices=raw,
        aligned_indices=aligned,
        column_orders=np.array(column_orders),
        selected=selected,
        consensus=consensus,
        reference=np.array(reference),
    )
    (out / "history.json").write_text(json.dumps(history))
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "inputs": d,
                "hidden": h,
                "codebook": k,
                "source_tasks": tasks,
                "starts": starts,
                "steps": args.steps,
                "support": 8192,
                "query": 2048,
                "fixed_readout": 1 / math.sqrt(h),
                "fixed_bias": 0,
                "extraction_inputs": ["fitted weights", "source query quality"],
                "alignment": "Hungarian on within-column equality patterns, then label support overlap; no observations or teacher read by extraction",
            },
            indent=2,
        )
    )
    # Post-freeze diagnostic: align the recovered consensus to hidden truth.
    true = hidden_i.cpu().numpy()
    signature = (consensus[:, None, :] == consensus[None, :, :]).reshape(d * d, h)
    true_sign = (true[:, None, :] == true[None, :, :]).reshape(d * d, h)
    old, new = linear_sum_assignment(
        np.not_equal(signature[:, :, None], true_sign[:, None, :]).sum(0)
    )
    order = np.empty(h, dtype=int)
    order[new] = old
    pred = consensus[:, order]
    votes = np.zeros((k, k), dtype=int)
    np.add.at(votes, (pred.flatten(), true.flatten()), 1)
    a, b = linear_sum_assignment(-votes)
    mapping = np.empty(k, dtype=int)
    mapping[a] = b
    diagnostic = {
        "source_query_best_mean": float(quality.reshape(tasks, starts).min(1).mean()),
        "correct_of_512": int((mapping[pred] == true).sum()),
        "selected_models": len(selected),
    }
    (out / "diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
    print(diagnostic, flush=True)


if __name__ == "__main__":
    main()
