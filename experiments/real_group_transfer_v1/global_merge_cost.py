"""Refine the signed global FC100 indexator with compensated split–merge moves."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .merge_cost import (
    BUDGET,
    DATA,
    LAYOUT,
    OUTPUT,
    PREVIOUS,
    heldout_bound,
    source_data,
)

SPLIT_CANDIDATES = 64
MERGE_NEIGHBORS = 6
MOVES = 32


class SharedQuadratic:
    """Task-specific codes with one signed grouping shared across tasks."""

    def __init__(self, source, metric, device):
        dims = LAYOUT.fan_in
        weights = source.weights.transpose(0, 1, 3, 2).reshape(
            -1, LAYOUT.hidden, dims
        )
        importance = np.maximum(np.linalg.norm(source.head, axis=-1), 1e-8)
        covariance = (
            source.covariance if metric == "covariance" else np.eye(dims)
        )
        hessian = importance[..., None, None] * np.expand_dims(covariance, -3)
        self.weight = torch.as_tensor(weights, device=device)
        curvature = hessian.reshape(*weights.shape, dims)
        self.curvature = torch.as_tensor(curvature, device=device)
        self.weighted = (self.curvature @ self.weight.unsqueeze(-1)).squeeze(
            -1
        )
        self.diagonal = self.curvature.diagonal(dim1=-2, dim2=-1).flatten(1)
        self.device = device

    def solve(self, labels, signs, *, inverse):
        tasks, neurons, dims = self.weight.shape
        observed = labels.reshape(neurons, dims)
        sign = signs.reshape(neurons, dims)
        pairs = observed[:, :, None] * BUDGET + observed[:, None, :]
        entries = self.curvature * (sign[:, :, None] * sign[:, None, :])
        gram = entries.new_zeros(tasks, BUDGET * BUDGET)
        gram.scatter_add_(
            1, pairs.reshape(1, -1).expand(tasks, -1), entries.flatten(1)
        )
        factor = torch.linalg.cholesky(gram.reshape(tasks, BUDGET, BUDGET))
        rhs = entries.new_zeros(tasks, BUDGET)
        rhs.scatter_add_(
            1, labels.expand(tasks, -1), self.weighted.flatten(1) * signs
        )
        point = torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)
        return point, torch.cholesky_inverse(factor) if inverse else None

    def approximation(self, labels, signs, point):
        return (point[:, labels] * signs).reshape(self.weight.shape)

    def error(self, approximation):
        residual = self.weight - approximation
        direction = (self.curvature @ residual.unsqueeze(-1)).squeeze(-1)
        return float((residual * direction).sum())

    def split(self, labels, signs, point, precision):
        tasks, _, dims = self.weight.shape
        residual = self.weight - self.approximation(labels, signs, point)
        gradient = (
            (self.curvature @ residual.unsqueeze(-1)).squeeze(-1).flatten(1)
        )
        viable = torch.bincount(labels)[labels] > 1
        local = (gradient.square() / self.diagonal).sum(dim=0)
        local = local.masked_fill(~viable, -torch.inf)
        proposed = local.topk(min(SPLIT_CANDIDATES, int(viable.sum()))).indices
        starts = (proposed // dims) * dims
        relative = torch.arange(dims, device=self.device)
        positions = starts[:, None] + relative
        curvature = (
            self.curvature.flatten(1, 2)[:, proposed] * signs[positions]
        )
        cross = curvature.new_zeros(tasks, len(proposed), BUDGET)
        cross.scatter_add_(
            2, labels[positions].expand(tasks, -1, -1), curvature
        )
        cross = cross.transpose(1, 2)
        projection = precision @ cross
        schur = self.diagonal[:, proposed] - (cross * projection).sum(dim=1)
        gain = (gradient[:, proposed].square() / schur).sum(dim=0)
        candidate = int(gain.argmax())
        pixel = int(proposed[candidate])
        parent, sign = int(labels[pixel]), int(signs[pixel])
        z, novelty = projection[:, :, candidate], schur[:, candidate]
        delta = gradient[:, pixel] / novelty
        new_point = point - z * delta[:, None]
        extra = delta + sign * new_point[:, parent]
        new_point = torch.cat((new_point, extra[:, None]), dim=1)

        # Update the inverse Gram when a singleton's indicator is added. The
        # new code represents the whole weight at the pixel, not its increment.
        augmented = precision.new_empty(tasks, BUDGET + 1, BUDGET + 1)
        augmented[:, :-1, :-1] = torch.baddbmm(
            precision, z.unsqueeze(2) / novelty[:, None, None], z.unsqueeze(1)
        )
        border = -z / novelty[:, None]
        column = border + sign * augmented[:, :-1, parent]
        corner = (
            1 / novelty
            + 2 * sign * border[:, parent]
            + augmented[:, parent, parent]
        )
        augmented[:, :-1, -1] = column
        augmented[:, -1, :-1] = column
        augmented[:, -1, -1] = corner
        return pixel, new_point, augmented, float(gain[candidate])

    def merge(self, point, precision):
        # Screen near pairs in profile distance. Compare both signs with the
        # exact constrained-refit cost on every task for shortlisted pairs.
        magnitude = point.square().sum(dim=0)
        distance = (
            magnitude[:, None] + magnitude[None] - 2 * (point.T @ point).abs()
        )
        distance.fill_diagonal_(torch.inf)
        closest = distance.topk(MERGE_NEIGHBORS, largest=False).indices
        row = torch.arange(point.shape[1], device=self.device)
        left = torch.minimum(row[:, None], closest).flatten()
        right = torch.maximum(row[:, None], closest).flatten()
        diag = precision.diagonal(dim1=1, dim2=2)
        curvature = diag[:, left] + diag[:, right]
        correlation = precision[:, left, right]
        costs = torch.stack(
            [
                (
                    (point[:, left] - eps * point[:, right]).square()
                    / (curvature - 2 * eps * correlation)
                ).sum(dim=0)
                for eps in (1, -1)
            ]
        )
        sign, pair = divmod(int(costs.argmin()), len(left))
        return (
            int(left[pair]),
            int(right[pair]),
            1 - 2 * sign,
            float(costs[sign, pair]),
        )

    def search(self, initial):
        index = torch.as_tensor(
            initial.T.ravel(), device=self.device, dtype=torch.long
        )
        labels, signs = index.abs() - 1, index.sign()
        point, precision = self.solve(labels, signs, inverse=True)
        q = self.error(self.approximation(labels, signs, point))
        trace = [{"q": q}]
        for _ in range(MOVES):
            pixel, estimate, expanded, gain = self.split(
                labels, signs, point, precision
            )
            left, right, sign, price = self.merge(estimate, expanded)
            if gain - price <= 1e-10 * max(1.0, q):
                break
            d = expanded[:, :, left] - sign * expanded[:, :, right]
            denom = d[:, left] - sign * d[:, right]
            offset = (estimate[:, left] - sign * estimate[:, right]) / denom
            keep = torch.cat(
                (
                    torch.arange(right, device=self.device),
                    torch.arange(right + 1, BUDGET + 1, device=self.device),
                )
            )
            point = (estimate - d * offset[:, None])[:, keep].contiguous()
            expanded.baddbmm_(
                -d.unsqueeze(2) / denom[:, None, None], d.unsqueeze(1)
            )
            precision = expanded[:, keep[:, None], keep].contiguous()
            labels[pixel], signs[pixel] = BUDGET, 1
            merged = labels == right
            signs[merged] *= sign
            labels[merged] = left
            labels -= (labels > right).long()
            measured = self.error(self.approximation(labels, signs, point))
            discrepancy = abs((q - measured) - (gain - price))
            if discrepancy > max(1.0, q) * 1e-8:
                raise ValueError(
                    f"Compensated update disagrees with measured loss: {discrepancy}"
                )
            q = measured
            trace.append(
                {
                    "q": q,
                    "split_gain": gain,
                    "merge_cost": price,
                    "relative_sign": sign,
                    "refit_error": discrepancy,
                }
            )
        index = ((labels + 1) * signs).reshape(initial.shape[::-1]).T
        return index.cpu().numpy().astype(np.int16), trace


@torch.inference_mode()
def refine(source, initial, metric, gpu):
    device = torch.device(f"cuda:{gpu}")
    started = time.perf_counter()
    quadratic = SharedQuadratic(source, metric, device)
    index, trace = quadratic.search(initial)
    elapsed = time.perf_counter() - started
    if not np.array_equal(np.unique(np.abs(index)), np.arange(1, BUDGET + 1)):
        raise ValueError("Signed refinement changed the code budget")
    if metric != "covariance":
        quadratic = SharedQuadratic(source, "covariance", device)
    diagnostics = {}
    for name, groups in (("ward", initial), ("refined", index)):
        coded = torch.as_tensor(
            groups.T.ravel(), device=device, dtype=torch.long
        )
        labels, signs = coded.abs() - 1, coded.sign()
        point, _ = quadratic.solve(labels, signs, inverse=False)
        estimated = quadratic.approximation(labels, signs, point)
        approximation = (
            estimated.reshape(
                -1,
                LAYOUT.sites,
                LAYOUT.channels,
                LAYOUT.fan_in,
            )
            .transpose(-2, -1)
            .cpu()
            .numpy()
        )
        diagnostics[name] = {
            "refit_q": quadratic.error(estimated),
            **heldout_bound(source, approximation),
        }
    print(
        f"{metric}: {len(trace) - 1} moves, "
        f"source covariance Q={diagnostics['refined']['refit_q']:.4f}",
        flush=True,
    )
    return index, {
        "seconds": elapsed,
        "trace": trace,
        "diagnostics": diagnostics,
    }


def run(gpus):
    out = OUTPUT / "global_indexators.npz"
    if out.exists():
        raise FileExistsError(out)
    if len(gpus) != 2 or gpus[0] == gpus[1]:
        raise ValueError(
            "Provide two distinct GPUs for the paired source refinement"
        )
    with np.load(
        PREVIOUS / "frozen_indexators.npz", allow_pickle=False
    ) as archive:
        initial = archive["asrr"]
    with np.load(DATA / "fc100.npz", allow_pickle=False) as archive:
        src = source_data(dict(archive))
    metrics = ("identity", "covariance")
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(refine, src, initial, metric, gpu)
            for metric, gpu in zip(metrics, gpus, strict=True)
        ]
        results = {
            metric: job.result()
            for metric, job in zip(metrics, jobs, strict=True)
        }
    np.savez_compressed(
        out, **{f"global_{k}": v[0] for k, v in results.items()}
    )
    save_json(
        OUTPUT / "global_diagnostics.json",
        {
            "metrics": {k: v[1] for k, v in results.items()},
            "budget": BUDGET,
            "moves": MOVES,
            "split_candidates": SPLIT_CANDIDATES,
            "merge_neighbors": MERGE_NEIGHBORS,
            "indexators_sha256": digest(out),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=2, type=int, default=[0, 1])
    run(parser.parse_args().gpus)
