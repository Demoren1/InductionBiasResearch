"""G2 exact source-loss control after analytical refitting of each codebook."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment

from .contrasts import noise_whitened_basis, signed_graph
from .gradient_spectrum import BASE, decode
from .kernel_control import basis


def reduced_fit(index, gram, cross, codes):
    outputs, inputs = cross.shape[-2:]
    assigned = basis(index, codes).reshape(outputs, inputs, codes)
    product = torch.einsum("tij,ojk->toik", gram, assigned)
    hessian = torch.einsum("oik,toil->tkl", assigned, product)
    linear = torch.einsum("oik,toi->tk", assigned, cross)
    empty = assigned.abs().sum((0, 1)) == 0
    hessian = hessian + torch.diag(empty.to(hessian.dtype))
    codebook = torch.linalg.solve(hessian, linear[..., None])[..., 0]
    return codebook, (linear * codebook).sum()


@torch.no_grad()
def curvature_partition(index, gram, cross, codes, steps=128):
    current = index.flatten().long().clone()
    _tasks, outputs, inputs = cross.shape
    labels = torch.arange(-codes, codes + 1, device=index.device)
    possible = basis(labels, codes)
    _, previous = reduced_fit(index, gram, cross, codes)
    scores, trace = [float(previous)], [current.cpu().numpy().astype(np.int16)]
    for _ in range(steps):
        assigned = basis(current, codes).reshape(outputs, inputs, codes)
        product = torch.einsum("tij,ojk->toik", gram, assigned).flatten(1, 2)
        hessian = torch.einsum("eK,teL->tKL", assigned.flatten(0, 1), product)
        linear = torch.einsum("oik,toi->tk", assigned, cross)
        delta = possible[None] - assigned.flatten(0, 1)[:, None]
        changed_hessian = (
            hessian[:, None, None]
            + delta[None, ..., :, None] * product[:, :, None, None, :]
            + product[:, :, None, :, None] * delta[None, ..., None, :]
            + gram.diagonal(dim1=-2, dim2=-1).repeat(1, outputs)[:, :, None, None, None]
            * delta[None, ..., :, None]
            * delta[None, ..., None, :]
        )
        counts = assigned.abs().sum((0, 1))
        changed_counts = (
            counts[None, None]
            + possible.abs()[None]
            - assigned.flatten(0, 1).abs()[:, None]
        )
        changed_hessian += torch.diag_embed((changed_counts == 0).double())[None]
        changed_linear = (
            linear[:, None, None] + cross.flatten(1)[:, :, None, None] * delta[None]
        )
        fitted = torch.linalg.solve(changed_hessian, changed_linear[..., None])[..., 0]
        values = (changed_linear * fitted).sum((0, 3))
        winner = int(values.argmax())
        position, column = divmod(winner, len(labels))
        value = values[position, column]
        if float(value - previous) <= max(1e-8, abs(float(previous)) * 1e-13):
            break
        current[position] = labels[column]
        _, actual = reduced_fit(current.reshape(index.shape), gram, cross, codes)
        if not torch.allclose(value, actual, atol=1e-7, rtol=1e-10):
            raise FloatingPointError("Incremental full-curvature score mismatch")
        previous = actual
        scores.append(float(previous))
        trace.append(current.cpu().numpy().astype(np.int16))
    return current.reshape(index.shape), np.stack(trace), scores


def register(root):
    path = root / "curvature_control_registration.json"
    if path.exists():
        raise RuntimeError("Control already registered")
    save_json(
        path,
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "scope": "Diagnostic on already opened six G1 cases, no confirmation claim. Whitening was inspected once on these cases before registration.",
            "starts": ["ols_profiles", "noise_whitened_pca"],
            "criterion": "Exact original source SSE after full-Hessian codebook refit, known discrete L0-fusion style control, no novelty claim.",
            "steps": 128,
            "graph_alpha": 0.05,
            "graph_looks": 1,
            "graph": "First registered calculation of Section9 signed contrast graph; unresolved allowed.",
            "theory_sha256": digest(BASE / "THEORY.md"),
            "code_sha256": digest(Path(__file__)),
            "contrasts_sha256": digest(BASE / "contrasts.py"),
        },
    )
    (root / "curvature_control").mkdir(exist_ok=False)


@torch.no_grad()
def run_family(root, family, device):
    plan = json.loads((root / "protocol.json").read_text())
    output = root / "curvature_control"
    if (output / f"family_{family}.json").exists():
        raise RuntimeError("Family already completed")
    with np.load(root / "kernel_control" / f"family_{family}.npz") as archive:
        gram = torch.tensor(archive["gram"], device=device)
        cross = torch.tensor(archive["cross_yx"], device=device)
        profiles = torch.tensor(archive["ols_profiles"], device=device)
        raw = torch.tensor(archive["kernel_raw"], device=device)
        noise = raw - torch.tensor(archive["kernel_corrected"], device=device)
        truth = archive["truth_audit_only"]
        variances = archive["noise_variance"]
    arrays = {"truth_audit_only": truth}
    vectors, spectrum = noise_whitened_basis(profiles, noise, plan["codes"])
    whitened, _, _ = decode(vectors, plan, plan["seed"] + family)
    with np.load(root / "showcases" / f"family_{family}_n8192.npz") as archive:
        ols = torch.tensor(archive["ols_profiles_indices"], device=device)
    starts = {"ols_profiles": ols, "noise_whitened_pca": whitened.reshape(ols.shape)}
    records = []
    for name, initial in starts.items():
        index, trace, scores = curvature_partition(initial, gram, cross, plan["codes"])
        recovery, aligned = alignment(index.cpu().numpy(), truth, plan["codes"])
        arrays[f"{name}_indices"] = index.cpu().numpy()
        arrays[f"{name}_aligned_indices"] = aligned
        arrays[f"{name}_trace"] = trace
        _, true_score = reduced_fit(
            torch.tensor(truth, device=device), gram, cross, plan["codes"]
        )
        records.append(
            {
                "start": name,
                "recovery": recovery,
                "scores": scores,
                "truth_minus_score_audit": float(true_score) - scores[-1],
            }
        )
    graph_index, graph, components = signed_graph(
        profiles.T.reshape(128, 14, 16),
        torch.linalg.inv(gram),
        float(variances.mean()),
        128 * 14 * (8192 - 16),
        plan["codes"],
        0.05,
    )
    if graph_index is not None:
        graph["recovery"], arrays["graph_aligned_indices"] = alignment(
            graph_index, truth, plan["codes"]
        )
        arrays["graph_indices"] = graph_index
    arrays["graph_components"] = components
    arrays["whitened_spectrum"] = spectrum.cpu().numpy()
    np.savez_compressed(output / f"family_{family}.npz", **arrays)
    save_json(
        output / f"family_{family}.json",
        {"family": family, "records": records, "graph": graph},
    )
    print(family, records, "graph", graph, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--families", type=int, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root)
    else:
        for family in args.families:
            run_family(args.root, family, args.device)


if __name__ == "__main__":
    main()
