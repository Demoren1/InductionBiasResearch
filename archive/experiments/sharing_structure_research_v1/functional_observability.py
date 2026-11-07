"""G18: low-order moment ambiguity and known mixed-functional contrasts.

The graph learner observes only query panels/replicated noisy labels. Population
moments and weight structure are evaluated separately, after whole-cohort freeze.
The resulting graph describes input interactions; it is not a weight indexator.
"""

import argparse
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from scipy.stats import f

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .functional_moments import population

BASE = Path(__file__).resolve().parent


def register(root):
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "stage": "G18 prospective observability diagnosis, known functional ANOVA contrast, not a complete/new I learner",
        "registered_utc": datetime.now(UTC).isoformat(),
        "seed": 26110518017,
        "families": list(range(8801, 8807)),
        "tasks": 8,
        "inputs": 16,
        "hidden": 16,
        "codes": 4,
        "bias": 0.15,
        "variants": ["local", "quartet_mix"],
        "observations": [512, 2048, 8192, 32768, 131072],
        "noise": [0.2, 1.0, 2.0],
        "chunk": 256,
        "replicates": 2,
        "alpha": 0.05,
        "jobs": [
            {"family": family, "variant": variant}
            for family in range(8801, 8807)
            for variant in ("local", "quartet_mix")
        ],
        "teacher_only": "8 repeated2D modules,2ReLU per module,4freshGaussian codes per source. Orthogonal four-point mixing produces another signed4-code I with C/2. Input and hidden coordinates permuted, unknown to graph learner.32 versus128 active weights.",
        "query_policy": "At each base X,Z: base, every single replacement and every pair replacement;Q137. All queries marginalGaussian, panel dependent. Two independent Gaussian labels per query. Same X,Z,unit noises and underlying C across variants/noise for paired diagnosis. Count2*T*N*Q labels; raw panels not stored. Such queries require active/simulator access, cannot be reconstructed from arbitrary IID files.",
        "learner": "For every pair, squared mean-label rectangle contrast divided by2 observed sigmahat^2, estimated independently from all replicate differences. Exact null F(TN,TNQ). No teacher mask/component count/code values or population moments in decision.",
        "alpha_scope": "Per family/variant/noise run across all5 looks and120 pair nulls. No whole-benchmark/cross-algorithm guarantee. Controls false edges only, not omissions, correct components or full weight-I.",
        "selection": "No HPO, teacher stopping or look selection. Store every registered graph. Freeze all12 network source outputs before structural/population oracle audit.",
        "oracle_audit": "Independently evaluate population m0/m1/m2,label-square H2/gradient anisotropic part,m3; different f on same fresh Gaussian X and equal f under known orthogonal input transform. This is oracle construction verification, not learned-I performance. Input interaction graph scored separately from I.",
        "limits": "Known Sobol/pick-freeze statistic; no novelty or competitor superiority claim. Cheap bilinear Sobol estimators and same-panel native LC/Yeh/LUT-Q still required before full-I comparison. Gaussian iid measurement noise only; heavy-tail/heteroscedastic coverage unproved.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "functional_observability.py",
                "functional_moments.py",
            )
        },
        "checks": "No software tests. Necessary mathematical/provenance/finiteness guards, minimal lint and figure inspection only.",
    }
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in plan["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    (root / "models").mkdir()


def teacher(family, variant, plan, device):
    rng = np.random.default_rng(plan["seed"] + family * 31)
    row_order, column_order = rng.permutation(16), rng.permutation(16)
    index = np.zeros((16, 16), dtype=np.int16)
    for point in range(8):
        for other in range(8):
            if variant == "local":
                sign = int(point == other)
            else:
                sign = (-1 if point == other else 1) * int(point // 4 == other // 4)
            index[2 * point : 2 * point + 2, 2 * other : 2 * other + 2] = (
                sign * np.arange(1, 5).reshape(2, 2)
            )
    index = index[row_order][:, column_order]
    centers = rng.normal(size=(plan["tasks"], plan["codes"]))
    if variant == "quartet_mix":
        centers /= 2
    indices = torch.tensor(index, device=device).long()
    codes = torch.tensor(centers, device=device)
    weights = codes[:, indices.abs().clamp_min(1) - 1] * indices.sign()[None]
    q4 = np.ones((4, 4)) / 2 - np.eye(4)
    q8 = np.kron(np.eye(2), q4)
    transform = np.kron(q8, np.eye(2))[np.ix_(row_order, row_order)]
    return weights, index, centers, transform


def queries(plan, device):
    inputs = plan["inputs"]
    pairs = np.array([(i, j) for i in range(inputs) for j in range(i + 1, inputs)])
    mask = torch.zeros(
        (1 + inputs + len(pairs), inputs), dtype=torch.bool, device=device
    )
    mask[1 : inputs + 1] = torch.eye(inputs, device=device, dtype=torch.bool)
    for pair, (i, j) in enumerate(pairs):
        mask[inputs + 1 + pair, [i, j]] = True
    return mask, pairs


def response(x, weights, plan):
    return (torch.einsum("nqi,tih->tnqh", x, weights) + plan["bias"]).relu().sum(
        -1
    ) / math.sqrt(plan["hidden"])


@torch.no_grad()
def run_case(root, plan, job_id, device):
    job = plan["jobs"][job_id]
    started = time.monotonic()
    weights, _, _, _ = teacher(job["family"], job["variant"], plan, device)
    masks, pairs = queries(plan, device)
    queries_per_base = len(masks)
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + job["family"] * 31 + 7
    )
    noises = weights.new_tensor(plan["noise"])
    sums = weights.new_zeros((len(noises), len(pairs)))
    cross_products = torch.zeros_like(sums)
    replicate_square = weights.new_zeros(len(noises))
    pair_i = torch.tensor(pairs[:, 0] + 1, device=device)
    pair_j = torch.tensor(pairs[:, 1] + 1, device=device)
    arrays, records = {}, []
    for end in range(plan["chunk"], max(plan["observations"]) + 1, plan["chunk"]):
        xz = torch.randn(
            (2, plan["chunk"], plan["inputs"]),
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        panel = torch.where(masks[None], xz[1, :, None], xz[0, :, None])
        clean = response(panel, weights, plan)
        error = torch.randn(
            (*clean.shape, 2), generator=generator, device=device, dtype=torch.float64
        )
        observed = (
            clean[None, ..., None] + noises[:, None, None, None, None] * error[None]
        )
        difference = observed[..., 0] - observed[..., 1]
        replicate_square += difference.square().sum((1, 2, 3))
        rectangle = (
            observed[:, :, :, plan["inputs"] + 1 :, :] - observed[:, :, :, pair_i, :]
        )
        rectangle -= observed[:, :, :, pair_j, :]
        rectangle += observed[:, :, :, :1, :]
        sums += rectangle.mean(-1).square().sum((1, 2))
        cross_products += (rectangle[..., 0] * rectangle[..., 1]).sum((1, 2))
        if end not in plan["observations"]:
            continue
        degrees = plan["tasks"] * end
        variance = replicate_square / (2 * degrees * queries_per_base)
        statistic = sums / (2 * degrees * variance[:, None])
        critical = float(
            f.isf(
                plan["alpha"] / (len(pairs) * len(plan["observations"])),
                degrees,
                degrees * queries_per_base,
            )
        )
        edges = statistic > critical
        for sigma_id, sigma in enumerate(plan["noise"]):
            graph = np.zeros((plan["inputs"], plan["inputs"]), dtype=np.uint8)
            accepted = edges[sigma_id].cpu().numpy()
            graph[pairs[:, 0], pairs[:, 1]] = accepted
            graph |= graph.T
            key = f"noise{sigma}_n{end}"
            arrays[key + "__graph"] = graph
            arrays[key + "__energy"] = (
                (cross_products[sigma_id] / degrees).cpu().numpy()
            )
            arrays[key + "__F"] = statistic[sigma_id].cpu().numpy()
            records.append(
                {
                    "key": key,
                    "noise": sigma,
                    "n": end,
                    "label_budget": 2 * degrees * queries_per_base,
                    "critical_F": critical,
                    "estimated_noise_variance": float(variance[sigma_id]),
                    "detected_edges": int(accepted.sum()),
                }
            )
        print(
            json.dumps(
                {"job": job_id, "n": end, "seconds": time.monotonic() - started}
            ),
            flush=True,
        )
    if not np.isfinite(statistic.cpu().numpy()).all() or not np.isfinite(critical):
        raise FloatingPointError("Nonfinite observed functional statistic")
    path = root / "models" / f"job_{job_id:03d}"
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    save_json(
        path.with_suffix(".json"),
        {
            "job": job,
            "records": records,
            "queries_per_base": queries_per_base,
            "protocol_sha256": digest(root / "protocol.json"),
            "seconds": time.monotonic() - started,
        },
    )


def freeze(root, plan):
    hashes = {}
    for i in range(len(plan["jobs"])):
        for suffix in (".npz", ".json"):
            path = root / "models" / (f"job_{i:03d}" + suffix)
            hashes[str(path.relative_to(root))] = digest(path)
    save_json(
        root / "models_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "sha256": hashes,
        },
    )


@torch.no_grad()
def audit(root, plan, device):
    frozen = json.loads((root / "models_frozen.json").read_text())
    records, oracle = [], []
    for family in plan["families"]:
        local, local_index, _, transform = teacher(family, "local", plan, device)
        mixed, mixed_index, _, _ = teacher(family, "quartet_mix", plan, device)
        moments0 = [population(w, plan["bias"]) for w in local.cpu().numpy()]
        moments1 = [population(w, plan["bias"]) for w in mixed.cpu().numpy()]
        moment_errors = {
            name: max(
                float(np.max(np.abs(a[name] - b[name])))
                for a, b in zip(moments0, moments1, strict=True)
            )
            for name in moments0[0]
            if name != "m3"
        }
        third_relative = [
            float(
                np.linalg.norm(a["m3"] - b["m3"]) / max(np.linalg.norm(a["m3"]), 1e-30)
            )
            for a, b in zip(moments0, moments1, strict=True)
        ]
        if max(moment_errors.values()) > 1e-8:
            raise FloatingPointError("Constructed population moment equality failed")
        generator = torch.Generator(device=device).manual_seed(
            plan["seed"] + family * 31 + 19
        )
        x = torch.randn(
            (16384, 1, plan["inputs"]),
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        original, alternative = response(x, local, plan), response(x, mixed, plan)
        transformed = response(x @ torch.tensor(transform, device=device), mixed, plan)
        oracle.append(
            {
                "family": family,
                "max_moment_absolute_errors": moment_errors,
                "third_moment_relative_difference": third_relative,
                "same_x_function_mse": float((original - alternative).square().mean()),
                "orthogonal_coupled_max_error": float(
                    (original - transformed).abs().max()
                ),
                "active_weights": [
                    int((local_index != 0).sum()),
                    int((mixed_index != 0).sum()),
                ],
            }
        )
        np.savez_compressed(
            root / "models" / f"family_{family}_oracle_only.npz",
            local_I=local_index,
            mixed_I=mixed_index,
            local_m2=np.stack([m["m2"] for m in moments0]),
            mixed_m2=np.stack([m["m2"] for m in moments1]),
            local_gradient_anisotropic=np.stack(
                [m["gradient_covariance_minus_output_square_I"] for m in moments0]
            ),
            mixed_gradient_anisotropic=np.stack(
                [m["gradient_covariance_minus_output_square_I"] for m in moments1]
            ),
        )
    for i, job in enumerate(plan["jobs"]):
        path = root / "models" / f"job_{i:03d}"
        for suffix in (".npz", ".json"):
            candidate = path.with_suffix(suffix)
            if digest(candidate) != frozen["sha256"][str(candidate.relative_to(root))]:
                raise RuntimeError("Frozen G18 output changed")
        metadata = json.loads(path.with_suffix(".json").read_text())
        _, index, _, _ = teacher(job["family"], job["variant"], plan, "cpu")
        active = (index != 0).astype(np.int16)
        truth = active @ active.T > 0
        np.fill_diagonal(truth, False)
        _, true_components = connected_components(truth, directed=False)
        true_partition = true_components[:, None] == true_components[None]
        with np.load(path.with_suffix(".npz")) as a:
            for record in metadata["records"]:
                graph = a[record["key"] + "__graph"].astype(bool)
                count, component = connected_components(graph, directed=False)
                same = component[:, None] == component[None]
                false_edges = int(np.triu(graph & ~truth, k=1).sum())
                missing = int(np.triu(truth & ~graph, k=1).sum())
                records.append(
                    {
                        "job_id": i,
                        "job": job,
                        **record,
                        "false_edges": false_edges,
                        "missing_edges": missing,
                        "exact_graph": false_edges == missing == 0,
                        "exact_partition": bool(np.array_equal(same, true_partition)),
                        "components": count,
                        "true_components": int(true_components.max() + 1),
                        "edge_recall": float(
                            (np.triu(graph & truth, k=1)).sum()
                            / np.triu(truth, k=1).sum()
                        ),
                    }
                )
    save_json(
        root / "audit.json",
        {
            "records": records,
            "population_oracle": oracle,
            "cohort_frozen_sha256": digest(root / "models_frozen.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "freeze", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action in ("freeze", "audit"):
        output = args.root / (
            "models_frozen.json" if args.action == "freeze" else "audit.json"
        )
        if output.exists():
            raise RuntimeError("Scientific phase already completed")
        if args.action == "freeze":
            freeze(args.root, plan)
        else:
            audit(args.root, plan, args.device)
        return
    for i in range(args.shard, len(plan["jobs"]), args.shards):
        if (args.root / "models" / f"job_{i:03d}.json").exists():
            raise RuntimeError("Refusing to repeat completed scientific fit")
        run_case(args.root, plan, i, args.device)


if __name__ == "__main__":
    main()
