"""G8: matched numerical refinement from cold and observed moment starts."""

import argparse
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices
from experiments.sharing_benchmark_v1.linear import Adam

from .nonlinear_moments import labels, make_teacher, project, weight_error

BASE = Path(__file__).resolve().parent


def register(root, parent):
    root.mkdir(parents=True, exist_ok=False)
    parent_spec = json.loads((parent / "protocol.json").read_text())
    plan = {
        "stage": "G8 numerical diagnosis on opened G6/G7 fresh-C geometries; not matched-total-data competitive confirmation",
        "registered_utc": datetime.now(UTC).isoformat(),
        "parent": str(parent.resolve()),
        "parent_protocol_sha256": digest(parent / "protocol.json"),
        "seed": parent_spec["seed"],
        "data_seed_offset": 4000000,
        "families": parent_spec["families"],
        "function": "fresh_gaussian_codes",
        "points": 8,
        "features": 2,
        "point_hidden": 2,
        "codes": 4,
        "inputs": 16,
        "hidden": 16,
        "bias": 0.15,
        "amplitude": 1.0,
        "noise": [0.0, 0.2],
        "initializations": [
            "cold",
            "moment_pairs8388608",
            "moment_pairs67108864",
            "moment_pairs536870912",
        ],
        "source": 262144,
        "readout_support": 8192,
        "validation": 8192,
        "steps": 6000,
        "batch": 1024,
        "checkpoint_every": 200,
        "rates": [0.0003, 0.001, 0.003],
        "seeds": [31, 32, 33],
        "warm_relative_jitter": [0.0, 0.01, 0.05],
        "rate_rule": "Same initial RMS(weight) scaling per row, cosine cooldown to0.01 of initial rate. Three relative rates x three restarts per initialization/noise.",
        "gauge": "Bias fixed to1 by positive homogeneity; all256 weights free. Amplitude>=0 and offset solved by observed linear least squares each minibatch, with envelope gradient. Numeric teacher bias/amplitude not supplied to optimizer.",
        "selection": "Same independent noisy validation MSE selects numeric checkpoint and restart. At validation, amplitude/offset fitted on separate observed readout-support split. No true I and no hard-I validation used. Scalar DP applied afterwards, explicitly a postprocessing control.",
        "data": "New fixed IID Gaussian source/refinement inputs independent of moment stage. Cold and warm receive identical refinement observations and optimizer. Warm additionally consumed stated G7 label budget; this diagnosis is not an equal-total-label benchmark. No teacher autograd, no mask fixed.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": digest(Path(__file__)),
        "checks": "No software tests; scientific objective guards, necessary lint and numerical diagnostics only.",
    }
    save_json(root / "protocol.json", plan)


def features(x, weights):
    return (torch.einsum("nd,sdh->snh", x, weights) + 1).relu().sum(-1) / math.sqrt(
        weights.shape[-1]
    )


def fit_readout(prediction, targets):
    centered = prediction - prediction.mean(-1, keepdim=True)
    response = targets - targets.mean(-1, keepdim=True)
    amplitude = (
        (centered * response).sum(-1) / centered.square().sum(-1).clamp_min(1e-20)
    ).clamp_min(0)
    offset = targets.mean(-1) - amplitude * prediction.mean(-1)
    return amplitude, offset


def run_family(root, plan, family, device):
    folder = root / f"family_{family}"
    folder.mkdir(exist_ok=False)
    teacher, truth, codes = make_teacher(family, plan["function"], plan, device)
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + plan["data_seed_offset"] + family
    )
    data = {}
    for split in ("source", "readout_support", "validation"):
        x = torch.randn(
            (plan[split], plan["inputs"]), generator=generator, device=device
        )
        clean = labels(x.double(), teacher, plan).float()
        noise = torch.randn(len(x), generator=generator, device=device)
        data[split] = x, clean[:, None] + noise[:, None] * x.new_tensor(plan["noise"])
    observation_hash = hashlib.sha256()
    for x, y in data.values():
        observation_hash.update(x.cpu().numpy().tobytes())
        observation_hash.update(y.cpu().numpy().tobytes())
    rows, initial = [], []
    parent = Path(plan["parent"])
    source_file = parent / f"family_{family}" / plan["function"] / "learner_outputs.npz"
    with np.load(source_file) as archive:
        for noise_id, noise in enumerate(plan["noise"]):
            for method in plan["initializations"]:
                for rate in plan["rates"]:
                    for restart_id, seed in enumerate(plan["seeds"]):
                        local = torch.Generator(device=device).manual_seed(
                            plan["seed"] + family + seed * 100003
                        )
                        if method == "cold":
                            value = torch.randn(
                                (plan["inputs"], plan["hidden"]),
                                generator=local,
                                device=device,
                            )
                        else:
                            pairs = int(method.removeprefix("moment_pairs"))
                            key = f"antithetic_affine__joint__noise{noise}__pairs{pairs}__weights_over_bias"
                            value = torch.tensor(
                                archive[key], device=device, dtype=torch.float32
                            )
                            relative = plan["warm_relative_jitter"][restart_id]
                            value = (
                                value
                                + relative
                                * value.square().mean().sqrt()
                                * torch.randn(
                                    value.shape, generator=local, device=device
                                )
                            )
                        initial.append(value)
                        rows.append(
                            {
                                "method": method,
                                "noise": noise,
                                "noise_id": noise_id,
                                "rate": rate,
                                "seed": seed,
                            }
                        )
    weights = torch.stack(initial).requires_grad_()
    rms = weights.detach().square().mean((-1, -2)).sqrt()
    base_rates = rms * weights.new_tensor([r["rate"] for r in rows])
    optimizer = Adam([weights], base_rates.clone())
    noise_ids = torch.tensor([r["noise_id"] for r in rows], device=device)
    best = weights.new_full((len(rows),), torch.inf)
    saved, saved_readout = weights.detach().clone(), weights.new_zeros((len(rows), 2))
    saved_step = torch.zeros(len(rows), dtype=torch.long, device=device)
    history, started = [], time.perf_counter()
    source_x, source_y = data["source"]
    support_x, support_y = data["readout_support"]
    validation_x, validation_y = data["validation"]
    support_y, validation_y = support_y[:, noise_ids].T, validation_y[:, noise_ids].T
    for step in range(plan["steps"] + 1):
        if step % plan["checkpoint_every"] == 0:
            with torch.no_grad():
                amplitude, offset = fit_readout(features(support_x, weights), support_y)
                prediction = (
                    features(validation_x, weights) * amplitude[:, None]
                    + offset[:, None]
                )
                loss = (prediction - validation_y).square().mean(-1)
                improved = loss < best
                saved[improved] = weights[improved]
                saved_readout[improved] = torch.stack((amplitude, offset), -1)[improved]
                saved_step[improved] = step
                best = torch.minimum(best, loss)
                history.append(
                    {
                        "step": step,
                        "validation_mse": loss.tolist(),
                        "seconds": time.perf_counter() - started,
                    }
                )
        if step == plan["steps"]:
            break
        index = torch.randint(
            len(source_x), (plan["batch"],), generator=generator, device=device
        )
        prediction = features(source_x[index], weights)
        targets = source_y[index][:, noise_ids].T
        amplitude, offset = fit_readout(prediction, targets)
        loss = (
            (
                prediction * amplitude.detach()[:, None]
                + offset.detach()[:, None]
                - targets
            )
            .square()
            .mean(-1)
        )
        optimizer.rates.copy_(
            base_rates
            * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / plan["steps"])))
        )
        optimizer.step(torch.autograd.grad(loss.sum(), [weights]))
    selected, arrays = (
        [],
        {
            "all_weights": saved.cpu().numpy(),
            "all_readout": saved_readout.cpu().numpy(),
            "all_validation_mse": best.cpu().numpy(),
            "all_selected_step": saved_step.cpu().numpy(),
            "initial_weights": torch.stack(initial).cpu().numpy(),
        },
    )
    for noise in plan["noise"]:
        for method in plan["initializations"]:
            positions = [
                i
                for i, row in enumerate(rows)
                if row["noise"] == noise and row["method"] == method
            ]
            chosen = min(
                positions,
                key=lambda i: (float(best[i]), rows[i]["seed"], rows[i]["rate"]),
            )
            index, centers, quantized = project(saved[chosen].double(), plan["codes"])
            key = f"{method}__noise{noise}"
            arrays[key + "__indices"] = index
            arrays[key + "__quantized"] = quantized.cpu().numpy()
            with torch.no_grad():
                amplitude, offset = fit_readout(
                    features(support_x, quantized.float()[None]),
                    support_y[chosen : chosen + 1],
                )
                error = (
                    (
                        features(validation_x, quantized.float()[None])
                        * amplitude[:, None]
                        + offset[:, None]
                        - validation_y[chosen : chosen + 1]
                    )
                    .square()
                    .mean()
                )
            selected.append(
                {
                    "method": method,
                    "noise": noise,
                    "key": key,
                    "chosen_row": chosen,
                    "selected_step": int(saved_step[chosen]),
                    "validation_mse": float(best[chosen]),
                    "projected_validation_mse": float(error),
                    "centers": centers.tolist(),
                }
            )
    np.savez_compressed(folder / "learner_outputs.npz", **arrays)
    save_json(
        folder / "learner_records.json",
        {
            "rows": rows,
            "selected": selected,
            "history": history,
            "observation_sha256": observation_hash.hexdigest(),
            "parent_model_sha256": digest(source_file),
        },
    )
    save_json(
        folder / "learner_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "sha256": {
                name: digest(folder / name)
                for name in ("learner_outputs.npz", "learner_records.json")
            },
        },
    )
    audit_arrays = {
        "truth_indices": truth,
        "truth_weights": teacher.cpu().numpy(),
        "truth_codes": codes,
    }
    for row in selected:
        metrics, aligned, _ = align_indices(
            arrays[row["key"] + "__indices"], truth, plan["codes"]
        )
        row["recovery"] = metrics
        chosen = row["chosen_row"]
        fitted = arrays["all_weights"][chosen] * arrays["all_readout"][chosen, 0]
        row["weight_relative_squared_error"] = weight_error(
            fitted, teacher.cpu().numpy()
        )
        audit_arrays[row["key"] + "__aligned_indices"] = aligned
    np.savez_compressed(folder / "audit_only.npz", **audit_arrays)
    save_json(
        folder / "audit.json",
        {
            "family": family,
            "records": selected,
            "frozen_sha256": digest(folder / "learner_frozen.json"),
            "seconds": time.perf_counter() - started,
        },
    )
    print(
        json.dumps(
            {
                "family": family,
                "complete": True,
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--family", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.parent)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.family not in plan["families"]:
        raise ValueError("Family must be registered")
    run_family(args.root, plan, args.family, args.device)


if __name__ == "__main__":
    main()
