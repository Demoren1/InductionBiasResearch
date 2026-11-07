"""Diagnose and improve fresh numerical fits with the Pattern 0110 I fixed."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pattern.datasets import input_table, labels_for

from .data import BASE, digest, load_archive, save_json
from .model import balanced_loss, clipped, initial_numeric, metrics
from .report import save_figure, table
from .single import fit_fixed, fresh_parameters, predict, score_split

SETTINGS = [
    {
        "name": "legacy",
        "backend": "legacy",
        "batch": 128,
        "paired_initialization_steps": 0,
    },
    {
        "name": "torch_minibatch",
        "backend": "torch",
        "batch": 128,
        "paired_initialization_steps": 0,
    },
    {
        "name": "torch_fullbatch",
        "backend": "torch",
        "batch": 768,
        "paired_initialization_steps": 0,
    },
    {
        "name": "paired_64",
        "backend": "legacy",
        "batch": 128,
        "paired_initialization_steps": 64,
    },
    {
        "name": "paired_256",
        "backend": "legacy",
        "batch": 128,
        "paired_initialization_steps": 256,
    },
]


def activations(inputs, indices, parameters):
    codebook = torch.cat([torch.zeros_like(parameters[:, :1]), parameters[:, :2]], 1)
    weights = codebook.gather(1, indices.flatten(1)).reshape(-1, 11, 8)
    return (
        torch.einsum("ni,fih->fnh", inputs, weights) + parameters[:, 2, None, None]
    ).relu()


def torch_adam(data, indices, initial, spec, rate, seed, device):
    started = time.perf_counter()
    index = torch.tensor(indices, dtype=torch.long, device=device)
    parameters = torch.tensor(
        initial, dtype=torch.float32, device=device
    ).requires_grad_()
    optimizer = torch.optim.Adam([parameters], lr=rate)
    x = torch.tensor(data["support_x"], device=device)
    y = torch.tensor(data["support_y"][0], device=device)
    rng = torch.Generator(device=device).manual_seed(seed + 701)
    best = parameters.detach().clone()
    best_loss = torch.full((len(initial),), torch.inf, device=device)
    best_step = torch.zeros(len(initial), dtype=torch.long, device=device)
    steps, train_history, query_history, recall_history = [], [], [], []
    parameter_history, gradient_history, activation_history = [], [], []
    for step in range(spec["steps"] + 1):
        if step % spec["checkpoint_every"] == 0 or step == spec["steps"]:
            query = score_split(data, "query", index, parameters)
            training = score_split(data, "support", index, parameters)
            with torch.no_grad():
                improved = query[:, 0] < best_loss
                best_loss[improved] = query[improved, 0]
                best[improved] = parameters[improved]
                best_step[improved] = step
                parameter_history.append(parameters.detach().cpu().numpy().copy())
                activation_history.append(
                    (activations(x, index, parameters) > 0)
                    .float()
                    .mean((1, 2))
                    .cpu()
                    .numpy()
                )
            data_gradient = torch.autograd.grad(
                balanced_loss(predict(x, index, parameters), y[None]).sum(), parameters
            )[0]
            gradient_history.append(data_gradient.cpu().numpy())
            steps.append(step)
            train_history.append(training[:, 0].cpu().numpy())
            query_history.append(query[:, 0].cpu().numpy())
            recall_history.append(query[:, 2].cpu().numpy())
            if not torch.isfinite(query).all():
                raise FloatingPointError("Non-finite numerical Adam fit")
        if step == spec["steps"]:
            break
        ids = (
            torch.arange(len(x), device=device)
            if spec["batch"] == len(x)
            else torch.randint(len(x), (spec["batch"],), generator=rng, device=device)
        )
        objective = balanced_loss(predict(x[ids], index, parameters), y[None, ids])
        objective = objective + spec["numeric_l2"] * parameters.square().sum(-1) / 2
        gradient = torch.autograd.grad(objective.sum(), parameters)[0]
        parameters.grad = clipped(gradient, spec["gradient_clip"], (-1,))
        optimizer.step()
    result = {
        "indices": indices,
        "initial_parameters": initial,
        "parameters": best.cpu().numpy(),
        "last_parameters": parameters.detach().cpu().numpy(),
        "best_step": best_step.cpu().numpy(),
        "history_steps": np.array(steps),
        "train_history": np.stack(train_history),
        "query_history": np.stack(query_history),
        "recall_history": np.stack(recall_history),
        "parameter_history": np.stack(parameter_history),
        "data_gradient_history": np.stack(gradient_history),
        "activation_history": np.stack(activation_history),
        "seconds": np.array(time.perf_counter() - started),
    }
    for split in ("support", "query", "selection"):
        result[split + "_metrics"] = score_split(data, split, index, best).cpu().numpy()
    return result


def prepare(parent):
    folder = parent / "numeric_optimization_20261104"
    folder.mkdir(exist_ok=False)
    plan = json.loads((parent / "protocol.json").read_text())
    archive = load_archive(parent / "joint_tuning/evaluation_fits.npz")
    oracle = archive["methods"].tolist().index("oracle")
    ids = np.arange(oracle * 32, (oracle + 1) * 32)
    original = {}
    for key in (
        "indices",
        "initial_parameters",
        "parameters",
        "best_step",
        "history_steps",
        "train_history",
        "query_history",
        "recall_history",
        "support_metrics",
        "query_metrics",
        "selection_metrics",
        "test_metrics",
        "population_metrics",
    ):
        original[key] = (
            archive[key]
            if key == "history_steps"
            else archive[key][:, ids]
            if key.endswith("history")
            else archive[key][ids]
        )
    np.savez_compressed(folder / "original_oracle.npz", **original)
    save_json(
        folder / "protocol.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "seed": plan["seed"],
            "fit": plan["fit"],
            "rate": 0.03,
            "norm": 0.5,
            "calibration_starts": 32,
            "confirmation_starts": 64,
            "confirmation_families": ["opposite_signs", "unconstrained_gaussian"],
            "settings": SETTINGS,
            "source_sha256": digest(parent / "source.npz"),
            "original_oracle_sha256": digest(folder / "original_oracle.npz"),
            "rules": [
                "Fix the original true Pattern 0110 indexator; no architecture updates. Fit only two code values and shared bias/readout/output offset.",
                "Reproduce original 32 starts with standard torch.optim.Adam using the same initial values, support minibatches, L2 and rowwise clipping; no software tests.",
                "Compare legacy Adam, standard PyTorch Adam, full-batch Adam, and two data-driven paired-polarity initializations on 32 new raw initializations. All use the previously source-selected lr.03 and norm.5.",
                "Paired initialization tries C and -C for 64/256 support updates each, chooses a branch by query BCE, preserves its Adam moments, and continues training. No desired signs or analytic code values are provided.",
                "Budget4096 counts both warmup branches. Thus the selected branch receives4096-warmup updates; discarded branch warmup costs the remaining warmup updates. Full-batch Adam uses more observations per update and is explicitly a noise control.",
                "Mean selection BCE chooses the policy before confirmation/test. Confirm selected policy and unmodified controls on64 new raw initializations. Query chooses checkpoints; report all-start accuracy and success counts, not just the best start.",
                "Existing task test results have already been inspected; this is an explicitly exploratory numerical diagnosis, not a new benchmark. Structural-search results are preserved.",
            ],
        },
    )
    data = load_archive(parent / "source.npz")
    index = torch.tensor(original["indices"], dtype=torch.long)
    numerical = torch.tensor(original["parameters"]).requires_grad_()
    inputs = torch.tensor(data["support_x"])
    targets = torch.tensor(data["support_y"][0])
    gradient = torch.autograd.grad(
        balanced_loss(predict(inputs, index, numerical), targets[None]).sum(), numerical
    )[0]
    with torch.no_grad():
        active = (
            (activations(inputs, index, numerical) > 0).float().mean((1, 2)).numpy()
        )
    save_json(
        folder / "original_diagnosis.json",
        {
            "initial_parameters": original["initial_parameters"].tolist(),
            "best_parameters": original["parameters"].tolist(),
            "active_relu_fraction": active.tolist(),
            "data_gradient": gradient.numpy().tolist(),
            "test_ba": original["test_metrics"][:, 1].tolist(),
            "negative_first_code": (original["initial_parameters"][:, 0] < 0).tolist(),
            "dead_relu_count": int((active == 0).sum()),
        },
    )
    snapshot = folder / "code_snapshot"
    snapshot.mkdir()
    for name in ("numeric_optimization.py", "single.py", "model.py", "data.py"):
        shutil.copy2(BASE / name, snapshot / name)
    print("PREPARED", folder, flush=True)


def reproduce(parent, device):
    folder = parent / "numeric_optimization_20261104"
    plan = json.loads((folder / "protocol.json").read_text())
    old = load_archive(folder / "original_oracle.npz")
    result = torch_adam(
        load_archive(parent / "source.npz"),
        old["indices"],
        old["initial_parameters"],
        plan["fit"],
        plan["rate"],
        plan["seed"] + 14000001,
        device,
    )
    np.savez_compressed(folder / "reproduction_torch.npz", **result)
    print("REPRODUCED", float(result["selection_metrics"][:, 1].mean()), flush=True)


def fit_setting(parent, plan, setting, initial, seed, device):
    folder = parent / "numeric_optimization_20261104"
    data = load_archive(parent / "source.npz")
    oracle = load_archive(folder / "original_oracle.npz")["indices"][0]
    indices = np.broadcast_to(oracle, (len(initial), 11, 8)).copy()
    spec = {
        **plan["fit"],
        "batch": setting["batch"],
        "paired_initialization_steps": setting["paired_initialization_steps"],
    }
    if setting["backend"] == "torch":
        return torch_adam(data, indices, initial, spec, plan["rate"], seed, device)
    return fit_fixed(
        data, indices, initial, [plan["rate"]] * len(initial), spec, seed, device
    )


def calibrate(parent, device):
    folder = parent / "numeric_optimization_20261104"
    plan = json.loads((folder / "protocol.json").read_text())
    initial = fresh_parameters(
        plan["calibration_starts"], plan["norm"], plan["seed"] + 18000001
    )
    ranking = []
    for setting in plan["settings"]:
        result = fit_setting(
            parent, plan, setting, initial, plan["seed"] + 18000001, device
        )
        np.savez_compressed(
            folder / ("calibration_" + setting["name"] + ".npz"), **result
        )
        row = {
            **setting,
            "selection_bce": float(result["selection_metrics"][:, 0].mean()),
            "selection_ba": float(result["selection_metrics"][:, 1].mean()),
            "selection_perfect": int(
                (result["selection_metrics"][:, 1] >= 1 - 1e-6).sum()
            ),
        }
        ranking.append(row)
        print("CALIBRATION", row, flush=True)
    ranking.sort(key=lambda row: (row["selection_bce"], row["name"]))
    save_json(
        folder / "policy_selection.json",
        {
            "selected": ranking[0],
            "ranking": ranking,
            "selection_utc": datetime.now(UTC).isoformat(),
            "confirmation_test_evaluated": False,
        },
    )


def confirm(parent, device):
    folder = parent / "numeric_optimization_20261104"
    plan = json.loads((folder / "protocol.json").read_text())
    selected = json.loads((folder / "policy_selection.json").read_text())["selected"]
    names = list(dict.fromkeys(["legacy", "torch_minibatch", selected["name"]]))
    settings = [
        next(row for row in plan["settings"] if row["name"] == name) for name in names
    ]
    initializations = {
        "opposite_signs": fresh_parameters(
            plan["confirmation_starts"], plan["norm"], plan["seed"] + 19000001
        ),
        "unconstrained_gaussian": initial_numeric(
            [plan["seed"] + 20000001],
            1,
            plan["confirmation_starts"],
            plan["norm"],
            torch.device("cpu"),
            codes=2,
        )
        .numpy()
        .reshape(-1, 5),
    }
    results = {}
    for family, initial in initializations.items():
        for setting in settings:
            name = (
                setting["name"]
                if family == "opposite_signs"
                else "gaussian_" + setting["name"]
            )
            results[name] = fit_setting(
                parent, plan, setting, initial, plan["seed"] + 19000001, device
            )
            np.savez_compressed(
                folder / ("confirmation_" + name + ".npz"), **results[name]
            )
            print("CONFIRMATION FIT", name, flush=True)
    save_json(
        folder / "confirmation_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "selected_policy": selected,
            "confirmation_methods": list(results),
            "test_evaluated": False,
            "fit_sha256": {
                name: digest(folder / ("confirmation_" + name + ".npz"))
                for name in results
            },
        },
    )
    summary = evaluate_confirmation(parent, results, device)
    save_json(folder / "confirmation_summary.json", summary)
    for name, row in summary.items():
        print(
            "CONFIRMATION",
            name,
            {key: value for key, value in row.items() if not key.endswith("metrics")},
            flush=True,
        )


def evaluate_confirmation(parent, results, device):
    heldout = load_archive(parent / "heldout.npz")
    _, all_x = input_table()
    all_y = labels_for(all_x, "0110")
    summary = {}
    with torch.no_grad():
        for name, result in results.items():
            indices = torch.tensor(result["indices"], dtype=torch.long, device=device)
            numerical = torch.tensor(result["parameters"], device=device)
            scores = (
                metrics(
                    predict(
                        torch.tensor(heldout["test_x"], device=device),
                        indices,
                        numerical,
                    ),
                    torch.tensor(heldout["test_y"][0], device=device)[None],
                )
                .cpu()
                .numpy()
            )
            population = (
                metrics(
                    predict(all_x.to(device), indices, numerical),
                    all_y.to(device)[None],
                )
                .cpu()
                .numpy()
            )
            selected_start = int(result["query_metrics"][:, 0].argmin())
            summary[name] = {
                "initialization_family": "unconstrained_gaussian"
                if name.startswith("gaussian_")
                else "opposite_signs",
                "mean_test_ba": float(scores[:, 1].mean()),
                "perfect_test_starts": int((scores[:, 1] >= 1 - 1e-6).sum()),
                "perfect_population_starts": int((population[:, 1] >= 1 - 1e-6).sum()),
                "starts": len(scores),
                "query_selected_test_ba": float(scores[selected_start, 1]),
                "test_metrics": scores.tolist(),
                "population_metrics": population.tolist(),
            }
    return summary


def initialize_bias(source, oracle, initial, policy):
    values = initial.copy()
    values[:, 2] = policy["bias"]
    if policy["center_output"]:
        with torch.no_grad():
            index = torch.tensor(
                np.broadcast_to(oracle, (len(values), 11, 8)).copy(), dtype=torch.long
            )
            values[:, 4] = (
                -predict(torch.tensor(source["support_x"]), index, torch.tensor(values))
                .mean(1)
                .numpy()
            )
    return values


def bias_calibrate(parent, device):
    folder = parent / "numeric_optimization_20261104"
    plan = json.loads((folder / "protocol.json").read_text())
    policies = [
        {"bias": bias, "center_output": center}
        for bias in (-0.5, 0.0, 0.5)
        for center in (False, True)
    ]
    save_json(
        folder / "bias_protocol.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "policies": policies,
            "starts_per_family": 32,
            "confirmation_starts_per_family": 64,
            "rules": [
                "The hidden bias b and output offset d already exist and remain trainable in every setting. Do not add model parameters.",
                "For each of two fresh initialization families, compare b0=-.5/0/+.5 and d0=0 or minus the mean initial support logit. Centering uses support inputs only, not labels or analytic code values.",
                "All policies use identical raw codebooks and support minibatches, lr.03, norm.5, L2.0001, and4096 updates. Checkpoints use query BCE.",
                "Choose the bias/offset policy by mean selection BCE across both families before evaluating its new confirmation starts or test.",
                "The existing two-branch policy is a separate initialization search, not an extra model parameter. Its discarded branch is charged to the4096-update budget.",
                "Prior task test results are known. This remains an exploratory numerical diagnosis on fixed Pattern0110 I.",
            ],
        },
    )
    source = load_archive(parent / "source.npz")
    oracle = load_archive(folder / "original_oracle.npz")["indices"][0]
    opposite = fresh_parameters(32, plan["norm"], plan["seed"] + 22000001)
    gaussian = (
        initial_numeric(
            [plan["seed"] + 23000001],
            1,
            32,
            plan["norm"],
            torch.device("cpu"),
            codes=2,
        )
        .numpy()
        .reshape(-1, 5)
    )
    raw = np.concatenate([opposite, gaussian])
    initial = [initialize_bias(source, oracle, raw, policy) for policy in policies]
    result = fit_setting(
        parent,
        plan,
        next(row for row in SETTINGS if row["name"] == "legacy"),
        np.concatenate(initial),
        plan["seed"] + 22000001,
        device,
    )
    np.savez_compressed(folder / "bias_calibration.npz", **result)
    ranking = []
    for number, policy in enumerate(policies):
        scores = result["selection_metrics"][number * 64 : (number + 1) * 64]
        row = {
            **policy,
            "selection_bce": float(scores[:, 0].mean()),
            "selection_ba": float(scores[:, 1].mean()),
            "selection_perfect": int((scores[:, 1] >= 1 - 1e-6).sum()),
            "opposite_perfect": int((scores[:32, 1] >= 1 - 1e-6).sum()),
            "gaussian_perfect": int((scores[32:, 1] >= 1 - 1e-6).sum()),
        }
        ranking.append(row)
        print("BIAS CALIBRATION", row, flush=True)
    ranking.sort(key=lambda row: row["selection_bce"])
    save_json(
        folder / "bias_selection.json",
        {
            "selection_utc": datetime.now(UTC).isoformat(),
            "selected": ranking[0],
            "ranking": ranking,
            "confirmation_test_evaluated": False,
        },
    )


def bias_confirm(parent, device):
    folder = parent / "numeric_optimization_20261104"
    plan = json.loads((folder / "protocol.json").read_text())
    policy = json.loads((folder / "bias_selection.json").read_text())["selected"]
    selected = json.loads((folder / "policy_selection.json").read_text())["selected"]
    source = load_archive(parent / "source.npz")
    oracle = load_archive(folder / "original_oracle.npz")["indices"][0]
    initializations = {
        "opposite_signs": fresh_parameters(64, plan["norm"], plan["seed"] + 24000001),
        "unconstrained_gaussian": initial_numeric(
            [plan["seed"] + 25000001],
            1,
            64,
            plan["norm"],
            torch.device("cpu"),
            codes=2,
        )
        .numpy()
        .reshape(-1, 5),
    }
    results = {}
    for family, initial in initializations.items():
        bias_initial = initialize_bias(source, oracle, initial, policy)
        for name, values, setting in (
            ("legacy", initial, SETTINGS[0]),
            ("selected_bias", bias_initial, SETTINGS[0]),
            (selected["name"], initial, selected),
        ):
            key = name if family == "opposite_signs" else "gaussian_" + name
            results[key] = fit_setting(
                parent, plan, setting, values, plan["seed"] + 24000001, device
            )
            np.savez_compressed(
                folder / ("bias_confirmation_" + key + ".npz"), **results[key]
            )
            print("BIAS CONFIRMATION FIT", key, flush=True)
    save_json(
        folder / "bias_confirmation_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "selected_bias_policy": policy,
            "selected_initialization_policy": selected,
            "test_evaluated": False,
            "fit_sha256": {
                name: digest(folder / ("bias_confirmation_" + name + ".npz"))
                for name in results
            },
        },
    )
    summary = evaluate_confirmation(parent, results, device)
    save_json(folder / "bias_confirmation_summary.json", summary)
    for name, row in summary.items():
        print(
            "BIAS CONFIRMATION",
            name,
            {key: value for key, value in row.items() if not key.endswith("metrics")},
            flush=True,
        )


def report(parent):
    folder = parent / "numeric_optimization_20261104"
    selection = json.loads((folder / "policy_selection.json").read_text())
    summary = json.loads((folder / "confirmation_summary.json").read_text())
    bias_selection = json.loads((folder / "bias_selection.json").read_text())
    bias_summary = json.loads((folder / "bias_confirmation_summary.json").read_text())
    old = load_archive(folder / "original_oracle.npz")
    reproduced = load_archive(folder / "reproduction_torch.npz")
    diagnosis = json.loads((folder / "original_diagnosis.json").read_text())
    polarity = old["initial_parameters"][:, 0] < 0
    (folder / "figures").mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for selected, label, color in (
        (polarity, "Initially C1<0<C2", "#4879b8"),
        (~polarity, "Initially C2<0<C1", "#d8784c"),
    ):
        axes[0, 0].plot(
            reproduced["history_steps"],
            reproduced["train_history"][:, selected].mean(1),
            color=color,
            label=label + ": support",
        )
        axes[0, 0].plot(
            reproduced["history_steps"],
            reproduced["query_history"][:, selected].mean(1),
            color=color,
            linestyle="--",
            label=label + ": query",
        )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Balanced BCE")
    axes[0, 0].set_xlabel("Numeric updates")
    axes[0, 0].legend(fontsize=8)
    for start in range(len(polarity)):
        trajectory = reproduced["parameter_history"][:, start, :2]
        axes[0, 1].plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="#4879b8" if polarity[start] else "#d8784c",
            alpha=0.65,
        )
    axes[0, 1].axhline(0, color="grey", linewidth=0.6)
    axes[0, 1].axvline(0, color="grey", linewidth=0.6)
    axes[0, 1].set_xlabel("C1")
    axes[0, 1].set_ylabel("C2")
    axes[0, 1].set_title("Standard Adam: actual codebook trajectories")
    dead = np.flatnonzero(np.array(diagnosis["active_relu_fraction"]) == 0)
    plotted = [0, 1, *dead.tolist()]
    for start in dict.fromkeys(plotted):
        gradient = np.linalg.norm(
            reproduced["data_gradient_history"][:, start, :3], axis=1
        )
        axes[1, 0].plot(
            reproduced["history_steps"],
            np.maximum(gradient, 1e-12),
            label=f"Original start {start}",
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Numeric updates")
    axes[1, 0].set_ylabel("Data gradient norm: C1/C2/bias")
    axes[1, 0].legend(fontsize=8)
    names = [name for name in summary if not name.startswith("gaussian_")]
    means = [100 * summary[name]["mean_test_ba"] for name in names]
    axes[1, 1].bar(names, means, color=["#d8784c", "#4879b8", "#65a58d"][: len(names)])
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_ylabel("Mean test BA over 64 new raw starts, %")
    for number, name in enumerate(names):
        axes[1, 1].text(
            number,
            means[number] + 1,
            f"{means[number]:.2f}%\n{summary[name]['perfect_test_starts']}/64 perfect",
            ha="center",
            fontsize=9,
        )
    for axis in axes.ravel():
        axis.grid(alpha=0.15)
    figure.suptitle(
        "Fixed true Pattern 0110 I: numerical optimization and data-driven initialization\nBoth warmup branches count toward the same 4096-update budget"
    )
    save_figure(figure, folder, "numeric_diagnosis")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    for centered, color in ((False, "#d8784c"), (True, "#4879b8")):
        settings = sorted(
            [
                row
                for row in bias_selection["ranking"]
                if row["center_output"] == centered
            ],
            key=lambda row: row["bias"],
        )
        axes[0, 0].plot(
            [row["bias"] for row in settings],
            [100 * row["selection_ba"] for row in settings],
            marker="o",
            color=color,
            label="Centered initial logits"
            if centered
            else "Initial output offset = 0",
        )
    axes[0, 0].set_xlabel("Initial shared hidden bias b")
    axes[0, 0].set_ylabel("Mean selection BA over 64 calibration starts, %")
    axes[0, 0].set_title("Bias and output offset: same five trainable parameters")
    axes[0, 0].legend(fontsize=8)
    names = ["legacy", "selected_bias", selection["selected"]["name"]]
    labels = ["Original init", "Selected bias + offset init", "Query choice: C or -C"]
    colors = ["#d8784c", "#4879b8", "#65a58d"]
    for axis, prefix, title in (
        (axes[0, 1], "", "64 new opposite-sign starts"),
        (axes[1, 0], "gaussian_", "64 new unconstrained Gaussian starts"),
    ):
        for name, label, color in zip(names, labels, colors, strict=True):
            fit = load_archive(folder / ("bias_confirmation_" + prefix + name + ".npz"))
            charged_steps = fit["history_steps"] + (
                int(fit["paired_initialization_steps"][0])
                if "paired_initialization_steps" in fit
                else 0
            )
            axis.plot(
                charged_steps,
                fit["query_history"].mean(1),
                color=color,
                label=label,
            )
            axis.plot(
                charged_steps,
                fit["train_history"].mean(1),
                color=color,
                linestyle="--",
                alpha=0.6,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Gradient evaluations, including discarded warmup branch")
        axis.set_ylabel("Mean BCE: query solid, support dashed")
        axis.set_title(title)
        axis.legend(fontsize=8)
    positions = np.arange(len(names))
    for offset, prefix, hatch, family in (
        (-0.19, "", "", "Opposite-sign starts"),
        (0.19, "gaussian_", "//", "Unconstrained Gaussian starts"),
    ):
        values = [100 * bias_summary[prefix + name]["mean_test_ba"] for name in names]
        axes[1, 1].bar(
            positions + offset,
            values,
            width=0.36,
            color=colors,
            hatch=hatch,
            alpha=0.85,
            label=family,
        )
        for number, value in enumerate(values):
            row = bias_summary[prefix + names[number]]
            axes[1, 1].text(
                number + offset,
                value + 0.8,
                f"{row['perfect_test_starts']}/64",
                ha="center",
                fontsize=8,
            )
    axes[1, 1].set_xticks(positions, ["Original", "Bias + offset", "C / -C warmup"])
    axes[1, 1].set_ylim(0, 109)
    axes[1, 1].set_ylabel("Mean test BA, %; labels = perfect starts")
    axes[1, 1].legend(fontsize=8, loc="lower right")
    for axis in axes.ravel():
        axis.grid(alpha=0.15)
    figure.suptitle(
        "Pattern 0110, true I fixed: bias initialization helps partly; selecting a codebook basin helps more\nEvery policy has the same 4096-gradient budget; all-start means, not best-of-64"
    )
    save_figure(figure, folder, "bias_comparison")
    rows = [
        [
            name,
            f"{100 * row['mean_test_ba']:.2f}",
            f"{row['perfect_test_starts']}/{row['starts']}",
            f"{row['perfect_population_starts']}/{row['starts']}",
            f"{100 * row['query_selected_test_ba']:.2f}",
        ]
        for name, row in summary.items()
    ]
    calibration_rows = [
        [
            row["name"],
            f"{row['selection_bce']:.4f}",
            f"{100 * row['selection_ba']:.2f}",
            f"{row['selection_perfect']}/32",
        ]
        for row in selection["ranking"]
    ]
    bias_rows = [
        [
            row["bias"],
            "centered" if row["center_output"] else "0",
            f"{row['selection_bce']:.4f}",
            f"{100 * row['selection_ba']:.2f}",
            f"{row['opposite_perfect']}/32",
            f"{row['gaussian_perfect']}/32",
        ]
        for row in sorted(
            bias_selection["ranking"],
            key=lambda row: (row["bias"], row["center_output"]),
        )
    ]
    bias_confirmation_rows = [
        [
            name,
            f"{100 * row['mean_test_ba']:.2f}",
            f"{row['perfect_test_starts']}/{row['starts']}",
            f"{row['perfect_population_starts']}/{row['starts']}",
            f"{100 * row['query_selected_test_ba']:.2f}",
        ]
        for name, row in bias_summary.items()
    ]
    good = int((old["test_metrics"][polarity, 1] >= 1 - 1e-6).sum())
    bad = int((old["test_metrics"][~polarity, 1] >= 1 - 1e-6).sum())
    audit = {
        "max_original_vs_torch_query_bce_difference": float(
            np.abs(old["query_metrics"][:, 0] - reproduced["query_metrics"][:, 0]).max()
        ),
        "original_perfect_starts": int(
            (old["selection_metrics"][:, 1] >= 1 - 1e-6).sum()
        ),
        "torch_perfect_starts": int(
            (reproduced["selection_metrics"][:, 1] >= 1 - 1e-6).sum()
        ),
        "original_dead_relu_starts": dead.tolist(),
    }
    save_json(folder / "optimizer_audit.json", audit)
    lines = [
        "# Pattern 0110: диагностика обучения чисел при истинном I",
        "Причина неудачных численных запусков при правильном I связана с инициализацией и областями сходимости ReLU-модели. Замена реализации Adam не исправила результат. Инициализация уже имеющихся сдвигов помогает частично; короткий выбор между кодбуками C и -C по query помогает сильнее.",
        "Модель всё время одна: h_j = ReLU(sum_i C[I_ij] x_i + b), logit = a sum_j h_j + d. C0=0 фиксирован; C1, C2, общий скрытый сдвиг b, коэффициент a и выходной сдвиг d обучаются. Всего пять численных параметров. Исходная инициализация b=0, a=1, d=0. Новые сдвиги не добавлялись.",
        f"В исходных 32 стартах все {good}/16 инициализаций C1<0<C2 дали 100%, из обратных знаков {bad}/16. Два начальных значения C переставлялись при неизменном I, поэтому менялась стартовая функция. Два старта полностью погасили скрытые ReLU; их градиент от данных по C1/C2/b равен нулю. L2 продолжает действовать, но не несёт информации о правильном направлении от данных. Знаки используются здесь для анализа сохранённых результатов, не для выбора новой инициализации.",
        f"Стандартный torch.optim.Adam с теми же начальными значениями, minibatch, L2 и ограничением нормы градиента каждой модели воспроизвёл {audit['torch_perfect_starts']}/32 точных запуска. Максимальное различие query BCE {audit['max_original_vs_torch_query_bce_difference']:.2g}. Исходный BatchedAdam добавляет 1e-12 под корень знаменателя; эта численная разница не объясняет провал. Полный batch также не устранил проблему; это контроль шума с большим числом примеров на update. Строгая локальная минимальность плохих точек не доказывалась.",
        "Исправление процедуры инициализации: для каждого сырого старта коротко обучить C и -C на support, сравнить query BCE, продолжить лучшую ветку вместе с её Adam moments. Аналитические значения и нужные знаки не подаются. 4096 численных gradient-evaluations на сырой старт включают обе ветки warmup: по 64 шага обеим, затем ещё 3968 выбранной. Архитектура всё время фиксирована, P не обучается.",
        table(
            [
                "Калибровка на32 новых стартах",
                "Mean selection BCE",
                "Mean selection BA, %",
                "Точных на selection",
            ],
            calibration_rows,
        ),
        f"До итогового контроля выбрана политика {selection['selected']['name']} по mean selection BCE. Затем64 новых старта прежнего семейства с противоположными знаками и64 независимых нормированных Gaussian-старта без ограничения знаков (в том числе одинаковые знаки). Checkpoint по query. Ни test, ни истинные численные значения не выбирают политику.",
        table(
            [
                "Политика",
                "Mean test BA, %",
                "100% на test",
                "100% на всех2048",
                "Query-выбор BA, %",
            ],
            rows,
        ),
        "![Диагностика численной оптимизации](figures/numeric_diagnosis.png)",
        "## Отдельная проверка предложения о сдвиге b",
        "На двух новых семействах по 32 сырых старта сравнили b0=-0.5/0/+0.5 и d0=0 либо d0=-mean_support(initial logit). Во втором случае начальные логиты центрируются около нуля; используются только входы support. Кодбуки и minibatch попарно одинаковые. Все пять параметров после инициализации свободно обучаются, фиксированного ограничения на b нет. Политику выбрали по среднему selection BCE двух семейств, затем проверили на ещё двух семействах по 64 новых старта.",
        table(
            [
                "b0",
                "d0",
                "Selection BCE",
                "Mean BA, %",
                "Точных opposite",
                "Точных Gaussian",
            ],
            bias_rows,
        ),
        f"Выбран b0={bias_selection['selected']['bias']}, center_output={bias_selection['selected']['center_output']}. Итоговый контроль ниже; gaussian_ означает произвольные начальные знаки, остальные строки — прежнее семейство с противоположными знаками. selected_bias меняет только начальные b/d; paired_64 меняет процедуру выбора начального кодбука при исходных b/d.",
        table(
            [
                "Политика",
                "Mean test BA, %",
                "100% на test",
                "100% на всех 2048",
                "Query-выбор BA, %",
            ],
            bias_confirmation_rows,
        ),
        "![Сдвиг и выбор начального кодбука](figures/bias_comparison.png)",
        "Выбор лучшего старта по query корректен: в этой серии он даёт 100% и в исходной процедуре. Доля успешных стартов описывает необходимый бюджет перезапусков. Даже C/-C не гарантирует сходимость с произвольного Gaussian-старта; безусловная устойчивость не установлена. Изменение начальных b/d само по себе не заменяет выбор области сходимости.",
        "Новая опция single.fit_fixed: specification['paired_initialization_steps']=64/256, значение 0 сохраняет старый алгоритм. Исторические результаты поиска структуры не перезаписывались; опция включена только в новых численных контролях. Проверка выполнена для фиксированного истинного 0110-I, который совпадает с найденным joint-поиском после выравнивания. Влияние на дальнейший soft-поиск P не измерялось. Программные тесты не создавались и не запускались; выполнены сами контролируемые обучения.",
    ]
    (folder / "REPORT.md").write_text("\n\n".join(lines) + "\n")
    snapshot = folder / "analysis_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ("numeric_optimization.py", "single.py", "model.py", "data.py"):
        shutil.copy2(BASE / name, snapshot / name)
    save_json(
        folder / "complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "selected_policy": selection["selected"],
            "confirmation": summary,
            "bias_policy": bias_selection["selected"],
            "bias_confirmation": bias_summary,
            "optimizer_audit": audit,
            "software_tests_run": False,
            "artifacts": {
                str(path.relative_to(folder)): digest(path)
                for path in sorted(folder.rglob("*"))
                if path.is_file() and path.name != "complete.json"
            },
        },
    )
    print(
        table(
            ["Policy", "Mean BA", "Test perfect", "Population perfect", "Selected BA"],
            rows,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "reproduce",
            "calibrate",
            "confirm",
            "bias_calibrate",
            "bias_confirm",
            "report",
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/single_0110_20261104"
    )
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage == "reproduce":
        reproduce(args.output, torch.device(args.device))
    elif args.stage == "calibrate":
        calibrate(args.output, torch.device(args.device))
    elif args.stage == "confirm":
        confirm(args.output, torch.device(args.device))
    elif args.stage == "bias_calibrate":
        bias_calibrate(args.output, torch.device(args.device))
    elif args.stage == "bias_confirm":
        bias_confirm(args.output, torch.device(args.device))
    else:
        report(args.output)


if __name__ == "__main__":
    main()
