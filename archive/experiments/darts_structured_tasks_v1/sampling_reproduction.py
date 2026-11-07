"""Reproduce the successful DARTS protocol before changing code selection."""

import argparse
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import BASE, digest, save_json
from .model import (
    BatchedOptimizer,
    clip_gradient,
    initial_numeric,
    predict,
    score,
    signed_values,
    soft_weights,
    source_tensors,
)
from .run import refit_search
from .sharing_ablation import fresh
from .sharing_report import evaluate
from .train import search as soft_search

METHODS = {"soft": 1, "hard_argmax": 1, "gumbel_1": 1, "gumbel_4": 4}


def prepare(parent, root):
    root.mkdir(parents=True, exist_ok=False)
    choice = json.loads((parent / "architecture_selection.json").read_text())["primary"]
    for method, samples in METHODS.items():
        output = root / method
        output.mkdir()
        plan = json.loads((parent / "protocol.json").read_text())
        spec = json.loads((parent / "specification.json").read_text())
        spec["name"] = f"sampling_reproduction_{method}"
        plan["sharing_heldout_offset"] = 7600000
        row = {
            key: choice[key]
            for key in (
                "configuration",
                "numeric_lr",
                "architecture_lr",
                "radius",
                "sparsity",
                "mode",
            )
        }
        row.update(family=method, commitment=0.0, samples=samples)
        for name in ("source.npz", "teacher_audit_only.npz"):
            shutil.copy2(parent / name, output / name)
        save_json(output / "protocol.json", plan)
        save_json(output / "specification.json", spec)
        snapshot = output / "code_snapshot"
        snapshot.mkdir()
        for name in (
            "sampling_reproduction.py",
            "train.py",
            "model.py",
            "data.py",
            "run.py",
            "sharing_report.py",
            "sharing_ablation.py",
            "evaluate.py",
        ):
            shutil.copy2(BASE / name, snapshot / name)
        save_json(
            output / "registration.json",
            {
                "registered_utc": datetime.now(UTC).isoformat(),
                "method": method,
                "configuration": row,
                "rules": [
                    "Reproduction on original seeds43/44/45, not new confirmation tasks or seeds.",
                    "Preserve original numeric_lr.01, architecture_lr.1, four simultaneous codebooks,512 numeric soft warmup steps, temperature1 to.03, sparsity.03, late entropy.02.",
                    "Soft control calls the original unmodified train.search. All methods use the original soft-matrix normalization of the same codebook initialization.",
                    "Hard variants retain the original probabilities softmax(alpha / temperature) as the sampling law; unlike the previous sampling experiment, policy temperature also cools.",
                    "Hard Gumbel uses one/four independent complete indexators, each shared across the minibatch and all four numerical replicas. No soft mixture is used after warmup.",
                    "Hard argmax uses the original soft probabilities as STE surrogate. Gumbel uses softmax(log(P)+Gumbel) as surrogate, with no extra temperature parameter.",
                    "All hard variants store raw MAP one-hot for refit; no mass transfer changes the sampling law. The soft control keeps its original extraction implementation.",
                    "Support optimization only. Query selects numeric checkpoints; selection chooses snapshots after3000 C-fit steps. No settings are retuned.",
                    "Final fresh-C controls and the same new heldout input draw follow source freeze. One previously inspected teacher.",
                    "No software tests.",
                ],
                "source_sha256": digest(output / "source.npz"),
                "snapshot_sha256": {p.name: digest(p) for p in snapshot.glob("*.py")},
            },
        )


def hard_search(output, plan, spec, rows, method, device):
    folder = output / "confirmation"
    folder.mkdir()
    save_json(folder / "rows.json", rows)
    settings = plan["search"]
    codes, classes = spec["codes"], 2 * spec["codes"] + 1
    arrays = []
    for row in rows:
        rng = np.random.default_rng(plan["seed"] + 100003 * row["seed"])
        item = rng.normal(
            0, settings["initial_logit_std"], (spec["inputs"], spec["hidden"], classes)
        )
        arrays.append(item)
    alpha = torch.tensor(np.array(arrays), dtype=torch.float32, device=device)
    with torch.no_grad():
        probability = settings["zero_probability"]
        alpha[..., 0] += math.log(2 * codes * probability / (1 - probability))
    alpha.requires_grad_()
    numeric = initial_numeric(
        rows,
        settings["restarts"],
        spec,
        plan["seed"] + 901,
        device,
        probabilities=alpha.detach().softmax(-1),
    ).requires_grad_()
    numeric_optimizer = BatchedOptimizer(numeric, rows)
    alpha_optimizer = BatchedOptimizer(alpha, rows, "architecture_lr")
    batch_generator = torch.Generator(device=device).manual_seed(plan["seed"] + 1001)
    sample_generator = torch.Generator(device=device).manual_seed(plan["seed"] + 830001)
    diagnostic_generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + 840001
    )
    data = source_tensors(output, device)
    variance = spec["target_variance"]
    replicas, samples = settings["restarts"], METHODS[method]
    started = time.perf_counter()
    for _ in range(settings["numeric_warmup"]):
        ids = torch.randint(
            len(data["support_x"]),
            (settings["batch"],),
            generator=batch_generator,
            device=device,
        )
        predicted = predict(
            data["support_x"][ids],
            soft_weights(alpha.detach().softmax(-1), numeric, codes),
            numeric,
            spec,
        )
        loss = (predicted - data["support_y"][ids]).square().mean(-1) / variance
        objective = loss + settings["numeric_l2"] * numeric.square().sum(-1) / 2
        gradient = torch.autograd.grad(objective.sum(), numeric)[0]
        numeric_optimizer.step(
            clip_gradient(gradient, settings["gradient_clip"], (-1,))
        )
    steps = settings["confirmation_steps"]
    checkpoints = {round(fraction * steps) for fraction in settings["snapshots"]}
    history, saved = [], []
    for step in range(steps + 1):
        progress = step / max(steps, 1)
        cooling = min(1.0, max(0.0, (progress - 0.3) / 0.7))
        temperature = settings["minimum_temperature"] ** cooling
        probabilities = (alpha / temperature).softmax(-1)
        log_policy = probabilities.clamp_min(1e-30).log()
        values = signed_values(numeric, codes)
        entropy = (
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean((1, 2))
        )
        if step % settings["checkpoint_every"] == 0 or step == steps:
            with torch.no_grad():
                map_assignment = F.one_hot(probabilities.argmax(-1), classes).float()
                weights = torch.einsum("sdhk,srk->srdh", map_assignment, values)
                hard_query = score(
                    data["query_x"], data["query_y"], weights, numeric, spec
                )
                if method == "hard_argmax":
                    query = hard_query
                    training = score(
                        data["support_x"][:512],
                        data["support_y"][:512],
                        weights,
                        numeric,
                        spec,
                    )
                else:
                    uniform = torch.rand(
                        (len(rows), 8, spec["inputs"], spec["hidden"], classes),
                        generator=diagnostic_generator,
                        device=device,
                    ).clamp_(1e-7, 1 - 1e-7)
                    noise = -(-uniform.log()).log()
                    assignments = F.one_hot(
                        (log_policy[:, None] + noise).argmax(-1), classes
                    ).float()
                    sampled_weights = torch.einsum(
                        "smdhk,srk->srmdh", assignments, values
                    ).flatten(1, 2)
                    expanded = numeric[:, :, None].expand(-1, -1, 8, -1).flatten(1, 2)
                    query = (
                        score(
                            data["query_x"],
                            data["query_y"],
                            sampled_weights,
                            expanded,
                            spec,
                        )
                        .reshape(len(rows), replicas, 8)
                        .mean(-1)
                    )
                    training = (
                        score(
                            data["support_x"][:512],
                            data["support_y"][:512],
                            sampled_weights,
                            expanded,
                            spec,
                        )
                        .reshape(len(rows), replicas, 8)
                        .mean(-1)
                    )
                history.append(
                    {
                        "step": step,
                        "query_mse": query.tolist(),
                        "hard_query_mse": hard_query.tolist(),
                        "train_mse": training.tolist(),
                        "entropy": entropy.tolist(),
                        "MAP_edges": (map_assignment[..., 0] == 0).sum((1, 2)).tolist(),
                        "seconds": time.perf_counter() - started,
                    }
                )
                save_json(folder / "history.json", history)
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite matched sampling reproduction")
                if step in checkpoints:
                    saved.append(
                        {
                            "step": step,
                            "probabilities": map_assignment.cpu().numpy(),
                            "policy_probabilities": probabilities.cpu().numpy(),
                            "numeric": numeric.cpu().numpy(),
                            "logits": alpha.cpu().numpy(),
                        }
                    )
                    np.savez_compressed(
                        folder / "snapshots.npz",
                        steps=np.array([item["step"] for item in saved]),
                        **{
                            key: np.stack([item[key] for item in saved])
                            for key in saved[0]
                            if key != "step"
                        },
                    )
                if step % 1000 == 0 or step == steps:
                    print(
                        f"{method} step={step} query={query.mean().item() / variance:.6g} MAP={hard_query.mean().item() / variance:.6g} seconds={time.perf_counter() - started:.1f}",
                        flush=True,
                    )
        if step == steps:
            break
        if method == "hard_argmax":
            relaxed = probabilities[:, None]
            hard = F.one_hot(probabilities.argmax(-1), classes).float()[:, None]
        else:
            uniform = torch.rand(
                (len(rows), 4, spec["inputs"], spec["hidden"], classes),
                generator=sample_generator,
                device=device,
            ).clamp_(1e-7, 1 - 1e-7)
            noise = -(-uniform[:, :samples].log()).log()
            perturbed = log_policy[:, None] + noise
            relaxed = perturbed.softmax(-1)
            hard = F.one_hot(perturbed.argmax(-1), classes).float()
        assignment = hard + (relaxed - relaxed.detach())
        weights = torch.einsum("smdhk,srk->srmdh", assignment, values).flatten(1, 2)
        ids = torch.randint(
            len(data["support_x"]),
            (settings["batch"],),
            generator=batch_generator,
            device=device,
        )
        expanded = numeric[:, :, None].expand(-1, -1, samples, -1).flatten(1, 2)
        predicted = predict(data["support_x"][ids], weights, expanded, spec)
        losses = (predicted - data["support_y"][ids]).square().mean(-1).reshape(
            len(rows), replicas, samples
        ).mean(-1) / variance
        regularized = losses + settings["numeric_l2"] * numeric.square().sum(-1) / 2
        numeric_gradient = torch.autograd.grad(
            regularized.sum(), numeric, retain_graph=True
        )[0]
        objective = losses.mean(-1) + rows[0]["sparsity"] * (
            1 - probabilities[..., 0]
        ).mean((1, 2))
        objective = (
            objective + settings["late_entropy"] * max(0, 2 * progress - 1) * entropy
        )
        alpha_gradient = torch.autograd.grad(objective.sum(), alpha)[0]
        numeric_optimizer.step(
            clip_gradient(numeric_gradient, settings["gradient_clip"], (-1,))
        )
        alpha_optimizer.step(
            clip_gradient(alpha_gradient, settings["alpha_clip"], (1, 2, 3))
        )
        with torch.no_grad():
            alpha.sub_(alpha.mean(-1, keepdim=True))
    save_json(
        folder / "complete.json",
        {
            "seconds": time.perf_counter() - started,
            "steps": steps,
            "roots": len(rows),
            "numeric_restarts": replicas,
            "samples": samples,
        },
    )


def run(root, device):
    for method in METHODS:
        output = root / method
        plan = json.loads((output / "protocol.json").read_text())
        spec = json.loads((output / "specification.json").read_text())
        config = json.loads((output / "registration.json").read_text())["configuration"]
        rows = [
            {**config, "seed": seed} for seed in plan["search"]["confirmation_seeds"]
        ]
        if method == "soft":
            soft_search(output, plan, spec, rows, "confirmation", device)
        else:
            hard_search(output, plan, spec, rows, method, device)
        refit_search(output, plan, spec, "confirmation", device)
        choice = json.loads((output / "architecture_selection.json").read_text())
        save_json(
            output / "family_selection.json",
            {"families": {method: choice["ranking"][0]}},
        )
        save_json(
            output / "source_frozen.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "sha256": {
                    name: digest(output / name)
                    for name in (
                        "source.npz",
                        "family_selection.json",
                        "architecture_selection.json",
                        "confirmation/hard_fits.npz",
                    )
                },
            },
        )
    for method in METHODS:
        fresh(root / method, device)


def report(root, reference):
    import matplotlib.pyplot as plt

    from .report import categorical, save_figure

    summaries = {}
    for method in METHODS:
        output = root / method
        evaluate(output)
        summaries[method] = json.loads((output / "summary.json").read_text())[
            "families"
        ][method]
    with np.load(reference / "confirmation/snapshots.npz") as archive:
        old = dict(archive)
    with np.load(root / "soft/confirmation/snapshots.npz") as archive:
        reproduction = dict(archive)
    agreement = {
        key: float(np.max(np.abs(old[key][:, :3] - reproduction[key])))
        for key in ("probabilities", "numeric")
    }
    matched_initial_states = {}
    for method in METHODS:
        if method == "soft":
            continue
        with np.load(root / method / "confirmation/snapshots.npz") as archive:
            matched_initial_states[method] = {
                "numeric_max_difference": float(
                    np.abs(reproduction["numeric"][0] - archive["numeric"][0]).max()
                ),
                "policy_max_difference": float(
                    np.abs(
                        reproduction["probabilities"][0]
                        - archive["policy_probabilities"][0]
                    ).max()
                ),
            }
    diagnostic = json.loads((root / "edge_drop_diagnostic.json").read_text())
    save_json(
        root / "summary.json",
        {
            "methods": summaries,
            "soft_reproduction_max_differences": agreement,
            "matched_initial_states": matched_initial_states,
            "edge_drop_diagnostic": diagnostic,
        },
    )
    figure, axes = plt.subplots(1, 5, figsize=(19, 4.3), constrained_layout=True)
    with np.load(root / "soft/aligned_states.npz") as archive:
        truth, order = archive["true_indices"], archive["row_order"]
    image = categorical(axes[0], truth[order], 4, "Истинный I")
    for axis, method in zip(axes[1:], METHODS):
        result = summaries[method]
        number = result["primary"]["row"]
        with np.load(root / method / "aligned_states.npz") as archive:
            matrix = archive[f"row_{number}_indices"]
        image = categorical(
            axis,
            matrix[order],
            4,
            f"{method}\nточных масок {result['exact_masks']}/3; I {result['exact_indexators']}/3",
        )
    figure.colorbar(image, ax=axes, ticks=range(-4, 5), shrink=0.7)
    figure.suptitle("Полная матрица: прежний протокол DARTS, исходные seed43–45")
    save_figure(figure, root / "indexators")
    text = [
        "# Контроль: прежний DARTS и жёсткое сэмплирование",
        "",
        "Исходные seed43/44/45 и прежние настройки. Это воспроизведение, не новая независимая проверка. Четыре численные реплики и512 soft numerical warmup шагов; alpha lr0.1, C lr0.01, температура политики1→0.03, sparsity0.03. В hard-ветках forward дискретный после warmup; вероятности сохраняются. Gumbel surrogate имеет температуру1 без дополнительного охлаждения.",
        "",
        f"Максимальные отличия soft-контроля от исходных snapshot: {agreement}.",
        "",
        "| Метод | Средняя selection NMSE | Точных масок/3 | Точных I/3 | Test NMSE выбранного I |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method, result in summaries.items():
        text.append(
            f"| {method} | {result['selection_nmse_mean']:.6g} | {result['exact_masks']}/3 | {result['exact_indexators']}/3 | {result['primary']['test_nmse']:.6g} |"
        )
    text.extend(
        [
            "",
            "Мягкий контроль использует исходную функцию train.search без изменений, включая исходный перенос массы при извлечении I. Hard-ветки сэмплируют исходную политику без переноса массы и фиксируют её MAP. Настройки не подбирались повторно; каждый I получает3000 C-fit шагов с четырёх исходных численных состояний. Учитель и новые heldout-входы открыты после source/fresh выбора. Полные данные и32 свежих C сохранены по каждому методу.",
            "",
            f"Совпадение начальных состояний после warmup между новыми методами: {matched_initial_states}. Старый архив и новый soft имеют небольшое различие C уже после warmup; их дальнейшие траектории не побитово идентичны. Здесь сравнивается повтор протокола, а не утверждается точное воспроизведение прежнего состояния.",
            "",
            "## Диагностика плохой маски предыдущего hard-sampling запуска",
            "",
            f"Отдельный source-only анализ прежнего Gumbel4 full, seed{diagnostic['seed']}, шаг{diagnostic['step']}: MAP имеет {diagnostic['active_edges']} активных связей, train NMSE {diagnostic['train_nmse']:.6g}, query NMSE {diagnostic['query_nmse']:.6g}. Каждая активная связь по очереди отключалась при фиксированных C и выходной голове; обучение не проводилось.",
            "",
            f"Из {diagnostic['drops_improving_query']} удалений, улучшающих query, {diagnostic['drops_improving_query_but_hurting_train']} ухудшают support. Лишь {diagnostic['drops_improving_train']} удаления улучшают support. Следовательно, support-only архитектурный градиент на этом состоянии удерживает связи, вредные для query. Это локальное свидетельство архитектурного переобучения, не исчерпывающее объяснение всей траектории.",
            "",
            f"У {diagnostic['linear_sign_disagreements']} из {diagnostic['active_edges']} удалений знак локальной линейной оценки −W·grad(L) отличается от знака фактического изменения support loss. Это диагностика конечного перехода в ноль, а не измерение фактического stochastic STE-gradient. Медианная вероятность нуля на активных связях {diagnostic['median_zero_probability_on_active']:.6g}; исправляющие назначения могут редко сэмплироваться.",
            "",
            "Диагностика относится к одному сохранённому MAP checkpoint до C-refit. Она не оценивает точный expected-loss градиент, совместные удаления или повторное обучение C после изменения I. Численные дельты, градиенты и P сохранены в edge_drop_diagnostic.npz.",
            "",
            "![Индексаторы](indexators.png)",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(text))
    save_json(
        root / "complete.json",
        {
            "sha256": {
                str(p.relative_to(root)): digest(p)
                for p in root.rglob("*")
                if p.is_file() and p != root / "complete.json"
            }
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "run", "report"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare(args.parent, args.root)
    elif args.stage == "run":
        run(args.root, torch.device("cuda:0"))
    else:
        (args.root / "reporting_code_snapshot.py").write_bytes(Path(__file__).read_bytes())
        report(args.root, args.parent)


if __name__ == "__main__":
    main()
