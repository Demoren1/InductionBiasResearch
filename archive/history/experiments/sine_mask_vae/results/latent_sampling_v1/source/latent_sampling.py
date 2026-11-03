"""Adaptive sampling of z with a frozen VAE and paired SGD budgets."""

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .core import population_loss, save_json, task
from .latent_search import (
    LatentSearchOptions,
    examples,
    inner_sgd,
    load_decoder,
    predictions,
)
from .models import hard_mask
from .settings import DIRECTORY, digest, families


SAMPLING = {
    "initial_radius": 1.0,
    "final_radius": 0.25,
    "prior_probability": 0.25,
    "accept_equal": True,
}


@torch.no_grad()
def sample_search(model, spec, seed, trials, options, device):
    started = time.perf_counter()
    data_rng = torch.Generator(device=device).manual_seed(seed + 700001)
    x, y = examples(
        spec, options.inner_steps * options.batch, data_rng, device
    )
    batches = x.reshape(options.inner_steps, options.batch, 8)
    targets = y.reshape(options.inner_steps, options.batch, 2)
    query, query_y = examples(
        spec, options.query_examples, data_rng, device
    )
    selection_x, selection_y = examples(
        spec, options.query_examples, data_rng, device
    )
    count = trials * options.restarts
    weight_rng = torch.Generator(device=device).manual_seed(seed)
    initial = options.initial_scale * torch.randn(
        count, 8, 2, generator=weight_rng, device=device
    )
    latent_rng = torch.Generator(device=device).manual_seed(seed + 3)
    current = torch.randn(
        count, 16, generator=latent_rng, device=device
    )
    initial_latent = current.clone()
    proposal_rng = torch.Generator(device=device).manual_seed(seed + 10103)
    current_query = torch.full((count,), float("inf"), device=device)
    best_empirical = torch.full((trials,), float("inf"), device=device)
    best_population = best_empirical.clone()
    best_mask = torch.zeros(trials, 8, 2, device=device)
    oracle = torch.zeros(8, 2, device=device)
    oracle[spec["rows"], torch.arange(2, device=device)] = 1
    history, masks_saved, losses_saved = [], [], []

    def retain(mask, weights, amplitude):
        nonlocal best_empirical, best_population, best_mask
        candidates = len(mask) // trials
        empirical = (
            predictions(selection_x, weights, amplitude) - selection_y
        ).square().mean((1, 2)).reshape(trials, candidates)
        population = population_loss(weights, amplitude, spec).reshape(
            trials, candidates
        )
        chosen = empirical.argmin(1)
        rows = torch.arange(trials, device=device)
        values = empirical[rows, chosen]
        improved = values < best_empirical
        best_empirical = torch.minimum(best_empirical, values)
        best_population = torch.where(
            improved, population[rows, chosen], best_population
        )
        selected_mask = mask.reshape(trials, candidates, 8, 2)[rows, chosen]
        best_mask = torch.where(
            improved[:, None, None], selected_mask, best_mask
        )
        masks_saved.append(mask.reshape(trials, candidates, 8, 2).cpu().numpy())
        losses_saved.append(population.cpu().numpy())

    for step in range(options.rounds):
        radius = SAMPLING["initial_radius"] * (
            SAMPLING["final_radius"] / SAMPLING["initial_radius"]
        ) ** (max(0, step - 1) / max(1, options.rounds - 2))
        proposal = current
        if step:
            proposal = current + radius * torch.randn(
                count, 16, generator=proposal_rng, device=device
            )
            refresh = torch.rand(
                count, generator=proposal_rng, device=device
            ) < SAMPLING["prior_probability"]
            prior = torch.randn(
                count, 16, generator=proposal_rng, device=device
            )
            proposal = torch.where(refresh[:, None], prior, proposal)
        mask = hard_mask(model.decode(proposal))
        train, query_loss, weights, amplitude = inner_sgd(
            mask, initial, batches, targets, query, query_y, options
        )
        if not torch.isfinite(query_loss).all():
            raise RuntimeError("Non-finite sampling objective")
        accepted = query_loss <= current_query
        current = torch.where(accepted[:, None], proposal, current)
        current_query = torch.minimum(current_query, query_loss)
        retain(mask, weights, amplitude)
        history.append({
            "round": step,
            "network_fits": (step + 1) * options.restarts,
            "radius": radius,
            "accepted_fraction": float(accepted.float().mean()),
            "train_mse": float(train.mean()),
            "proposal_query_mse": float(query_loss.mean()),
            "incumbent_query_mse": float(current_query.mean()),
            "selected_population_mse": float(best_population.mean()),
            "selected_recall_at_2": float(
                (best_mask * oracle).sum((-1, -2)).mean() / 2
            ),
        })
    final_mask = best_mask[:, None].expand(
        -1, options.final_restarts, -1, -1
    ).reshape(-1, 8, 2)
    final_rng = torch.Generator(device=device).manual_seed(seed + 90001)
    final_initial = options.initial_scale * torch.randn(
        len(final_mask), 8, 2, generator=final_rng, device=device
    )
    train, query_loss, weights, amplitude = inner_sgd(
        final_mask, final_initial, batches, targets, query, query_y, options
    )
    retain(final_mask, weights, amplitude)
    history.append({
        "round": options.rounds,
        "network_fits": options.fits_per_search,
        "train_mse": float(train.mean()),
        "proposal_query_mse": float(query_loss.mean()),
        "incumbent_query_mse": float(current_query.mean()),
        "selected_population_mse": float(best_population.mean()),
        "selected_recall_at_2": float(
            (best_mask * oracle).sum((-1, -2)).mean() / 2
        ),
    })
    result = {
        "method": "adaptive_sampling",
        "searches": trials,
        "success": int((best_population < 0.1).sum()),
        "mean_selected_population_mse": float(best_population.mean()),
        "mean_selected_recall_at_2": float(
            (best_mask * oracle).sum((-1, -2)).mean() / 2
        ),
        "seconds": time.perf_counter() - started,
        "fits_per_search": options.fits_per_search,
        "training_examples_per_search": (
            options.fits_per_search * options.inner_steps * options.batch
        ),
        "vae_has_gradients": any(
            p.grad is not None for p in model.parameters()
        ),
        "latent_requires_grad": current.requires_grad,
        "history": history,
    }
    arrays = {
        "selected_population": best_population.cpu().numpy(),
        "selected_empirical": best_empirical.cpu().numpy(),
        "selected_masks": best_mask.cpu().numpy(),
        "candidate_masks": np.concatenate(masks_saved, axis=1),
        "candidate_population": np.concatenate(losses_saved, axis=1),
        "initial_latent": initial_latent.cpu().numpy(),
        "final_latent": current.cpu().numpy(),
    }
    return result, arrays


def publish(run, baseline):
    if not (baseline / "TEST_COMPLETE.json").exists():
        print("Sampling complete; waiting for baseline evaluation to finish.")
        return
    records = [json.loads(p.read_text()) for p in (run / "test").glob("*.json")]
    controls = [
        json.loads(p.read_text()) for p in (baseline / "test").glob("*.json")
    ]
    methods = {
        "adaptive_sampling": "Adaptive sampling + refit",
        "random_z": "Independent sampling + refit",
        "prior_sampling": "Prior sampling, 64 starts",
        "gradient_z": "Gradient z + refit",
    }
    groups = {
        method: [r for r in records + controls if r["method"] == method]
        for method in methods
    }
    summary = {}
    for method, group in groups.items():
        searches = sum(r["searches"] for r in group)
        summary[method] = {
            "success": sum(r["success"] for r in group),
            "searches": searches,
            "rate": sum(r["success"] for r in group) / searches,
            "population_mse": float(np.mean([
                r["mean_selected_population_mse"] for r in group
            ])),
            "recall_at_2": float(np.mean([
                r["mean_selected_recall_at_2"] for r in group
            ])),
            "seconds": sum(r["seconds"] for r in group),
        }
    destination = DIRECTORY / "results" / run.name
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "summary.json", summary)
    save_json(destination / "records.json", records)
    shutil.copy2(run / "plan.json", destination / "plan.json")
    source = destination / "source"
    source.mkdir(exist_ok=True)
    for path in (run / "source").glob("*.py"):
        shutil.copy2(path, source / path.name)
    os.environ.setdefault("MPLCONFIGDIR", str(run / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    histories = [r["history"][:-1] for r in records]
    x = [h["network_fits"] for h in histories[0]]
    for key, label in (
        ("train_mse", "Train MSE of proposals"),
        ("proposal_query_mse", "Query MSE of proposals"),
        ("incumbent_query_mse", "Query MSE of retained codes"),
    ):
        mean = np.mean([[h[key] for h in history] for history in histories], axis=0)
        axes[0].plot(x, mean, "o-", label=label, markersize=3)
    axes[0].set_title("Adaptive sampling loss")
    axes[0].set_xlabel("Exploration fits per search")
    axes[0].legend(fontsize=8)
    for method in list(methods)[:3]:
        histories = [r["history"] for r in groups[method]]
        x = [h["network_fits"] for h in histories[0]]
        mean = np.mean([
            [h["selected_recall_at_2"] for h in history]
            for history in histories
        ], axis=0)
        axes[1].plot(x, mean, "o-", label=methods[method], markersize=3)
    axes[1].set_title("Selected mask Recall@2 (diagnostic)")
    axes[1].set_xlabel("Network fits per search")
    axes[1].set_ylim(0, 1.03)
    axes[1].legend(fontsize=8)
    rates = [100 * summary[m]["rate"] for m in methods]
    bars = axes[2].bar(
        ["Adaptive", "Random\n+ refit", "Prior", "Gradient"], rates,
        color=["#147d92", "#d18d28", "#7161a8", "#aaa"],
    )
    axes[2].bar_label(bars, labels=[f"{r:.1f}%" for r in rates], padding=3)
    axes[2].set_ylim(0, max(rates) * 1.2)
    axes[2].set_title("Successful searches; equal SGD samples")
    axes[2].set_ylabel("Population MSE < 0.1 (%)")
    for axis in axes[:2]:
        axis.grid(alpha=0.2)
    figure.savefig(destination / "curves.png", dpi=160)
    plt.close(figure)
    rows = []
    for method, name in methods.items():
        value = summary[method]
        rows.append(
            f"| {name} | {value['success']}/{value['searches']} | "
            f"{100 * value['rate']:.2f}% | {value['population_mse']:.5f} | "
            f"{value['recall_at_2']:.3f} |"
        )
    report = """# Поиск z адаптивным сэмплированием

VAE заморожен. Четыре кода из N(0, I), восемь раундов. После первого
раунда каждый рестарт предлагает z + sigma * epsilon, epsilon ~ N(0, I).
Радиус геометрически убывает от 1.0 до 0.25. С вероятностью 0.25 вместо
локального предложения берётся новый сэмпл prior. Код принимается при
невозрастании query MSE после 1024 шагов SGD весов; равенство допускает
движение внутри области одной маски. Градиентов по z или VAE нет.

Всего 32 обучения кандидатов, затем 32 независимых старта весов на лучшей
маске, выбранной отдельной эмпирической selection-выборкой. Бюджет всех
методов — 64 × 1024 × 64 = 4 194 304 обработанных обучающих примера.
Данные, начальные четыре кода, четыре инициализации весов и финальные
32 инициализации совпадают с парным random_z-контролем. Популяционная MSE
и правильная маска используются только для измерений.

Проверены новые функции 301/302, четыре прежние структурные test-пары,
три сохранённых VAE seed и 16 поисков на контекст. Гиперпараметры заданы
до оценки; отдельная validation-диагностика сохранена. Контроли переиспользованы
из завершённого latent_search_v1 без повторного обучения. Это не новый
слепой структурный holdout. Временные затраты параллельных запусков могут
зависеть от загрузки сервера; равенство FLOPs не заявляется.

| Метод | Успехи | Доля | MSE выбранного | Recall@2 |
|---|---:|---:|---:|---:|
""" + "\n".join(rows) + """

![Loss, Recall@2 и успех поиска](curves.png)

Настройки и хеши — plan.json, исходники — source/, подробные контексты —
records.json, локальные коды и маски — runs/latent_sampling_v1.
Контроли подробно описаны в ../latent_search_v1/REPORT.md.
"""
    (destination / "REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Report: {destination / 'REPORT.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "report"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline", type=Path, default=DIRECTORY / "runs/latent_search_v1")
    parser.add_argument("--run-dir", type=Path, default=DIRECTORY / "runs/latent_sampling_v1")
    args = parser.parse_args()
    baseline_plan = json.loads((args.baseline / "plan.json").read_text())
    config = baseline_plan["config"]
    options = LatentSearchOptions(**config["search"])
    run = args.run_dir.resolve()
    if run == args.baseline.resolve() or run == Path(baseline_plan["reference"]):
        parser.error("Use a separate run directory")
    if args.phase == "report":
        publish(run, args.baseline)
        return
    torch.set_num_threads(1)
    run.mkdir(parents=True, exist_ok=True)
    hashes = {
        name: digest(DIRECTORY / name)
        for name in ("latent_sampling.py", "latent_search.py", "core.py", "models.py")
    }
    plan = {
        "baseline_plan": baseline_plan,
        "sampling": SAMPLING,
        "implementation_sha256": hashes,
        "device": args.device,
    }
    path = run / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError("Changed sampling protocol; use a new run directory")
    save_json(path, plan)
    source = run / "source"
    source.mkdir(exist_ok=True)
    for name in hashes:
        shutil.copy2(DIRECTORY / name, source / name)
    for split in ("validation", "test"):
        destination = run / split
        destination.mkdir(exist_ok=True)
        functions = (
            [config["validation_function"]] if split == "validation"
            else config["evaluation_functions"]
        )
        trials = config[
            "validation_trials" if split == "validation" else "evaluation_trials"
        ]
        for model_seed in config["model_seeds"]:
            model, _ = load_decoder(
                Path(baseline_plan["reference"]), config["reference_variant"],
                model_seed, args.device,
            )
            for function in functions:
                for a, b in families(split):
                    name = f"adaptive_a{a}_b{b}_f{function}_s{model_seed}"
                    path = destination / f"{name}.json"
                    if path.exists():
                        continue
                    seed = (
                        480100 + function * 10007 + a * 1009
                        + b * 17 + model_seed * 31
                    )
                    result, arrays = sample_search(
                        model, task(a, b, function), seed, trials,
                        options, args.device,
                    )
                    result.update(
                        a=a, b=b, function=function, model_seed=model_seed,
                    )
                    np.savez_compressed(destination / f"{name}.npz", **arrays)
                    save_json(path, result)
                    print(
                        split, name, f"{result['success']}/{trials}",
                        f"MSE={result['mean_selected_population_mse']:.5f}",
                        f"seconds={result['seconds']:.1f}", flush=True,
                    )
        save_json(run / f"{split.upper()}_COMPLETE.json", {
            "contexts": len(functions) * len(config["model_seeds"]) * 4,
        })
    publish(run, args.baseline)


if __name__ == "__main__":
    main()
