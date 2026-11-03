"""Aggregate frozen evaluations and inspect generation without retuning."""

import argparse
import csv
import json

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.sine_ar_repair_v1.calibrate import generators
from experiments.sine_ar_repair_v1.run import (
    OUTPUT,
    ROOT,
    configurations,
    dump,
    source_masks,
)

ASSETS = ROOT / "mds/assets/2026-11-03/sine_ar_repair"
LABELS = {
    "strict_t2": "Прежний AR, T=2",
    "strict_t6": "Прежний AR, T=6",
    "relaxed_t1": "Мягкий банк, T=1",
    "relaxed_selection_tuned": "Мягкий банк + отбор",
    "joint": "Joint AR",
    "token_noise": "AR + шум токенов",
    "cross_noise": "AR + шум чужого выхода",
    "block": "AR по выходам",
    "vae": "NF+VAE",
    "partwise_vae": "Partwise NF+VAE",
    "mask_vae": "VAE только масок",
    "partwise_mask_vae": "Partwise VAE масок",
    "joint_block_mixture": "Смесь joint / по выходам",
    "frequency": "Независимые биты",
    "product": "Произведение частей банка",
    "replay": "Повтор масок банка",
    "uniform": "Случайные 2 из 16",
    "dense_sgd": "Плотный SGD",
    "dense_adam": "Плотный Adam",
    "oracle": "Oracle + SGD",
}


def label(name):
    if name.endswith("_prior_only"):
        return LABELS[name.removesuffix("_prior_only")] + ", исходный SGD"
    return LABELS[name]


def encode(masks):
    bits = masks.reshape(*masks.shape[:-2], 16).astype(np.int64)
    return (bits * (1 << np.arange(16))).sum(-1)


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, specs):
    pairs = sorted({(s["a"], s["b"]) for s in specs})
    grouped = np.array(
        [
            values[
                [i for i, s in enumerate(specs) if (s["a"], s["b"]) == p]
            ].mean()
            for p in pairs
        ]
    )
    rng = np.random.default_rng(261103840)
    draws = grouped[rng.integers(len(pairs), size=(10000, len(pairs)))]
    return np.quantile(draws.mean(1), [0.025, 0.975]).tolist()


def aggregate():
    methods = json.loads((OUTPUT / "selection_frozen.json").read_text())[
        "methods"
    ]
    bank = source_masks("cpu").numpy().reshape(-1, 8, 2)
    source_codes = np.unique(encode(bank))
    rows, per_search, paired = [], {}, []
    for condition in ["novel", "familiar"]:
        specs = json.loads((OUTPUT / condition / "tasks.json").read_text())
        truth = np.zeros((len(specs), 8, 2), dtype=np.uint8)
        for t, spec in enumerate(specs):
            truth[t, spec["rows"], [0, 1]] = 1
        for method in methods:
            name = method["name"]
            with np.load(OUTPUT / condition / f"{name}.npz") as result:
                masks = result["masks"]
                selected = result["selected"]
                ti, si = np.indices(selected.shape)
                exploration = int(result["exploration_fits"])
                exact = (masks == truth[:, None, None]).all((-1, -2))
                covers = (masks >= truth[:, None, None]).all((-1, -2))
                success = result["selected_population"] < 0.1
                chosen_masks = masks[ti, si, selected]
                in_bank = np.isin(encode(chosen_masks), source_codes)
                lower, upper = interval(success, specs)
                refit = exact[:, :, exploration:]
                refit_covers = covers[:, :, exploration:]
                row = {
                    "condition": condition,
                    "method": name,
                    "searches": int(success.size),
                    "successes": int(success.sum()),
                    "success_percent": float(success.mean() * 100),
                    "pair_bootstrap_low": lower * 100,
                    "pair_bootstrap_high": upper * 100,
                    "mean_population_mse": float(
                        result["selected_population"].mean()
                    ),
                    "mean_query_mse": float(result["selected_query"].mean()),
                    "exact_offered": int(
                        exact[:, :, :exploration].any(2).sum()
                    ),
                    "cover_offered": int(
                        covers[:, :, :exploration].any(2).sum()
                    ),
                    "exact_in_refit": int(refit.any(2).sum())
                    if refit.shape[2]
                    else None,
                    "cover_in_refit": int(refit_covers.any(2).sum())
                    if refit_covers.shape[2]
                    else None,
                    "exact_final": int(exact[ti, si, selected].sum()),
                    "cover_final": int(covers[ti, si, selected].sum()),
                    "successful_mask_in_strict_bank": int(
                        (success & in_bank).sum()
                    ),
                    "successful_mask_new_to_strict_bank": int(
                        (success & ~in_bank).sum()
                    ),
                    "mean_active_edges": float(
                        chosen_masks.sum((-1, -2)).mean()
                    ),
                    "per_prior_seed_success": [
                        float(v.mean() * 100)
                        for v in np.split(success, 3, axis=1)
                    ],
                    "fits_per_search": masks.shape[2],
                    "steps_per_fit": method.get("policy", {}).get(
                        "steps", 1024
                    ),
                    "elapsed_fit_seconds": float(result["seconds"]),
                }
                rows.append(row)
                per_search[condition, name] = success
        for method in methods:
            name = method["name"]
            for reference in [
                "strict_t2",
                "strict_t6",
                "relaxed_t1",
                "dense_adam",
            ]:
                differences = per_search[condition, name].astype(
                    float
                ) - per_search[condition, reference].astype(float)
                lower, upper = interval(differences, specs)
                paired.append(
                    {
                        "condition": condition,
                        "method": name,
                        "reference": reference,
                        "difference_pp": float(differences.mean() * 100),
                        "pair_bootstrap_low": lower * 100,
                        "pair_bootstrap_high": upper * 100,
                    }
                )
    dump(OUTPUT / "summary.json", rows)
    dump(OUTPUT / "paired_differences.json", paired)
    save_csv(ASSETS / "comparison.csv", rows)
    save_csv(ASSETS / "paired_differences.csv", paired)
    screen = json.loads((OUTPUT / "screen_summary.json").read_text())
    family_count = len(
        json.loads((OUTPUT / "confirmation_candidates.json").read_text())
    )
    cost = {
        "source_bank_fits_reused": 262144,
        "source_bank_search_budget_equivalents": 4096,
        "trained_priors": len(configurations()) * 3,
        "prior_optimizer_steps": len(configurations()) * 3 * 6000,
        "screen_settings": len(screen),
        "screen_searches": len(screen) * 16 * 2 * 3,
        "family_confirmation_and_tuning_searches": family_count
        * (2 + 6 + 5)
        * 16
        * 8
        * 3,
        "mixture_and_relaxed_tuning_searches": (3 + 6) * 16 * 8 * 3,
        "dense_tuning_searches": (12 + 16) * 4 * 16,
        "target_searches": sum(r["searches"] for r in rows),
        "training_examples_per_search": 4194304,
        "note": (
            "Training examples and optimizer steps, not FLOPs or wall time. "
            "Query evaluation and post-freeze diagnostic sampling are extra."
        ),
    }
    dump(OUTPUT / "compute_cost.json", cost)
    rule = json.loads((OUTPUT / "report_protocol.json").read_text())
    choices = []
    for name in rule["primary_ar_candidates"]:
        if name == "joint_block_mixture":
            record = json.loads(
                (OUTPUT / "confirmation/controls/selection.json").read_text()
            )["mixture"]
        else:
            record = json.loads(
                (OUTPUT / "confirmation" / name / "selection.json").read_text()
            )
        choices.append({"name": name, "mean_query": record["mean_query"]})
    dump(
        OUTPUT / "primary_selection.json",
        {
            "rule": rule,
            "candidates": choices,
            "selected": min(choices, key=lambda r: r["mean_query"]),
        },
    )
    dump(
        OUTPUT / "COMPLETE.json",
        {"complete": True, "method_count": len(methods)},
    )
    return rows


@torch.no_grad()
def diagnose(device):
    frozen = json.loads((OUTPUT / "selection_frozen.json").read_text())[
        "methods"
    ]
    bank = source_masks(device)
    source = bank.cpu().numpy().reshape(-1, 8, 2)
    source_codes = np.unique(encode(source))
    histogram_codes = np.arange(65536, dtype=np.int64)
    bits = ((histogram_codes[:, None] >> np.arange(16)) & 1).reshape(-1, 8, 2)
    column_codes = (bits * (1 << np.arange(8))[None, :, None]).sum(1)
    source_parts = [np.unique(column_codes[source_codes, j]) for j in range(2)]
    recombinations = (
        np.isin(column_codes[:, 0], source_parts[0])
        & np.isin(column_codes[:, 1], source_parts[1])
        & ~np.isin(histogram_codes, source_codes)
    )
    specs = json.loads((OUTPUT / "novel/tasks.json").read_text())
    targets = np.zeros((len(specs), 8, 2), dtype=np.uint8)
    for t, spec in enumerate(specs):
        targets[t, spec["rows"], [0, 1]] = 1
    target_codes = encode(targets)
    rows, histograms = [], {}
    for method in frozen:
        name = method["name"]
        if "dense" in method or name.endswith("_prior_only"):
            continue
        if name == "relaxed_selection_tuned":
            continue
        setting = method["setting"]
        counts = []
        for index, model in enumerate(generators(setting, device)):
            torch.manual_seed(261103850 + index * 1000)
            samples = model.sample(
                32768,
                setting["scale"],
                bank,
                setting.get("sampling_noise", False),
            )
            codes = encode(samples.cpu().numpy().reshape(-1, 8, 2))
            counts.append(np.bincount(codes, minlength=65536))
        counts = np.asarray(counts)
        histograms[name] = counts
        p = counts.sum(0) / counts.sum()
        q0 = np.bincount(column_codes[:, 0], weights=p, minlength=256)
        q1 = np.bincount(column_codes[:, 1], weights=p, minlength=256)

        def entropy(probability):
            positive = probability[probability > 0]
            return float(-(positive * np.log(positive)).sum())

        rows.append(
            {
                "method": name,
                "draws": int(counts.sum()),
                "unique_masks": int((p > 0).sum()),
                "strict_bank_mass": float(p[source_codes].sum()),
                "new_recombination_mass": float(p[recombinations].sum()),
                "mean_novel_exact_mass": float(p[target_codes].mean()),
                "min_novel_exact_mass": float(p[target_codes].min()),
                "mean_active_edges": float((p * bits.sum((1, 2))).sum()),
                "one_edge_per_output_mass": float(
                    p[(bits.sum(1) == 1).all(1)].sum()
                ),
                "whole_entropy_nats": entropy(p),
                "output_mutual_information_plugin": entropy(q0)
                + entropy(q1)
                - entropy(p),
                "note": (
                    "MC diagnostic after freezing; 3 x 32768 draws. "
                    "Plugin MI is biased; no model selection."
                ),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    np.savez_compressed(OUTPUT / "generation_histograms.npz", **histograms)
    dump(OUTPUT / "generation_diagnostics.json", rows)
    save_csv(ASSETS / "generation_diagnostics.csv", rows)


def save_figure(fig, name):
    for extension in ["png", "svg"]:
        fig.savefig(
            ASSETS / f"{name}.{extension}", dpi=180, bbox_inches="tight"
        )
    plt.close(fig)


def figures(rows):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    by_key = {(r["condition"], r["method"]): r for r in rows}
    names = [
        "strict_t2",
        "strict_t6",
        "relaxed_t1",
        "joint",
        "token_noise",
        "cross_noise",
        "block",
        "vae",
        "partwise_vae",
        "product",
        "dense_adam",
        "oracle",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
    y = np.arange(len(names))
    for ax, condition, title in zip(
        axes, ["novel", "familiar"], ["Новые сочетания", "Знакомые сочетания"]
    ):
        final = np.array(
            [by_key[condition, n]["success_percent"] for n in names]
        )
        original = np.array(
            [
                by_key[condition, n + "_prior_only"]["success_percent"]
                if (condition, n + "_prior_only") in by_key
                else np.nan
                for n in names
            ]
        )
        ax.barh(
            y,
            final,
            height=0.62,
            color="#257a8a",
            label="Настроены prior и SGD/отбор",
        )
        ax.plot(
            original,
            y,
            "o",
            color="#e69836",
            markersize=6,
            label="Тот же prior, исходный SGD/отбор",
        )
        for index, value in enumerate(final):
            ax.text(
                min(value + 1, 101),
                index,
                f"{value:.1f}",
                va="center",
                fontsize=9,
            )
        ax.set(xlim=(0, 110), xlabel="Успех, % (384 поиска)", title=title)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [label(n) for n in names])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Один банк случайных решений; новые функции; "
        "одинаковый целевой бюджет",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, "comparison")

    names = [
        "strict_t2",
        "relaxed_t1",
        "joint",
        "token_noise",
        "cross_noise",
        "block",
        "partwise_vae",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in names:
        row = by_key["novel", name]
        values = [
            row["cover_offered"],
            row["cover_in_refit"],
            row["successes"],
        ]
        axes[0].plot(
            range(3),
            np.array(values) / row["searches"] * 100,
            "o-",
            label=label(name),
        )
    axes[0].set_xticks(
        range(3), ["Предложены\nнужные рёбра", "Переданы\nв refit", "Успех"]
    )
    axes[0].set(
        ylabel="Доля поисков, %",
        ylim=(0, 105),
        title="Потери покрытия и выбора на новых сочетаниях",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="lower left")
    diagnostics = json.loads(
        (OUTPUT / "generation_diagnostics.json").read_text()
    )
    values = {r["method"]: r for r in diagnostics}
    order = [
        "strict_t2",
        "joint",
        "token_noise",
        "cross_noise",
        "block",
        "vae",
        "partwise_vae",
        "product",
    ]
    axes[1].barh(
        np.arange(len(order)),
        [values[n]["mean_novel_exact_mass"] * 100 for n in order],
        color="#257a8a",
    )
    axes[1].set_yticks(
        np.arange(len(order)), [label(n) for n in order], fontsize=8
    )
    axes[1].invert_yaxis()
    axes[1].set(
        xlabel="Вероятность точной новой маски, %",
        title="Покрытие одной случайной генерацией",
    )
    fig.tight_layout()
    save_figure(fig, "coverage")

    configs = {c["id"]: c for c in configurations()}
    screen = json.loads((OUTPUT / "screen_summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    for identifier in ["model_00", "model_02", "model_06"]:
        traces = [
            json.loads((OUTPUT / identifier / f"history_{i}.json").read_text())
            for i in range(3)
        ]
        x = [r["step"] for r in traces[0]]
        for key, style in [("train_clean", "-"), ("validation_clean", "--")]:
            y = np.mean([[r[key] for r in h] for h in traces], axis=0)
            width = configs[identifier]["width"]
            axes[0].plot(
                x,
                y,
                style,
                label=f"w={width}, {'train' if style == '-' else 'val'}",
            )
    axes[0].set(
        xscale="log",
        ylim=(0.12, 0.4),
        xlabel="Шаг обучения prior",
        ylabel="NLL на токен",
        title="Joint AR: качество описания source-банка",
    )
    axes[0].legend(fontsize=7)
    for family in [
        "joint",
        "token_noise",
        "cross_noise",
        "block",
        "vae",
        "partwise_vae",
    ]:
        chosen = [r for r in screen if r["family"] == family]
        steps = sorted({r["step"] for r in chosen})
        error = [
            min(r["mean_query"] for r in chosen if r["step"] == s)
            for s in steps
        ]
        axes[1].plot(steps, error, "o-", label=label(family))
    axes[1].set(
        xscale="log",
        xlabel="Шаг обучения prior",
        ylabel="Query MSE на screen",
        title="Лучшая настройка каждого checkpoint",
    )
    axes[1].legend(fontsize=6)
    for step in [100, 1000, 6000]:
        chosen = sorted(
            (
                r
                for r in screen
                if r["model"] == "model_02" and r["step"] == step
            ),
            key=lambda r: r["scale"],
        )
        axes[2].plot(
            [r["scale"] for r in chosen],
            [r["mean_query"] for r in chosen],
            "o-",
            label=f"{step} шагов",
        )
    axes[2].set(
        xlabel="Температура генерации",
        ylabel="Query MSE на screen",
        title="Joint AR: температура и длительность",
    )
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "training")

    latent = json.loads((OUTPUT / "latent_diagnostics.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for identifier in ["model_19", "model_20", "model_21"]:
        selected = [r for r in latent if r["model"] == identifier]
        steps = sorted({r["step"] for r in selected})
        beta = configs[identifier]["beta"]
        for ax, key in zip(
            axes, ["train_kl", "validation_exact_reconstruction"]
        ):
            values = [
                np.mean([r[key] for r in selected if r["step"] == s])
                for s in steps
            ]
            ax.plot(steps, values, "o-", label=rf"$\beta={beta}$")
            ax.set_xscale("log")
            ax.set_xlabel("Шаг обучения prior")
            ax.grid(alpha=0.2)
    axes[0].set(
        ylabel="Средний KL", title="Partwise NF+VAE:\nинформация в latent"
    )
    axes[1].set(
        ylabel="Доля точно восстановленных масок",
        title="Точное восстановление source-validation\nиз среднего posterior",
        ylim=(-0.03, 1.05),
    )
    axes[0].legend()
    fig.tight_layout()
    save_figure(fig, "vae_training")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["aggregate", "diagnose", "figures"], required=True
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.phase == "aggregate":
        aggregate()
    elif args.phase == "diagnose":
        diagnose(args.device)
    else:
        figures(json.loads((OUTPUT / "summary.json").read_text()))


if __name__ == "__main__":
    main()
