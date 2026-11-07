import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from pattern.autoregressive.debug import proposals
from pattern.autoregressive.model import IndexMADE
from pattern.autoregressive.report import read_history, structural_metrics
from pattern.autoregressive.run import save_json

NAMES = {
    "ar_original": "AR: исходная генерация",
    "ar_calibrated": "AR: выбранная температура",
    "ar_greedy": "AR: жадный выбор",
    "frequency_original": "Частоты: исходная генерация",
    "frequency_calibrated": "Частоты: выбранная температура",
    "frequency_greedy": "Частоты: жадный выбор",
    "inferred": "Извлечённый общий индексатор",
    "dense_adam": "Плотный Adam",
    "dense_sgd": "Плотный SGD",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", default="pattern/runs/autoregressive_debug_20261101_v1"
    )
    args = parser.parse_args()
    output = Path(args.run)
    plan = json.loads((output / "frozen_protocol.json").read_text())
    selection = json.loads((output / "selection_frozen.json").read_text())
    assets = Path("mds/assets/2026-11-01/autoregressive_debug")
    assets.mkdir(parents=True, exist_ok=True)
    prior = Path(plan["prior_run"])
    arrays = np.load(prior / "bank_inputs.npz")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    model = IndexMADE().to(device)
    model.load_state_dict(torch.load(prior / "prior.pt", weights_only=True))
    model.eval().requires_grad_(False)
    frequency = torch.tensor(arrays["frequency"], device=device)
    consensus = torch.tensor(arrays["consensus"].reshape(1, 88), device=device)
    rows = np.arange(11)[:, None] - np.arange(8)[None]
    truth = np.where((rows >= 0) & (rows < 4), rows + 1, 0)
    mechanism = []
    saved_samples = {}
    for family, generator in [("ar", model), ("frequency", None)]:
        for temperature in plan["temperatures"]:
            with torch.no_grad():
                logits = (
                    model(consensus) if generator is not None else frequency.log()[None]
                )
                if temperature == 0:
                    greedy = proposals(generator, frequency, 1, 0, plan["seed"] + 80)
                    logp = 0.0 if torch.equal(greedy, consensus) else -float("inf")
                else:
                    logp = -float(
                        F.cross_entropy(
                            (logits / temperature).flatten(0, 1),
                            consensus.flatten(),
                            reduction="sum",
                        )
                    )
                count = plan["diagnostic_samples"]
                if temperature == 0:
                    samples = proposals(
                        generator, frequency, 1, temperature, plan["seed"] + 80
                    ).expand(count, -1)
                elif generator is None:
                    samples = proposals(
                        generator, frequency, count, temperature, plan["seed"] + 80
                    )
                else:
                    samples = torch.cat(
                        [
                            proposals(
                                generator,
                                frequency,
                                min(512, count - start),
                                temperature,
                                plan["seed"] + 80 + start,
                            )
                            for start in range(0, count, 512)
                        ]
                    )
            samples = samples.cpu().numpy().reshape(-1, 11, 8)
            metrics = [structural_metrics(sample, truth)[0] for sample in samples]
            name = f"{family}_t{temperature:g}"
            mechanism.append(
                {
                    "family": family,
                    "temperature": temperature,
                    "canonical_consensus_probability": float(np.exp(logp)),
                    "exact_of_8192": sum(item["exact"] for item in metrics),
                    "mean_support_f1": float(
                        np.mean([item["support_f1"] for item in metrics])
                    ),
                    "mean_sharing_f1": float(
                        np.mean([item["sharing_f1"] for item in metrics])
                    ),
                    "unique_samples": len(np.unique(samples.reshape(-1, 88), axis=0)),
                }
            )
            saved_samples[name] = samples
            print("COVERAGE", mechanism[-1], flush=True)
    np.savez_compressed(output / "temperature_proposals.npz", **saved_samples)
    save_json(output / "mechanism_audit.json", mechanism)
    summary = []
    examples = {}
    for method, name in NAMES.items():
        data = np.load(output / (method + ".npz"))
        query = data["query_loss"].reshape(4, 32)
        chosen = np.arange(4) * 32 + query.argmin(1)
        row = {
            "method": method,
            "name": name,
            "selected_test_accuracy": float(data["test_accuracy"][chosen].mean()),
            "selected_population_accuracy": float(
                data["population_accuracy"][chosen].mean()
            ),
            "selected_test_bce": float(data["test_loss"][chosen].mean()),
            "mean_start_test_accuracy": float(data["test_accuracy"].mean()),
            "perfect_starts": int((data["population_accuracy"] == 1).sum()),
            "perfect_tasks": int((data["population_accuracy"][chosen] == 1).sum()),
            "per_task_population_accuracy": data["population_accuracy"][
                chosen
            ].tolist(),
        }
        if not method.startswith("dense_"):
            structures = [
                structural_metrics(data["indices"][index], truth)[0] for index in chosen
            ]
            row["selected_exact_indexators"] = sum(item["exact"] for item in structures)
            examples[method] = structural_metrics(data["indices"][chosen[0]], truth)[1]
        else:
            row["selected_exact_indexators"] = None
            examples[method] = data["parameter_0"][chosen[0]]
        summary.append(row)
    calibration = json.loads((output / "calibration/summary.json").read_text())
    payload = {
        "summary": summary,
        "calibration": calibration,
        "choices": selection["choices"],
        "mechanism": mechanism,
    }
    save_json(output / "summary.json", payload)
    save_json(assets / "summary.json", payload)
    with (assets / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    positions = np.arange(len(summary))
    axes[0].barh(
        positions,
        [row["selected_test_accuracy"] * 100 for row in summary],
        color="#167e86",
    )
    axes[1].barh(positions, [row["perfect_starts"] for row in summary], color="#365a80")
    for axis in axes:
        axis.set_yticks(positions, [row["name"] for row in summary])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_xlabel("Test после выбора по query, %")
    axes[0].set_xlim(0, 102)
    axes[1].set_xlabel("Безошибочные старты на 2048 входах, из 128")
    fig.savefig(assets / "comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for family, name in [("ar", "Авторегрессия"), ("frequency", "Независимые частоты")]:
        own = sorted(
            [row for row in mechanism if row["family"] == family],
            key=lambda row: row["temperature"],
        )
        axes[0].plot(
            [row["temperature"] for row in own],
            [row["exact_of_8192"] / 8192 * 100 for row in own],
            "o-",
            label=name,
        )
        own = sorted(
            [row for row in calibration if row["family"] == family],
            key=lambda row: row["temperature"],
        )
        axes[1].plot(
            [row["temperature"] for row in own],
            [row["query_bce"] for row in own],
            "o-",
            label=name,
        )
    axes[0].set_title("Покрытие: диагностика после фиксации метода")
    axes[0].set_ylabel("Точных эталонных структур, %")
    axes[1].set_title("Выбор температуры только по исходным задачам")
    axes[1].set_ylabel("Средняя выбранная source-query BCE")
    for axis in axes:
        axis.set_xlabel("Температура; 0 — жадный выбор")
        axis.legend()
        axis.grid(alpha=0.2)
    fig.savefig(assets / "temperature.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for method in [
        "ar_original",
        "ar_calibrated",
        "frequency_calibrated",
        "ar_greedy",
        "dense_adam",
        "dense_sgd",
    ]:
        history = read_history(output / (method + "_history.json"))
        for axis, split, metric in [
            (axes[0, 0], "train", "loss"),
            (axes[0, 1], "query", "loss"),
            (axes[1, 0], "train", "recall_positive"),
            (axes[1, 1], "query", "recall_positive"),
        ]:
            axis.plot(
                [row["step"] for row in history],
                [np.mean(row[split][metric]) for row in history],
                label=NAMES[method],
            )
    for axis, title in zip(
        axes.flat,
        [
            "Train: balanced BCE",
            "Query: BCE",
            "Train: recall положительного класса",
            "Query: recall положительного класса",
        ],
    ):
        axis.set_title(title)
        axis.set_xlabel("Шаг оптимизатора")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(assets / "training.png")
    plt.close(fig)
    shown = [
        "ar_original",
        "ar_calibrated",
        "frequency_calibrated",
        "ar_greedy",
        "inferred",
    ]
    fig, axes = plt.subplots(1, 6, figsize=(13, 4), constrained_layout=True)
    for axis, method in zip(axes, ["truth", *shown]):
        axis.imshow(
            truth if method == "truth" else examples[method],
            vmin=0,
            vmax=4,
            cmap="viridis",
            aspect="auto",
        )
        axis.set_title(
            "Эталон" if method == "truth" else NAMES[method].replace(": ", ":\n"),
            fontsize=9,
        )
        axis.set_xlabel("Скрытый нейрон")
    axes[0].set_ylabel("Вход")
    fig.savefig(assets / "indexators.png")
    plt.close(fig)
    checked = {}
    for name, expected in plan["hashes"].items():
        path = Path(name)
        snapshot = output / ("snapshot_" + path.name)
        if (
            path.parent == Path("pattern/autoregressive")
            and path.suffix == ".py"
            and snapshot.exists()
        ):
            path = snapshot
        checked[name] = hashlib.sha256(path.read_bytes()).hexdigest() == expected
    save_json(output / "hash_audit.json", checked)
    print("RESULTS", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
