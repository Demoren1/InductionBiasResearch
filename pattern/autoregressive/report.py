import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from pattern.autoregressive.model import IndexMADE
from pattern.codebook.models import IndexVAE, index_probabilities

NAMES = {
    "autoregressive": "Авторегрессия",
    "frequency": "Независимые частоты",
    "bank": "Готовый индексатор банка",
    "uniform": "Однородный случайный",
    "untrained_ar": "Необученная авторегрессия",
    "vae_sample": "VAE: предложение",
    "vae_z": "VAE + оптимизация z",
    "inferred": "Извлечённый общий I",
    "dense_adam": "Плотный Adam",
    "dense_sgd": "Плотный SGD",
    "oracle": "Правильный I задан",
}


def read_history(path):
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(path.with_suffix(path.suffix + ".gz"), "rt") as stream:
        return json.load(stream)


def plot_task(assets):
    sequence = np.array([int(bit) for bit in "11001010110"])
    template = np.array([0, 0, 1, 0])
    windows = np.lib.stride_tricks.sliding_window_view(sequence, 4)
    matches = (windows == template).all(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].imshow(sequence[None], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for index, bit in enumerate(sequence):
        axes[0].text(
            index,
            0,
            str(bit),
            ha="center",
            va="center",
            color="white" if bit else "black",
            fontsize=17,
        )
    axes[0].set_xticks(np.arange(11), np.arange(1, 12))
    axes[0].set_yticks([])
    axes[0].set_xlabel("Номер входа")
    axes[0].set_title("Вход: ищем шаблон 0010")
    axes[1].imshow(windows, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row in range(8):
        for column in range(4):
            bit = windows[row, column]
            axes[1].text(
                column,
                row,
                str(bit),
                ha="center",
                va="center",
                color="white" if bit else "black",
            )
    axes[1].set_yticks(
        np.arange(8),
        [
            f"Окно {i + 1}" + (" — совпало" if hit else "")
            for i, hit in enumerate(matches)
        ],
    )
    axes[1].set_xticks(np.arange(4), np.arange(1, 5))
    axes[1].set_xlabel("Позиция внутри окна")
    axes[1].set_title("Все восемь окон; правильный ответ y = 1")
    fig.savefig(assets / "task.png", dpi=150)
    plt.close(fig)


def structural_metrics(indices, truth):
    support = indices != 0
    target = truth != 0
    old, new = linear_sum_assignment(-(support.T.astype(int) @ target.astype(int)))
    order = np.empty(8, dtype=int)
    order[new] = old
    aligned = indices[:, order]
    active = aligned != 0
    support_f1 = 2 * (active & target).sum() / max(1, active.sum() + target.sum())
    table = np.zeros((4, 4), dtype=int)
    for actual, expected in zip(aligned[target], truth[target]):
        if actual:
            table[actual - 1, expected - 1] += 1
    rows, columns = linear_sum_assignment(-table)
    mapping = np.zeros(5, dtype=int)
    mapping[rows + 1] = columns + 1
    aligned = mapping[aligned]
    pair_count = lambda count: count * (count - 1) / 2
    predicted_pairs = sum(pair_count((aligned == label).sum()) for label in range(1, 5))
    true_pairs = sum(pair_count((truth == label).sum()) for label in range(1, 5))
    matched = sum(
        pair_count(((aligned == actual) & (truth == expected)).sum())
        for actual in range(1, 5)
        for expected in range(1, 5)
    )
    sharing_f1 = 2 * matched / max(1, predicted_pairs + true_pairs)
    return {
        "support_f1": float(support_f1),
        "sharing_f1": float(sharing_f1),
        "exact": bool(np.array_equal(aligned, truth)),
        "active_edges": int(active.sum()),
    }, aligned


def proposal_audit(plan, output, device):
    count = plan["prior"]["diagnostic_samples"]
    arrays = np.load(output / "bank_inputs.npz")
    train = torch.tensor(arrays["labels"][arrays["roles"] == 0], device=device)
    frequency = torch.tensor(arrays["frequency"], device=device)
    consensus = torch.tensor(arrays["consensus"].reshape(1, -1), device=device)
    model = IndexMADE(width=plan["prior"]["width"]).to(device)
    model.load_state_dict(torch.load(output / "prior.pt", weights_only=True))
    model.eval().requires_grad_(False)
    torch.manual_seed(plan["seed"] + 50)
    with torch.no_grad():
        ar = torch.cat(
            [model.sample(min(512, count - start)) for start in range(0, count, 512)]
        )
        independent = torch.multinomial(frequency, count, replacement=True).T
        uniform = torch.multinomial(
            frequency.mean(0).expand(88, -1), count, replacement=True
        ).T
        chosen = train[torch.randint(len(train), (count,), device=device)]
        source_plan = json.loads(Path(plan["source_protocol"]).read_text())
        vae = IndexVAE(len(source_plan["source_patterns"]), source_plan["vae"]).to(
            device
        )
        vae.load_state_dict(
            torch.load(Path(plan["source_run"]) / "prior.pt", weights_only=True)
        )
        vae.eval().requires_grad_(False)
        latent = torch.randn(count, source_plan["vae"]["latent_dim"], device=device)
        codes = index_probabilities(vae.decode(latent)).argmax(-1)
        mapping = torch.tensor(arrays["mapping"], device=device)
        vae_samples = mapping[codes].flatten(1)
        ar_logp = -F.cross_entropy(
            model(consensus).flatten(0, 1), consensus.flatten(), reduction="sum"
        )
        frequency_logp = frequency.log().gather(1, consensus.T).sum()
    samples = {
        "autoregressive": ar,
        "frequency": independent,
        "bank": chosen,
        "uniform": uniform,
        "vae_sample": vae_samples,
    }
    rows = np.arange(11)[:, None] - np.arange(8)[None]
    truth = np.where((rows >= 0) & (rows < 4), rows + 1, 0)
    result = {
        "samples_per_method": count,
        "ar_log_probability_source_consensus": float(ar_logp),
        "frequency_log_probability_source_consensus": float(frequency_logp),
        "source_consensus_matches_truth": structural_metrics(
            arrays["consensus"], truth
        )[0]["exact"],
        "methods": {},
    }
    saved = {}
    for method, values in samples.items():
        values = values.cpu().numpy().reshape(-1, 11, 8)
        metrics = [structural_metrics(value, truth)[0] for value in values]
        exact = sum(item["exact"] for item in metrics)
        train_set = {row.tobytes() for row in arrays["labels"][arrays["roles"] == 0]}
        copies = sum(value.flatten().tobytes() in train_set for value in values)
        result["methods"][method] = {
            "exact_indexators": exact,
            "estimated_exact_probability": exact / count,
            "mean_support_f1": float(np.mean([item["support_f1"] for item in metrics])),
            "mean_sharing_f1": float(np.mean([item["sharing_f1"] for item in metrics])),
            "mean_active_edges": float(
                np.mean([item["active_edges"] for item in metrics])
            ),
            "exact_bank_copies": copies,
            "unique_samples": len(np.unique(values.reshape(-1, 88), axis=0)),
        }
        saved[method] = values
    np.savez_compressed(output / "diagnostic_proposals.npz", **saved)
    (output / "proposal_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result, truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pattern/runs/autoregressive_20261101_v1")
    args = parser.parse_args()
    output = Path(args.run)
    plan = json.loads((output / "frozen_protocol.json").read_text())
    assets = Path("mds/assets/2026-11-01") / output.name
    assets.mkdir(parents=True, exist_ok=True)
    plot_task(assets)
    torch.set_num_threads(4)
    audit, truth = proposal_audit(plan, output, torch.device("cuda"))
    summary = []
    examples = {}
    for method in plan["fit"]["methods"]:
        data = np.load(output / (method + ".npz"))
        query = data["query_loss"].reshape(4, 32)
        chosen = np.arange(4) * 32 + query.argmin(1)
        row = {
            "method": method,
            "name": NAMES[method],
            "perfect_starts": int((data["population_accuracy"] == 1).sum()),
            "perfect_tasks": int((data["population_accuracy"][chosen] == 1).sum()),
            "selected_test_accuracy": float(data["test_accuracy"][chosen].mean()),
            "selected_population_accuracy": float(
                data["population_accuracy"][chosen].mean()
            ),
            "selected_test_bce": float(data["test_loss"][chosen].mean()),
            "mean_start_test_accuracy": float(data["test_accuracy"].mean()),
            "selected_test_recall": float(data["test_recall_positive"][chosen].mean()),
            "per_task_test_accuracy": data["test_accuracy"][chosen].tolist(),
            "per_task_population_accuracy": data["population_accuracy"][
                chosen
            ].tolist(),
        }
        if not method.startswith("dense_"):
            structural = [
                structural_metrics(data["indices"][index], truth)[0] for index in chosen
            ]
            row["selected_support_f1"] = float(
                np.mean([item["support_f1"] for item in structural])
            )
            row["selected_sharing_f1"] = float(
                np.mean([item["sharing_f1"] for item in structural])
            )
            row["selected_exact_indexators"] = sum(item["exact"] for item in structural)
            examples[method] = structural_metrics(data["indices"][chosen[0]], truth)[1]
        else:
            row.update(
                selected_support_f1=None,
                selected_sharing_f1=None,
                selected_exact_indexators=None,
            )
            examples[method] = data["parameter_0"][chosen[0]]
        summary.append(row)
    payload = {
        "summary": summary,
        "proposal_audit": audit,
        "likelihood": json.loads((output / "likelihood_audit.json").read_text()),
    }
    (assets / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
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
    axes[0].set_xlabel("Точность на test после выбора по query, %")
    axes[0].set_xlim(0, 102)
    axes[1].set_xlabel("Безошибочные старты на всех 2048 входах, из 128")
    fig.savefig(assets / "comparison.png")
    plt.close(fig)
    methods = [
        "autoregressive",
        "frequency",
        "bank",
        "vae_z",
        "inferred",
        "dense_adam",
        "dense_sgd",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for method in methods:
        history = read_history(output / (method + "_history.json"))
        x = [row["step"] for row in history]
        for axis, split, metric in [
            (axes[0, 0], "train", "loss"),
            (axes[0, 1], "query", "loss"),
            (axes[1, 0], "train", "recall_positive"),
            (axes[1, 1], "query", "recall_positive"),
        ]:
            axis.plot(
                x, [np.mean(row[split][metric]) for row in history], label=NAMES[method]
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
    history = read_history(output / "prior_history.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    for split, name in [("train", "Обучение"), ("validation", "Проверка")]:
        for axis, metric in zip(axes, ["nll_per_index", "support_recall"]):
            axis.plot(
                [row["step"] for row in history],
                [row[split][metric] for row in history],
                label=name,
            )
    for axis, title in zip(
        axes, ["NLL на один индекс", "Recall ненулевых индексов (teacher forcing)"]
    ):
        axis.set_title(title)
        axis.set_xlabel("Шаг обучения генератора")
        axis.legend()
        axis.grid(alpha=0.2)
    fig.savefig(assets / "prior_training.png")
    plt.close(fig)
    shown = [
        "autoregressive",
        "frequency",
        "bank",
        "vae_z",
        "inferred",
        "dense_adam",
        "dense_sgd",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 8), constrained_layout=True)
    for axis, method in zip(axes.flat, ["truth", *shown]):
        matrix = truth if method == "truth" else examples[method]
        axis.imshow(
            matrix,
            cmap="coolwarm" if method.startswith("dense_") else "viridis",
            aspect="auto",
        )
        axis.set_title("Аналитический I" if method == "truth" else NAMES[method])
        axis.set_xlabel("Скрытый нейрон")
        axis.set_ylabel("Вход")
    fig.savefig(assets / "indexators.png")
    plt.close(fig)
    frozen = plan["hashes"]
    checked = {}
    for name, expected in frozen.items():
        path = Path(name)
        snapshot = output / ("snapshot_" + path.name)
        if (
            path.parent == Path("pattern/autoregressive")
            and path.suffix == ".py"
            and snapshot.exists()
        ):
            path = snapshot
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checked[name] = actual == expected
    (output / "hash_audit.json").write_text(json.dumps(checked, indent=2) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
