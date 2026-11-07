"""Freeze the complete G28 cohort, then evaluate held-out data and draw figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_indexator_v1.model import metrics

from .joint_readout_control import BASE, PATTERN, feature, pattern_predict, reg_values
from .pattern_prefix import timestamp

LABELS = {
    "shared": "Shared scalar",
    "fixed": "Fixed sum",
    "pooled": "Learned pooled W2",
    "free": "Independent W2",
    "random_free": "Random I + independent W2",
}
COLORS = {
    "shared": "#787878",
    "fixed": "#787878",
    "pooled": "#3275ad",
    "free": "#dc7d28",
    "random_free": "#af4169",
}


def freeze(root):
    plan = json.loads((root / "protocol.json").read_text())
    if digest(BASE / "joint_readout_control.py") != plan["code_sha256"]:
        raise ValueError("Scientific runner changed after registration")
    models = [root / "pattern_frozen.json"] + [
        root / f"sets_frozen_{family}.json" for family in range(4)
    ]
    for path in models:
        frozen = json.loads(path.read_text())
        name = (
            "pattern_fits.npz"
            if path.name.startswith("pattern")
            else (f"sets_fits_{path.stem.rsplit('_', 1)[1]}.npz")
        )
        if digest(root / name) != frozen["sha256"]:
            raise ValueError("Frozen fitted parameters changed")
    for name, expected in plan["parents"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Historical source freeze changed")
    save_json(
        root / "cohort_frozen.json",
        {
            "frozen_utc": timestamp(),
            "scope": "All I and all numeric predictors fixed before any G28 test/permutation evaluation.",
            "sha256": {p.name: digest(p) for p in models},
            "evaluator_sha256": digest(Path(__file__)),
        },
    )


@torch.no_grad()
def evaluate_pattern(root, device):
    data = load_archive(root / "pattern_data.npz")
    heldout = load_archive(PATTERN / "data/binary_test.npz")
    fitted = load_archive(root / "pattern_fits.npz")
    frozen = json.loads((root / "pattern_frozen.json").read_text())
    full_x = np.concatenate(
        [data[k + "_x"] for k in ("support", "query", "selection")]
        + [heldout["test_x"]]
    )
    full_y = np.concatenate(
        [data[k + "_y"] for k in ("support", "query", "selection")]
        + [heldout["test_y"]],
        axis=1,
    )
    rows = []
    scores = {}
    for row in frozen["rows"]:
        key, kind = row["key"], row["head"]
        index = data["random_index"] if kind == "random_free" else data["index"]
        parameters = torch.tensor(fitted[key + "_parameters"], device=device)
        test = (
            metrics(
                pattern_predict(
                    feature(heldout["test_x"], index, 2, device, torch.float32),
                    parameters,
                    kind != "shared",
                ),
                torch.tensor(heldout["test_y"], device=device),
            )
            .cpu()
            .numpy()
        )
        full = (
            metrics(
                pattern_predict(
                    feature(full_x, index, 2, device, torch.float32),
                    parameters,
                    kind != "shared",
                ),
                torch.tensor(full_y, device=device),
            )
            .cpu()
            .numpy()
        )
        winner = row["winner"]
        result = {
            **row,
            "test_bce": float(test[winner, 0]),
            "test_balanced_accuracy": float(test[winner, 1]),
            "test_recall": float(test[winner, 2]),
            "full_balanced_accuracy": float(full[winner, 1]),
            "perfect_full_starts": int(np.isclose(full[:, 1], 1).sum()),
            "starts": len(parameters),
        }
        rows.append(result)
        scores[key + "_test"] = test
        scores[key + "_full"] = full
    summary = []
    for kind in ("shared", "free", "random_free"):
        subset = [r for r in rows if r["head"] == kind]
        summary.append(
            {
                "head": kind,
                "seeds": len(subset),
                "mean_test_bce": float(np.mean([r["test_bce"] for r in subset])),
                "mean_test_balanced_accuracy": float(
                    np.mean([r["test_balanced_accuracy"] for r in subset])
                ),
                "worst_test_balanced_accuracy": min(
                    r["test_balanced_accuracy"] for r in subset
                ),
                "mean_full_balanced_accuracy": float(
                    np.mean([r["full_balanced_accuracy"] for r in subset])
                ),
                "perfect_full_starts": sum(r["perfect_full_starts"] for r in subset),
                "total_starts": sum(r["starts"] for r in subset),
            }
        )
    np.savez_compressed(root / "pattern_evaluation.npz", **scores)
    return summary, rows


@torch.no_grad()
def evaluate_sets(root, device):
    plan = json.loads((root / "protocol.json").read_text())
    rows, trajectories = [], {}
    heatmaps = {}
    for family in range(4):
        data = load_archive(root / f"sets_data_{family}.npz")
        private = load_archive(root / "private" / f"sets_{family}.npz")
        fitted = load_archive(root / f"sets_fits_{family}.npz")
        frozen = json.loads((root / f"sets_frozen_{family}.json").read_text())
        for row in frozen["rows"]:
            key, task, kind = row["key"], row["task"], row["head"]
            prefix = key + "__"
            index = fitted[prefix + "index"]
            groups = torch.tensor(fitted[prefix + "head_groups"], device=device)
            all_parameters = fitted[prefix + "parameters"]
            winners = fitted[prefix + "winners"]
            selected_ids = np.arange(16) * 64 + winners.flatten()
            parameters = torch.tensor(all_parameters[selected_ids], device=device)
            x, truth = private[task + "_test_x"], private[task + "_test_y"]
            outputs = truth.shape[-1]
            prediction = reg_values(
                feature(x, index, 4, device, torch.float64),
                parameters,
                groups,
                outputs,
                0.15,
                kind == "fixed",
            )
            anchors = np.concatenate([data[f"{task}_zero_{c}"] for c in range(2)])
            prediction = prediction.cpu().numpy() + anchors[:, None]
            error = (prediction - np.tile(truth, (2, 1, 1))) ** 2
            mse = error.mean((1, 2))
            variance = truth.var(axis=1).mean(1)
            nmse = mse / np.tile(variance, 2)
            permutation = (
                reg_values(
                    feature(private["permutation_x"], index, 4, device, torch.float64),
                    parameters,
                    groups,
                    outputs,
                    0.15,
                    kind == "fixed",
                )
                .cpu()
                .numpy()
                .reshape(16, 64, 16, outputs)
            )
            rms = np.sqrt(
                ((permutation - permutation.mean(2, keepdims=True)) ** 2).mean(
                    (1, 2, 3)
                )
            )
            normalized_rms = rms / np.sqrt(np.tile(variance, 2))
            val = fitted[prefix + "validation"].reshape(2, 8, 64)
            for condition, sigma in enumerate(plan["sets"]["noise"]):
                for target in range(8):
                    i = condition * 8 + target
                    rows.append(
                        {
                            "family": family,
                            "task": task,
                            "head": kind,
                            "noise": sigma,
                            "target": target,
                            "mse": float(mse[i]),
                            "nmse": float(nmse[i]),
                            "permutation_normalized_rms": float(normalized_rms[i]),
                            "validation_mse": float(val[condition, target].min()),
                            "selected_start": int(winners[condition, target]),
                            "numerical_parameters": row["parameters"],
                        }
                    )
            tr = fitted[prefix + "train_history"][:, selected_ids].reshape(-1, 2, 8)
            vr = fitted[prefix + "validation_history"][:, selected_ids].reshape(
                -1, 2, 8
            )
            for condition in range(2):
                tag = f"{task}__{kind}__{condition}"
                if tag not in trajectories:
                    trajectories[tag] = {
                        "steps": fitted[prefix + "steps"],
                        "train": [],
                        "validation": [],
                    }
                trajectories[tag]["train"].append(tr[:, condition])
                trajectories[tag]["validation"].append(vr[:, condition])
            if family == 0 and kind in ("pooled", "free"):
                p = all_parameters[selected_ids[8]]
                code = np.r_[0, p[:4]]
                weight = np.sign(index) * code[np.abs(index)]
                head = p[4:].reshape(-1, outputs)[groups.cpu().numpy()] / 4
                heatmaps[key + "__W1"] = weight
                heatmaps[key + "__W2"] = head.T
                heatmaps[key + "__I"] = index
    summary = []
    for task in plan["sets"]["tasks"]:
        for sigma in plan["sets"]["noise"]:
            for kind in plan["sets"]["heads"]:
                subset = [
                    r
                    for r in rows
                    if r["task"] == task and r["noise"] == sigma and r["head"] == kind
                ]
                if not subset:
                    continue
                summary.append(
                    {
                        "task": task,
                        "noise": sigma,
                        "head": kind,
                        "functions": len(subset),
                        "mean_mse": float(np.mean([r["mse"] for r in subset])),
                        "mean_nmse": float(np.mean([r["nmse"] for r in subset])),
                        "worst_nmse": max(r["nmse"] for r in subset),
                        "functions_below_1e-4": sum(r["nmse"] < 1e-4 for r in subset),
                        "mean_permutation_normalized_rms": float(
                            np.mean([r["permutation_normalized_rms"] for r in subset])
                        ),
                    }
                )
    for value in trajectories.values():
        value["train"] = np.concatenate(value["train"], axis=1)
        value["validation"] = np.concatenate(value["validation"], axis=1)
    np.savez_compressed(root / "heatmap_values.npz", **heatmaps)
    return summary, rows, trajectories, heatmaps


def save_figure(fig, directory, name):
    fig.savefig(directory / (name + ".png"), dpi=160, bbox_inches="tight")
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def figures(root, pattern_rows, trajectories, heatmaps):
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    fitted = load_archive(root / "pattern_fits.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for kind in ("shared", "free", "random_free"):
        selected = [r for r in pattern_rows if r["head"] == kind]
        train = np.stack(
            [fitted[r["key"] + "_train"][:, r["winner"]] for r in selected]
        )
        val = np.stack(
            [fitted[r["key"] + "_validation"][:, r["winner"]] for r in selected]
        )
        recall = np.stack(
            [fitted[r["key"] + "_recall"][:, r["winner"]] for r in selected]
        )
        steps = fitted[selected[0]["key"] + "_steps"]
        axes[0].plot(steps, train.mean(0), color=COLORS[kind], linestyle="--")
        axes[0].plot(steps, val.mean(0), color=COLORS[kind], label=LABELS[kind])
        axes[1].plot(steps, recall.mean(0), color=COLORS[kind], label=LABELS[kind])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Balanced BCE")
    axes[0].set_title("Pattern 0110: train dashed / validation solid")
    axes[1].set_ylabel("Positive recall on validation")
    axes[1].set_title("Three seeds; checkpoint/start selected on validation")
    for ax in axes:
        ax.set_xlabel("Gradient evaluations")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, directory, "pattern_learning")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for task, ax in zip(("original_sum", "learned_pool_matrix"), axes, strict=True):
        for kind in ("fixed", "pooled", "free", "random_free"):
            tag = f"{task}__{kind}__1"
            if tag not in trajectories:
                continue
            values = trajectories[tag]
            ax.plot(values["steps"], values["train"].mean(1), "--", color=COLORS[kind])
            ax.plot(
                values["steps"],
                values["validation"].mean(1),
                color=COLORS[kind],
                label=LABELS[kind],
            )
        ax.set_title(task.replace("_", " ") + "; averaged raw noise 0.2")
        ax.set_yscale("log")
        ax.set_xlabel("Joint Gauss-Newton steps")
        ax.set_ylabel("MSE; train dashed / validation solid")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, directory, "deepsets_learning")
    data = load_archive(root / "pattern_data.npz")
    chosen = next(r for r in pattern_rows if r["head"] == "free" and r["seed"] == 0)
    parameters = fitted[chosen["key"] + "_parameters"][chosen["winner"]]
    code = np.r_[0, parameters[:2]]
    fig, axes = plt.subplots(
        1, 3, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1, 1.2]}
    )
    images = [
        data["index"],
        np.sign(data["index"]) * code[np.abs(data["index"])],
        parameters[None, 3:-1],
    ]
    titles = ["Frozen learned I", "W1 from newly fitted C", "Independent learned W2"]
    for ax, matrix, title in zip(axes, images, titles, strict=True):
        limit = max(float(np.abs(matrix).max()), 1e-8)
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Hidden neuron")
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes[0].set_ylabel("Input coordinate")
    axes[1].set_ylabel("Input coordinate")
    axes[2].set_yticks([0], ["Output"])
    fig.tight_layout()
    save_figure(fig, directory, "pattern_weights")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for row, kind in enumerate(("pooled", "free")):
        key = "learned_pool_matrix__" + kind
        for col, suffix in enumerate(("I", "W1", "W2")):
            matrix = heatmaps[key + "__" + suffix]
            limit = max(float(np.abs(matrix).max()), 1e-8)
            im = axes[row, col].imshow(
                matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto"
            )
            axes[row, col].set_title(LABELS[kind] + ": " + suffix, fontsize=10)
            axes[row, col].set_xlabel("Hidden neuron")
            fig.colorbar(im, ax=axes[row, col], shrink=0.7)
        axes[row, 0].set_ylabel("Input coordinate")
        axes[row, 1].set_ylabel("Input coordinate")
        axes[row, 2].set_yticks([0, 1], ["Output 1", "Output 2"])
    fig.suptitle("DeepSets: fixed family 0 / task 0 / noise 0.2; direct fitted values")
    fig.tight_layout()
    save_figure(fig, directory, "deepsets_weights")


def report(root, pattern, sets):
    lines = [
        "# Совместное обучение кодбука и W2 при найденном индексаторе",
        "",
        "Проверяется численная адаптация уже найденного и замороженного I. Поиск I при неизвестной W2 этим экспериментом не проверяется. Индексаторы взяты из G26/G27 без изменения; все C, W2 и состояния оптимизаторов инициализируются заново.",
        "",
        "## Метод и протокол",
        "",
        "Первый слой задаётся индексатором и удвоенным кодбуком:",
        "",
        "$$",
        r"W_{1,dh}=\operatorname{sgn}(I_{dh})c_{|I_{dh}|},\qquad c_0=0.",
        "$$",
        "",
        "Pattern 0110: два числа C, общий обучаемый скрытый bias, выходной bias. Базовая голова — общий обучаемый скаляр (всего 5 параметров), свободная W2 — восемь независимых коэффициентов (12 параметров). Одинаковые 768 support / 256 validation / 512 test, ещё 512 резервных точек. Полный куб2048 оценивается дополнительно. Три независимых набора стартов,32старта на каждый из двух LR0.01/0.03,4096вычислений градиента Adam. По64шага для обеих ветвей C/-C входят в общий бюджет. Все численные параметры обновляются совместно, L2=0.001; head penalty усредняется по коэффициентам. Выбор checkpoint/start/LR только по validation BCE.",
        "",
        "DeepSets: четыре прежние геометрии, восемь функций на геометрию. Исходная скалярная задача повторно использует support/validation G27. Дополнительные32новые векторные функции используют новые Gaussian C и новую знаковую матрицу2x2 после pooling двух локальных признаков; она повторяется по восьми точкам. Полностью свободная голова ученика имеет размер2x16. Этот teacher сохраняет инвариантность множества.",
        "",
        "Головы: фиксированная сумма; общая обучаемая матрица после pooling; независимая W2 по всем16нейронам. Группы pooled-головы выводятся только из сигнатур столбцов найденного I, без teacher point-pairing. Random-I контроль сохраняет количества кодов/нулей, но перемешивает позиции. Полностью свободная W2 сама по себе не гарантирует симметрию множества.",
        "",
        "В регрессии известный общий b=0.15 сохраняется. Неизвестный выходной сдвиг определяется наблюдаемым ответом в нуле:",
        "",
        "$$",
        r"\widehat f_t(x)=\bar y_t(0)+\sum_h a_{t,h}\left([b+(W_{1,t}^{\mathsf T}x)_h]_+-[b]_+\right).",
        "$$",
        "",
        "Используются4096support/1024validation/8192test,64свежих знаковых старта. Одновременный damped Gauss-Newton по C и W2,64шага с backtracking; validation выбирает checkpoint/start. Это совместная оптимизация, без поочерёдного фиксирования C/W2. Для половины стартов W2 инициализируется одинаковыми коэффициентами, для половины — независимыми случайными. Никакие истинные C/W2 не инициализируют learner. Шум raw0/0.2 усреднён по64ответам: SD среднего0/0.025. Опорный ответ в нуле использует262144повтора. Gaussian-средние симулируются аналитически, как в G27.",
        "",
        "Все численные предсказатели фиксируются до первого нового test/permutation-аудита. NMSE = MSE / средняя по выходам дисперсия чистых test-ответов. Перестановочная ошибка — RMS разброса выходов для16перестановок каждого из64новых множеств, делённая на test SD. Контроль исследует обучение при верном найденном I и не сравнивает весь вычислительный бюджет конкурентов.",
        "",
        "## Pattern 0110",
        "",
        "| Голова | Test BA, среднее | Худший seed | BCE | Полных успехов среди стартов |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pattern:
        lines.append(
            f"|{LABELS[row['head']]}|{100 * row['mean_test_balanced_accuracy']:.2f}%|{100 * row['worst_test_balanced_accuracy']:.2f}%|{row['mean_test_bce']:.5f}|{row['perfect_full_starts']}/{row['total_starts']}|"
        )
    lines += [
        "",
        "![Pattern loss и recall](figures/pattern_learning.png)",
        "",
        "![Прямые I, W1 и W2](figures/pattern_weights.png)",
        "",
        "## DeepSets",
        "",
        "| Задача | Raw noise | Голова | Средняя NMSE | Худшая NMSE | NMSE<1e-4 | Перестановочный RMS/SD |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in sets:
        lines.append(
            f"|{row['task']}|{row['noise']}|{LABELS[row['head']]}|{row['mean_nmse']:.3e}|{row['worst_nmse']:.3e}|{row['functions_below_1e-4']}/{row['functions']}|{row['mean_permutation_normalized_rms']:.3e}|"
        )
    lines += [
        "",
        "![DeepSets train и validation](figures/deepsets_learning.png)",
        "",
        "![Прямые индексаторы и матрицы](figures/deepsets_weights.png)",
        "",
        "## Граница результата",
        "",
        "Допускается совместно обучать C и W2 после фиксации I. Включение свободной W2 в численную фазу не обосновывает сохранение старого prefix-count поиска при неизвестном readout teacher. Подсчёт нейронов и положительные осевые суммы первоначального extractor по-прежнему имеют ограничения. Для DeepSets shared pooled-голова сохраняет архитектурную инвариантность; independent-голова может нарушать её при шуме/неполной оптимизации. Изображения, неизвестные индивидуальные hidden-bias и новый поиск I здесь не проверялись.",
        "",
        "Источники: protocol.json, data_frozen.json, cohort_frozen.json, summary.json, pattern_metrics.csv, sets_metrics.csv, code_snapshot/. Старые G26/G27 не изменялись. Тяжёлые checkpoints не создавались.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.root.resolve()
    freeze(root)
    pattern, pattern_rows = evaluate_pattern(root, args.device)
    sets, sets_rows, trajectories, heatmaps = evaluate_sets(root, args.device)
    write_csv(root / "pattern_metrics.csv", pattern_rows)
    write_csv(root / "sets_metrics.csv", sets_rows)
    figures(root, pattern_rows, trajectories, heatmaps)
    report(root, pattern, sets)
    save_json(
        root / "summary.json",
        {"evaluated_utc": timestamp(), "pattern": pattern, "sets": sets},
    )
    policy = json.loads((BASE / "GPU_POLICY.json").read_text())
    policy["active_continuation"] = None
    continuation = str(root.relative_to(BASE) / "protocol.json")
    if continuation not in policy["completed_continuations"]:
        policy["completed_continuations"].append(continuation)
    policy["last_completed_stage"] = (
        "G28 frozen-I joint C/W2 numerical controls on Pattern0110 and DeepSets; new summary/report. No unknown-readout I-search claim."
    )
    policy["current_execution"] = (
        "G28 scientific fits and held-out evaluation complete. No next experiment scheduled. GPUs6/7 remained reserved."
    )
    save_json(BASE / "GPU_POLICY.json", policy)
    print(json.dumps({"pattern": pattern, "sets": sets}), flush=True)


if __name__ == "__main__":
    main()
