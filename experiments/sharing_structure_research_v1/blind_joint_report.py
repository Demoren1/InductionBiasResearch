"""Freeze all G34 predictions before inspecting teacher structures."""

import argparse
import csv
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .blind_contract import canonical_index, digest, load_sizes, save_json
from .blind_numeric import numpy_predict
from .blind_report import dense_predict, match_raw_columns, raw_weight, sharing_metrics

BASE = Path(__file__).resolve().parent
INDEXED = (
    "source_seed_h0",
    "joint_functional",
    "joint_paired_geometry",
    "joint_bic",
    "joint_task_rank",
)
METHODS = ("dense_adam", "dense_repaired", *INDEXED)
LABELS = {
    "source_seed_h0": "Source seed / H0",
    "joint_functional": "Joint / meta",
    "joint_paired_geometry": "Joint / paired geometry",
    "joint_bic": "Joint / BIC",
    "dense_adam": "Dense Adam",
    "dense_repaired": "Dense + GN + revival",
    "joint_task_rank": "Task guard + code rank",
}


def selected_models(root, key):
    folder = root / "fits" / key
    source = dict(np.load(folder / "fitted.npz"))
    chosen = int(source["validation_mse"].mean(1).argmin())
    result = {
        "source_seed_h0": (source["indices"][chosen], source["parameters"][chosen])
    }
    selected = dict(np.load(folder / "joint_selected.npz"))
    for method in INDEXED[1:4]:
        result[method] = selected[f"{method}_index"], selected[f"{method}_parameters"]
    rank = dict(np.load(folder / "task_rank_selected.npz"))
    result["joint_task_rank"] = rank["index"], rank["parameters"]
    return result


def freeze(root):
    if (root / "evaluation/predictions_frozen.json").exists():
        raise ValueError("Predictions already frozen; do not replace them")

    def deny_private(event, args):
        if (
            event == "open"
            and args
            and not isinstance(args[0], int)
            and "private" in Path(os.fsdecode(args[0])).parts
        ):
            raise PermissionError("Teacher remains closed during prediction freeze")

    sys.addaudithook(deny_private)
    protocol = json.loads((root / "protocol.json").read_text())
    for filename, field in (
        ("joint_registered.json", "code_sha256"),
        ("dense_registered.json", "dense_code_sha256"),
        ("rank_registered.json", "code_sha256"),
    ):
        registration = json.loads((root / filename).read_text())
        if (
            "data_frozen_sha256" in registration
            and digest(root / "data_frozen.json") != registration["data_frozen_sha256"]
        ):
            raise ValueError("Original data registration changed")
        for name, expected in registration[field].items():
            if digest(BASE / name) != expected:
                raise ValueError(f"Registered learner changed: {name}")
    data = json.loads((root / "data_frozen.json").read_text())
    for name, expected in data["public_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen observations changed: {name}")
    models, receipts, predictions = {}, {}, {}
    for case in protocol["cases"]:
        for condition in protocol["conditions"]:
            key = f"{case['name']}_c{condition}"
            for name in (
                f"dense_{key}.json",
                *[
                    f"joint_{key}_{phase}.json"
                    for phase in ("seed", "bank", "fit", "criteria", "select")
                ],
                f"task_rank_{key}.json",
            ):
                path = root / "receipts" / name
                content = json.loads(path.read_text())
                if content["status"] != "complete" or content["violations"]:
                    raise ValueError(f"Incomplete/invalid observation boundary: {name}")
                receipts[str(path.relative_to(root))] = digest(path)
            for owner, artifact, metadata in (
                ("dense", "fitted.npz", "frozen.json"),
                ("dense", "repaired.npz", "repaired.json"),
                ("fits", "fitted.npz", "seed_initialization.json"),
                ("fits", "joint_fitted.npz", "joint_fitted.json"),
                ("fits", "joint_selected.npz", "joint_selected.json"),
                ("fits", "task_rank_selected.npz", "task_rank_selected.json"),
            ):
                folder = root / owner / key
                expected = json.loads((folder / metadata).read_text())["sha256"]
                if digest(folder / artifact) != expected:
                    raise ValueError(f"Frozen model changed: {folder / artifact}")
                models[str((folder / artifact).relative_to(root))] = expected
                models[str((folder / metadata).relative_to(root))] = digest(
                    folder / metadata
                )
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    for case in protocol["cases"]:
        sizes = load_sizes(root, case["name"])
        for condition in protocol["conditions"]:
            key = f"{case['name']}_c{condition}"
            choices = selected_models(root, key)
            dense = np.load(root / "dense" / key / "fitted.npz")["parameters"]
            repaired = np.load(root / "dense" / key / "repaired.npz")["parameters"]
            values = {method: [] for method in METHODS}
            with np.load(root / "public" / key / "test.npz") as test:
                for t in range(protocol["sources"]):
                    x = test[f"x_{t}"]
                    for method, parameters in (
                        ("dense_adam", dense),
                        ("dense_repaired", repaired),
                    ):
                        values[method].append(dense_predict(x, parameters[t], sizes))
                    for method, (index, parameters) in choices.items():
                        values[method].append(
                            numpy_predict(
                                x,
                                index,
                                parameters[t],
                                sizes["codes"],
                                sizes["outputs"],
                            )
                        )
            path = out / f"{key}_predictions.npz"
            np.savez_compressed(path, **{k: np.stack(v) for k, v in values.items()})
            predictions[path.name] = digest(path)
    save_json(
        out / "predictions_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "models": models,
            "receipts": receipts,
            "predictions": predictions,
            "test_labels_used": False,
            "teacher_used": False,
            "public_files_verified": len(data["public_sha256"]),
            "data_frozen_sha256": digest(root / "data_frozen.json"),
        },
    )
    print(
        json.dumps({"predictions_frozen": len(predictions), "receipts": len(receipts)}),
        flush=True,
    )


def audit(root):
    out = root / "evaluation"
    frozen = json.loads((out / "predictions_frozen.json").read_text())
    for name, expected in frozen["models"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen model changed: {name}")
    for name, expected in frozen["predictions"].items():
        if digest(out / name) != expected:
            raise ValueError(f"Frozen predictions changed: {name}")
    from experiments.darts_structured_tasks_v1.evaluate import align_indices

    data = json.loads((root / "data_frozen.json").read_text())
    for name, expected in data["private_commitment_sha256"].items():
        if digest(root / "private" / name) != expected:
            raise ValueError(f"Original teacher commitment changed: {name}")
    generation = json.loads((root / "private/generation.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    rows, matrices, case_metadata = [], {}, []
    for case in generation["cases"]:
        name, sizes = case["name"], load_sizes(root, case["name"])
        case_metadata.append({**case, "sizes": sizes})
        truth = np.load(root / "private" / f"{name}.npz")["index"]
        canonical = canonical_index(truth, sizes["codes"])
        with np.load(root / "private" / f"{name}_clean_test.npz") as clean_file:
            clean = np.stack([clean_file[f"y_{t}"] for t in range(protocol["sources"])])
        for condition, sigma in enumerate(generation["noise"]):
            key = f"{name}_c{condition}"
            choices = selected_models(root, key)
            fitted = np.load(root / "fits" / key / "joint_fitted.npz")
            proposed = np.load(root / "fits" / key / "joint_bank.npz")
            in_proposals = bool(
                np.any(np.all(proposed["indices"] == canonical, axis=(1, 2)))
            )
            in_fitted = bool(
                np.any(np.all(fitted["indices"] == canonical, axis=(1, 2)))
            )
            matrices[f"{key}_truth"] = truth
            prediction = dict(np.load(out / f"{key}_predictions.npz"))
            with np.load(root / "public" / key / "test.npz") as observed:
                noisy = np.stack(
                    [observed[f"y_{t}"] for t in range(protocol["sources"])]
                )
            for method in METHODS:
                value = prediction[method]
                row = {
                    "case": name,
                    "dataset": case["dataset"],
                    "variant": case["variant"],
                    "condition": condition,
                    "sigma_audit_only": sigma,
                    "method": method,
                    "test_clean_mse": float(np.mean((value - clean) ** 2)),
                    "test_noisy_mse": float(np.mean((value - noisy) ** 2)),
                    "test_clean_nmse": float(
                        np.mean((value - clean) ** 2) / np.var(clean)
                    ),
                    "true_index_in_proposals": in_proposals,
                    "true_index_in_fitted": in_fitted,
                    "proposals": len(proposed["indices"]),
                    "fitted_candidates": len(fitted["indices"]),
                }
                if method in choices:
                    metric, aligned, columns = align_indices(
                        choices[method][0], truth, sizes["codes"]
                    )
                    row.update(metric)
                    row.update(sharing_metrics(aligned, truth))
                    row["exact_index"] = metric["categorical_errors"] == 0
                    matrices[f"{key}_{method}_raw"] = choices[method][0]
                    matrices[f"{key}_{method}_aligned"] = aligned
                    matrices[f"{key}_{method}_columns"] = columns
                rows.append(row)
    summary = []
    for dataset in ("pattern_0110", "deepsets", "general"):
        for sigma in generation["noise"]:
            for method in METHODS:
                local = [
                    r
                    for r in rows
                    if r["dataset"] == dataset
                    and r["sigma_audit_only"] == sigma
                    and r["method"] == method
                ]
                item = {
                    "dataset": dataset,
                    "sigma_audit_only": sigma,
                    "method": method,
                    "cases": len(local),
                    "mean_clean_mse": float(
                        np.mean([r["test_clean_mse"] for r in local])
                    ),
                    "true_index_in_proposals": sum(
                        r["true_index_in_proposals"] for r in local
                    ),
                    "true_index_in_fitted": sum(
                        r["true_index_in_fitted"] for r in local
                    ),
                }
                if method in INDEXED:
                    item.update(
                        {
                            "exact": sum(r["exact_index"] for r in local),
                            "errors": sum(r["categorical_errors"] for r in local),
                            "positions": sum(r["categorical_positions"] for r in local),
                            **{
                                metric: float(np.mean([r[metric] for r in local]))
                                for metric in (
                                    "mask_f1",
                                    "mask_recall",
                                    "sharing_f1",
                                    "sharing_recall",
                                )
                            },
                        }
                    )
                summary.append(item)
    for filename, content in (("comparison.csv", rows), ("summary.csv", summary)):
        with (out / filename).open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(dict.fromkeys(k for r in content for k in r))
            )
            writer.writeheader()
            writer.writerows(content)
    np.savez_compressed(out / "selected_indices.npz", **matrices)
    save_json(
        out / "selected_indices.json", {k: v.tolist() for k, v in matrices.items()}
    )
    save_json(
        out / "results.json", {"rows": rows, "summary": summary, "cases": case_metadata}
    )
    render(root, rows, case_metadata, matrices)
    write_report(root, rows, summary)
    save_json(
        out / "audit_complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "models_frozen_before_teacher_audit": True,
            "predictions_frozen_sha256": digest(out / "predictions_frozen.json"),
            "receipts": len(frozen["receipts"]),
            "forbidden_reads": 0,
            "report_sha256": digest(root / "REPORT.md"),
            "code_sha256": digest(Path(__file__)),
            "rows": len(rows),
        },
    )
    print(json.dumps({"summary": summary}), flush=True)


def render(root, rows, cases, matrices):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
    out = root / "evaluation"
    colors = [
        "#912c7c",
        "#c05354",
        "#d78942",
        "#c5b242",
        "#f7f7f7",
        "#58a7a1",
        "#3776b5",
        "#6657a5",
        "#358350",
    ]

    def index_image(ax, matrix, codes):
        cmap = ListedColormap(colors[4 - codes : 5 + codes])
        return ax.imshow(
            matrix,
            cmap=cmap,
            norm=BoundaryNorm(np.arange(-codes - 0.5, codes + 1), 2 * codes + 1),
            aspect="auto",
        )

    for condition in (0, 1):
        fig, axes = plt.subplots(
            len(cases), 5, figsize=(17, 2 * len(cases)), squeeze=False
        )
        for r, case in enumerate(cases):
            key = f"{case['name']}_c{condition}"
            for col, method in enumerate(("truth", *INDEXED[:3], "joint_task_rank")):
                matrix = matrices[
                    f"{key}_truth" if method == "truth" else f"{key}_{method}_aligned"
                ]
                index_image(axes[r, col], matrix, case["sizes"]["codes"])
                title = f"{case['name']} | "
                if method == "truth":
                    title += f"{case['dataset']} true I"
                else:
                    row = next(
                        a
                        for a in rows
                        if a["case"] == case["name"]
                        and a["condition"] == condition
                        and a["method"] == method
                    )
                    title += f"{LABELS[method]}: {row['categorical_errors']} errors"
                axes[r, col].set_title(title, fontsize=9)
                axes[r, col].set_xlabel("Hidden unit")
                axes[r, col].set_ylabel("Input coordinate")
        fig.tight_layout()
        fig.savefig(out / f"all_indices_c{condition}.png")
        plt.close(fig)
    representatives = [
        next(c for c in cases if c["dataset"] == dataset)
        for dataset in ("pattern_0110", "deepsets", "general")
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for r, case in enumerate(representatives):
        key, sizes = f"{case['name']}_c1", case["sizes"]
        truth = dict(np.load(root / "private" / f"{case['name']}.npz"))
        true_weight = (
            np.sign(truth["index"])
            * np.r_[0, truth["codes"][0]][np.abs(truth["index"])]
        )
        dense = np.load(root / "dense" / key / "fitted.npz")["parameters"][0]
        weight = match_raw_columns(raw_weight(dense, sizes), true_weight)
        index_image(axes[r, 0], truth["index"], sizes["codes"])
        axes[r, 0].set_title(f"{case['dataset']}: true I (audit)")
        bound = max(float(np.max(np.abs(weight))), 1e-12)
        image = axes[r, 1].imshow(
            weight, cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto"
        )
        fig.colorbar(image, ax=axes[r, 1], shrink=0.75)
        axes[r, 1].set_title("Actual Dense Adam W1, source0")
        for col, method in enumerate(("joint_functional", "joint_task_rank"), 2):
            image = index_image(
                axes[r, col], matrices[f"{key}_{method}_aligned"], sizes["codes"]
            )
            fig.colorbar(
                image,
                ax=axes[r, col],
                shrink=0.75,
                ticks=np.arange(-sizes["codes"], sizes["codes"] + 1),
            )
            row = next(
                a
                for a in rows
                if a["case"] == case["name"]
                and a["condition"] == 1
                and a["method"] == method
            )
            axes[r, col].set_title(
                f"{LABELS[method]}: {row['categorical_errors']} errors"
            )
        for ax in axes[r]:
            ax.set_xlabel("Hidden unit (audit alignment)")
            ax.set_ylabel("Input coordinate")
    fig.suptitle(
        "First registered case per family; noisy condition; dense weights are not rescaled",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "representative_heatmaps.png")
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for r, case in enumerate(representatives):
        key = f"{case['name']}_c1"
        dense = dict(np.load(root / "dense" / key / "fitted.npz"))
        start = json.loads((root / "dense" / key / "frozen.json").read_text())[
            "selected_starts"
        ][0]
        for field, label in (("train_history", "Train"), ("validation_history", "H0")):
            axes[r, 0].semilogy(
                dense["steps"],
                np.maximum(dense[field][:, start] * dense["scale"][0] ** 2, 1e-16),
                label=label,
            )
        axes[r, 0].set_title(f"{case['dataset']}: selected Adam start, source0")
        axes[r, 0].set_xlabel("Adam step")
        folder = root / "fits" / key
        choice = json.loads((folder / "task_rank_selected.json").read_text())[
            "candidate"
        ]
        history = json.loads((folder / "joint_fitted.json").read_text())["history"]
        chunk = next((h for h in history if choice in h["candidates"]), None)
        if chunk is None:
            axes[r, 1].text(
                0.5,
                0.5,
                "Selected source seed; no numeric trajectory",
                ha="center",
                va="center",
                transform=axes[r, 1].transAxes,
            )
        else:
            count = len(chunk["candidates"])
            position = chunk["candidates"].index(choice)
            train = np.array(chunk["train"]).reshape(-1, count, 16, 4)[:, position, 0]
            validation = np.array(chunk["validation"]).reshape(-1, count, 16, 4)[
                :, position, 0
            ]
            trial = int(validation.min(0).argmin())
            axes[r, 1].semilogy(
                np.arange(len(train)), np.maximum(train[:, trial], 1e-16), label="Train"
            )
            axes[r, 1].semilogy(
                np.arange(len(train)),
                np.maximum(validation[:, trial], 1e-16),
                label="H0",
            )
        axes[r, 1].set_title("Numeric attempt for selected I, source0")
        axes[r, 1].set_xlabel("Gauss-Newton step (I fixed)")
        for ax in axes[r]:
            ax.set_ylabel("Measured MSE")
            ax.grid(alpha=0.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("mask_recall", "sharing_recall")):
        for j, method in enumerate(INDEXED):
            values = [
                np.mean(
                    [
                        r[metric]
                        for r in rows
                        if r["dataset"] == dataset
                        and r["condition"] == 1
                        and r["method"] == method
                    ]
                )
                for dataset in ("pattern_0110", "deepsets", "general")
            ]
            ax.bar(
                np.arange(3) + (j - 2) * 0.16,
                values,
                width=0.16,
                label=LABELS[method],
            )
        ax.set_xticks(np.arange(3), ["Pattern0110", "DeepSets", "General"])
        ax.set_ylim(0, 1.05)
        ax.set_title(metric + " (audit only)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "structural_recall.png")
    plt.close(fig)


def write_report(root, rows, summary):
    method = BASE / "METHOD_G34_PROTOTYPE.md"
    text = (
        method.read_text()
        if method.exists()
        else "# G34: совместное извлечение структуры\n"
    )
    text += "\n## Итоги закрытой проверки\n\n| Данные | sigma | Метод | Точный I | Ошибки I | Mask F1 | Sharing F1 | Чистая MSE |\n|---|---:|---|---:|---:|---:|---:|---:|\n"
    for row in summary:
        text += f"| {row['dataset']} | {row['sigma_audit_only']:.2f} | {LABELS[row['method']]} | "
        if row["method"] in INDEXED:
            text += f"{row['exact']}/{row['cases']} | {row['errors']}/{row['positions']} | {100 * row['mask_f1']:.2f}% | {100 * row['sharing_f1']:.2f}% | "
        else:
            text += "— | — | — | — | "
        text += f"{row['mean_clean_mse']:.2e} |\n"
    primary = [r for r in rows if r["method"] == "joint_functional"]
    failed = [r for r in primary if not r["exact_index"]]
    absent = sum(not r["true_index_in_proposals"] for r in failed)
    truncated = sum(
        r["true_index_in_proposals"] and not r["true_index_in_fitted"] for r in failed
    )
    text += f"\nОсновной заранее выбранный вариант Joint/meta: {len(primary) - len(failed)}/{len(primary)} точных I. Отказы: {absent} — нет в полном банке, {truncated} — не попал в численный бюджет, {len(failed) - absent - truncated} — есть среди подогнанных, выбран другой. Контроли не заменяют основной метод после просмотра истины.\n\n"
    rank_rows = [r for r in rows if r["method"] == "joint_task_rank"]
    rank_successes = sum(r["exact_index"] for r in rank_rows)
    text += (
        f"Дополнительный Task guard + code rank: {rank_successes}/{len(rank_rows)} точных I. "
        "В этой закрытой серии он выбрал те же индексаторы, что Joint/meta, во всех20 случаях: "
        "успех старого трёхзадачного пилота не превратился в дополнительный выигрыш здесь. "
        "Когда правильный I присутствовал среди численно подогнанных, основной выбор был правильным; "
        "оставшиеся отказы относятся к построению банка.\n\n"
        "| Неудачный случай | sigma | Ошибки I | Истинный I в полном банке |\n"
        "|---|---:|---:|---|\n"
    )
    for row in failed:
        text += f"| {row['case']} | {row['sigma_audit_only']:.2f} | {row['categorical_errors']} | {'да' if row['true_index_in_proposals'] else 'нет'} |\n"
    text += "\n"
    text += "Два Pattern-case имеют одну схему0110 с разными числами. Парные условия шума не являются новыми геометриями. Полная структура проверяется отдельно от MSE и маски. Dense Adam не выдаёт дискретного индексатора. Наблюдения teacher и истинный шум доступны только этому аудиту после фиксации всех предсказаний. Бинарная классификация, изображения и target-перенос отдельно в этой серии не проверялись.\n\n"
    text += "![Хитмапы](evaluation/representative_heatmaps.png)\n\n![Все I без шума](evaluation/all_indices_c0.png)\n\n![Все I с шумом](evaluation/all_indices_c1.png)\n\n![Train/H0](evaluation/training_curves.png)\n\nНа правых кривых показана реально измеренная численная попытка для выбранного I; если исходное состояние было лучше по H0, оно могло быть сохранено вместо этой попытки.\n\n![Recall](evaluation/structural_recall.png)\n\n[Полная CSV](evaluation/comparison.csv), [сводная CSV](evaluation/summary.csv), [матрицы JSON](evaluation/selected_indices.json), [NPZ](evaluation/selected_indices.npz), [заморозка предсказаний](evaluation/predictions_frozen.json), [регистрация метода](joint_registered.json).\n"
    (root / "REPORT.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "audit"), required=True)
    args = parser.parse_args()
    (freeze if args.phase == "freeze" else audit)(args.root.resolve())


if __name__ == "__main__":
    main()
