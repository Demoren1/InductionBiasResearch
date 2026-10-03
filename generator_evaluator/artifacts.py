"""Atomic artifacts, per-task paired comparisons and explained diagnostics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def _atomic_write(path: Path, write_payload, *, binary: bool) -> None:
    """Write through an owned, unique sibling file and replace the destination."""
    temporary = None
    descriptor = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        if binary:
            stream = os.fdopen(descriptor, "wb")
        else:
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = None  # The stream now owns it.
        with stream:
            write_payload(stream)
            stream.flush()
        os.replace(temporary, path)
    except OSError as exc:
        temporary_context = f" (temporary file {temporary})" if temporary else ""
        reason = exc.strerror or str(exc)
        raise OSError(
            exc.errno,
            f"Failed to atomically write artifact {path}{temporary_context}: {reason}",
            exc.filename,
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to atomically write artifact {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                # Keep the original write/replace error as the primary failure.
                pass


def save_json(path: Path, payload) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2,
                             allow_nan=False) + "\n"
    _atomic_write(path, lambda stream: stream.write(serialized), binary=False)


def save_torch(path: Path, payload) -> None:
    # Banks stay ordinary in-memory objects. On disk, unchanged teacher rows
    # are shared across inputs, live/best checkpoints and frozen artifacts.
    from .bank_storage import externalize_banks
    path = Path(path)
    stored = externalize_banks(payload, path)
    _atomic_write(path, lambda stream: torch.save(stored, stream), binary=True)


def paired_comparison(losses, dense_losses) -> dict:
    """Descriptive paired Student interval across fresh initializations only."""
    from scipy.stats import t
    delta = np.asarray(losses, dtype=float) - np.asarray(dense_losses, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("paired losses must be nonempty finite vectors")
    mean = float(delta.mean())
    half = (float(t.ppf(.975, len(delta)-1) * delta.std(ddof=1) / len(delta)**.5)
            if len(delta) > 1 else None)
    return dict(delta=mean, paired_deltas=delta.tolist(),
                ci95=None if half is None else [mean-half, mean+half],
                point_improvement=mean < 0,
                interval_below_zero=half is not None and mean+half < 0,
                uncertainty_unit="fresh initialization on one fixed support/query task",
                independent_initializations=len(delta))


def write_plots(out: Path, history: list[dict], examples: list[dict], *,
                evaluator_history=None, calibration=None, within_task_selection=False,
                generator_pretraining=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    evaluator_history = evaluator_history or []
    calibration = calibration or []
    generator_pretraining = generator_pretraining or []
    save_json(folder / "source_values.json", dict(generator=history, evaluator=evaluator_history,
                                                calibration=calibration,
                                                generator_pretraining=generator_pretraining))
    for filename, rows, metrics in (
            ("generator_reconstruction", generator_pretraining,
             (("reconstruction_loss", "Reconstruction loss"),
              ("reconstruction_overlap", "Reconstructed mask overlap"))),
            ("generator_agreement", history,
             (("direct_agreement_overlap", "Aligned exact-K mask overlap"),
              ("direct_agreement_loss", "Direct agreement loss"),
              ("direct_agreement_weight", "Agreement weight")))):
        if not any(key in row for row in rows for key, _ in metrics):
            continue
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))
        for axis, (key, label) in zip(axes, metrics):
            roles = list(dict.fromkeys(row.get("pattern", "generator") for row in rows))
            for role in roles:
                values = [row[key] for row in rows if row.get("pattern", "generator") == role and key in row]
                if values:
                    axis.plot(values, label=str(role))
            axis.set(xlabel="Update", ylabel=label)
            axis.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"{filename}.{ext}")
        plt.close(fig)
    if history:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        cost_label = "Own pattern predicted delta vs dense" if within_task_selection else "Worst predicted delta vs dense"
        for axis, key, label in zip(axes, ("predicted_cost", "policy_gradient_loss", "permutation_loss"),
                                    (cost_label, "Policy gradient surrogate", "Permutation logits MSE")):
            points = [(i, row[key]) for i, row in enumerate(history) if key in row]
            if points:
                axis.plot(*zip(*points))
            axis.set(xlabel="Generator update", ylabel=label)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"generator_losses.{ext}")
        plt.close(fig)
    evaluator_events = [event for stage in evaluator_history for event in stage]
    if evaluator_events:
        fig, axis = plt.subplots(figsize=(7, 4))
        selection_label = "Selection queries" if within_task_selection else "New tasks"
        joint_label = "Selection queries and new topologies" if within_task_selection else "New tasks and topologies"
        for key, label in (("train_mse", "Train"), ("mask_validation_mse", "New topologies"),
                           ("meta_validation_mse", selection_label), ("joint_validation_mse", joint_label)):
            points = [(i, event[key]) for i, event in enumerate(evaluator_events) if key in event]
            if points:
                axis.plot(*zip(*points), label=label)
        axis.set(xlabel="Evaluator epoch (across refreshes)", ylabel="Measured-target prediction MSE")
        axis.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"evaluator_losses.{ext}")
        plt.close(fig)
    if calibration:
        actual = [row["actual"] for row in calibration]
        predicted = [row["predicted"] for row in calibration]
        std = [row["std"] for row in calibration]
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.errorbar(actual, predicted, yerr=std, fmt="o", alpha=.65)
        low, high = min(actual + predicted), max(actual + predicted)
        axis.plot([low, high], [low, high], "--", color="black")
        axis.set(xlabel="Real terminal query error", ylabel="Ensemble prediction ± std")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"surrogate_calibration.{ext}")
        plt.close(fig)
    for index, example in enumerate(examples):
        mask = torch.as_tensor(example["mask"]).cpu()
        weights = example["effective_weights"]
        effective = (torch.stack(weights) if isinstance(weights, list) else torch.as_tensor(weights)).cpu()
        while effective.ndim > 2:
            effective = effective[0]
        fig, axes = plt.subplots(1, 2, figsize=(9, 5))
        axes[0].imshow(mask, aspect="auto", cmap="Greys", vmin=0, vmax=1)
        bound = max(float(effective.abs().max()), 1e-8)
        artist = axes[1].imshow(effective, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
        fig.colorbar(artist, ax=axes[1], label="Signed effective weight")
        for axis in axes:
            axis.set(xlabel="Hidden neuron", ylabel="Input feature")
        axes[0].set_title(f"{example['method']}: binary mask")
        axes[1].set_title("Terminal weights, initialization 0")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"mask_weights_{index}_{example['method']}.{ext}")
        plt.close(fig)
    if examples and all("history" in example for example in examples):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for example in examples:
            child_history = example["history"]
            if isinstance(child_history, list):
                steps = [row["step"] for row in child_history[0]]
                support = np.asarray([[row["support_objective"] for row in replica] for replica in child_history]).mean(0)
                query = np.asarray([[row["query_bce"] for row in replica] for replica in child_history]).mean(0)
            else:
                steps = torch.as_tensor(child_history["steps"]).tolist()
                support = torch.as_tensor(child_history["support_objective"]).flatten(1).mean(1).numpy()
                query = torch.as_tensor(child_history["queryNMSE"]).flatten(1).mean(1).numpy()
            axes[0].plot(steps, support, label=example["method"])
            axes[1].plot(steps, query, label=example["method"])
        axes[0].set(xlabel="Support-only optimizer step", ylabel="Support objective (including L2)")
        axes[1].set(xlabel="Support-only optimizer step", ylabel="Query error (diagnostic only)")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(folder / f"selected_child_losses.{ext}")
        plt.close(fig)
    (folder / "CAPTIONS_RU.md").write_text(
        "# Пояснения к графикам\n\n"
        "Кривые генератора: горизонтальная ось — номер обновления. Слева — худшая по задачам "
        "предсказанная разность ошибки с dense; отрицательное значение означает только прогноз выигрыша. "
        "В центре — численное значение score-function surrogate, не настоящая query-ошибка. "
        "Справа — MSE logits при совместных перестановках банка с одинаковым шумом. "
        "При инвариантной архитектуре оно близко к нулю по построению.\n\n"
        "Кривые оценщика: ось X — номер его эпохи с учётом последующих дообучений. "
        "Ось Y — MSE между прогнозом ensemble и настоящими terminal-query метками. "
        "Отдельно показаны train, новые топологии, новые задачи и их совместный holdout. "
        "Validation-кривые не участвуют в обновлении параметров оценщика.\n\n"
        "Calibration: ось X — реально измеренная query-ошибка, ось Y — прогноз до добавления новой метки. "
        "Вертикальные отрезки показывают один population std ансамбля, а не доверительный интервал. "
        "Пунктир — линия точного совпадения; расхождения диагностируют ошибку суррогата.\n\n"
        "Кривые выбранных дочерних сетей: X — число support-only шагов. Слева — support objective "
        "с регуляризацией, справа — диагностическая query-ошибка. Показано среднее по новым "
        "инициализациям первой test-задачи; query не выбирает checkpoint.\n\n"
        "Heatmaps: строки — фиксированные входные признаки, столбцы — hidden-нейроны. "
        "Слева показана бинарная маска; справа — фактические signed $W\\odot M$ терминальной сети "
        "для первой новой инициализации. Красный цвет положительный, синий отрицательный; "
        "масштаб весов симметричен относительно нуля для каждого изображения. "
        "Полные веса всех инициализаций сохранены в child artifacts.\n\n"
        "Короткая smoke-проверка не устанавливает полезность генератора, сходимость или перенос. "
        "Интервалы в summary.json характеризуют лишь разброс новых инициализаций "
        "на фиксированных данных, а не неопределённость по новым задачам.\n", encoding="utf-8")
