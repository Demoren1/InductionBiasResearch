"""Resumable five-pair MNIST8m bank, VAE, agreement, and transfer experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from pattern.mnist8m_night_transfer import plot_masks


PAIRS = ((0, 6), (1, 7), (2, 5), (3, 8), (4, 9))
DENSITIES = (0.02, 0.05)
STAGES = ("bank", "prepare", "align", "vae", "search", "evaluate", "transfer",
          "strict", "report")


@dataclass(frozen=True)
class Job:
    label: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    log: Path


def pair_name(pair: tuple[int, int]) -> str:
    return f"pair{pair[0]}{pair[1]}"


def density_name(density: float) -> str:
    return f"{round(density * 100)}pct"


def raw_command(folder: Path, pair: tuple[int, int], density: float,
                stage: str, *extra: str) -> tuple[str, ...]:
    return (sys.executable, "-m", "pattern.mnist8m_raw_mlp_bce", "--out",
            str(folder), "--digits", str(pair[0]), str(pair[1]), "--density",
            str(density), "--hidden", "64", "--stage", stage, *extra)


def utilization(gpu: int) -> int:
    result = subprocess.check_output(
        ("nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader"),
        text=True,
    )
    rows = {int(parts[0]): int(parts[1].strip().split()[0])
            for line in result.splitlines() if (parts := line.split(","))}
    if gpu not in rows:
        raise ValueError(f"GPU {gpu} is not present in nvidia-smi: {rows}")
    return rows[gpu]


def run_jobs(jobs: list[Job], gpus: list[int], stage: str, *, dry_run: bool,
             cpu_only: bool = False) -> None:
    if dry_run:
        reused = sum(all(path.exists() for path in job.outputs) for job in jobs)
        print(f"{stage}: {len(jobs)} jobs, {reused} already complete", flush=True)
        for job in jobs[:2]:
            print("  ", " ".join(job.command), flush=True)
        return
    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        pending.put(job)
    errors: list[str] = []
    lock = threading.Lock()
    progress = tqdm(total=len(jobs), desc=stage, unit="job")

    def worker(gpu: int | None) -> None:
        while True:
            if gpu is not None:
                try:
                    while utilization(gpu) != 0:
                        if pending.empty():
                            return
                        time.sleep(10)
                except Exception as exc:
                    with lock:
                        errors.append(f"GPU {gpu}: {exc!r}")
                    return
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            if all(path.exists() for path in job.outputs):
                with lock:
                    progress.update()
                    progress.set_postfix_str(f"reuse {job.label}")
                pending.task_done()
                continue
            try:
                if gpu is not None:
                    if utilization(gpu) != 0:
                        pending.put(job)
                        pending.task_done()
                        time.sleep(10)
                        continue
                job.log.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = "2"
                env["MKL_NUM_THREADS"] = "2"
                if gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                with job.log.open("w") as handle:
                    result = subprocess.run(job.command, env=env, stdout=handle,
                                            stderr=subprocess.STDOUT, check=False)
                error = (f"{job.label}: exit={result.returncode}, log={job.log}"
                         if result.returncode or not all(p.exists() for p in job.outputs)
                         else None)
            except Exception as exc:
                error = f"{job.label}: {exc!r}, log={job.log}"
            with lock:
                if error:
                    errors.append(error)
                progress.update()
                progress.set_postfix_str(f"{job.label} on {'CPU' if gpu is None else f'GPU {gpu}'}")
            pending.task_done()

    slots: list[int | None] = ([None] * min(8, max(1, os.cpu_count() // 2))
                               if cpu_only else list(gpus))
    threads = [threading.Thread(target=worker, args=(slot,), daemon=True)
               for slot in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    progress.close()
    if errors:
        raise RuntimeError(f"{stage} failed:\n" + "\n".join(errors))


def source_folder(out: Path, pair: tuple[int, int], reuse_pair38: Path) -> Path:
    return reuse_pair38 if pair == (3, 8) else out / "bank20" / pair_name(pair)


def prepared_folder(out: Path, pair: tuple[int, int], density: float) -> Path:
    return out / "prepared" / pair_name(pair) / density_name(density)


def anchored_folder(out: Path, pair: tuple[int, int], density: float,
                    task: int, index: int) -> Path:
    return out / "runs" / pair_name(pair) / density_name(density) / f"anchor_t{task}_i{index}"


def anchors(bank: Path, per_task: int) -> list[tuple[int, int]]:
    result = []
    for task in (0, 1):
        payload = torch.load(bank / f"bank_task{task}.pt", map_location="cpu",
                             weights_only=True, mmap=True)
        count = len(payload["importance"])
        order = torch.randperm(count, generator=torch.Generator().manual_seed(3130 + task))
        n_val = max(32, round(.15 * count))
        if n_val + per_task > count:
            raise ValueError(f"only {count - n_val} training maps in {bank}")
        result.extend((task, int(index)) for index in order[n_val:n_val + per_task])
    return result


def validate_banks(args: argparse.Namespace) -> None:
    for pair in PAIRS:
        source = source_folder(args.out, pair, args.reuse_pair38)
        for task, digit in enumerate(pair):
            path = source / f"bank_task{task}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            settings = payload["settings"]
            if (payload["digit"] != digit or settings.get("candidates") != args.bank_candidates or
                    settings.get("max_steps") != args.bank_steps or
                    settings.get("top_fraction") != .1 or
                    settings.get("visible_digits") is not None):
                raise ValueError(f"bank protocol mismatch: {path}")
        for density in DENSITIES:
            prepared = prepared_folder(args.out, pair, density)
            for task, digit in enumerate(pair):
                path = prepared / f"bank_task{task}.pt"
                payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                expected = round(density * 784 * 64)
                if (payload["digit"] != digit or
                        payload["settings"].get("derived_connections") != expected or
                        payload["settings"].get("visible_digits") is not None or
                        int(payload["masks"][0].sum()) != expected):
                    raise ValueError(f"prepared bank protocol mismatch: {path}")


def save_manifest(out: Path, args: argparse.Namespace, configurations: list[dict]) -> None:
    manifest = {
        "pairs": PAIRS, "densities": DENSITIES,
        "anchors_per_task": args.anchors_per_task,
        "bank_candidates": args.bank_candidates,
        "bank_steps": args.bank_steps,
        "vae_epochs": args.vae_epochs,
        "search_steps": args.search_steps,
        "evaluation_steps": args.evaluation_steps,
        "transfer_steps": args.transfer_steps,
        "reuse_pair38": str(args.reuse_pair38.resolve()),
        "configurations": configurations,
    }
    path = out / "manifest.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != json.loads(json.dumps(manifest)):
            raise ValueError(f"resume settings differ from {path}; use a new --out")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def summary(out: Path, configs: list[dict]) -> None:
    rows = []
    for config in tqdm(configs, desc="summarize", unit="run"):
        folder = Path(config["folder"])
        path = folder / "evaluation.pt"
        if not path.exists():
            continue
        result = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        names = result["names"]
        metrics = result["metrics"]
        diagnostics_path = folder / "alignment_diagnostics.json"
        diagnostics = (json.loads(diagnostics_path.read_text())
                       if diagnostics_path.exists() else [])
        for task, digit in enumerate(config["pair"]):
            def mean(metric: str, name: str) -> float:
                return float(metrics[metric][task, names.index(name)].mean())

            row = {"pair": config["pair"], "density": config["density"],
                   "anchor_task": config["anchor_task"],
                   "anchor_index": config["anchor_index"], "digit": digit}
            diagnostic = next((item for item in diagnostics if item["task"] == task), None)
            if diagnostic is not None:
                row["alignment_iou"] = {
                    "raw": diagnostic["raw_heldout_mean_iou"],
                    "aligned": diagnostic["aligned_heldout_mean_iou"]}
            for metric in ("balanced_accuracy", "bce"):
                row[metric] = {
                    "agreement": mean(metric, "shared_consensus_lambda1_capped"),
                    "mean": mean(metric, f"bank_task{task}_mean_capped"),
                    "vae": mean(metric, f"shared_vae{task}_lambda1"),
                    "direct": mean(metric, f"bank_task{task}_best"),
                    "random": float(np.mean([mean(metric, name) for name in names
                                              if name.startswith("random_capped_")])),
                    "dense": mean(metric, "dense"),
                }
            row["uncapped_accuracy"] = {
                "agreement": mean("balanced_accuracy", "shared_consensus_lambda1"),
                "mean": mean("balanced_accuracy", f"bank_task{task}_mean")}
            masks = result["masks"]
            row["covered_pixels"] = {
                "agreement_uncapped": int((masks[names.index(
                    "shared_consensus_lambda1")].sum(1) > 0).sum()),
                "agreement_capped": int((masks[names.index(
                    "shared_consensus_lambda1_capped")].sum(1) > 0).sum()),
                "mean_uncapped": int((masks[names.index(
                    f"bank_task{task}_mean")].sum(1) > 0).sum()),
                "mean_capped": int((masks[names.index(
                    f"bank_task{task}_mean_capped")].sum(1) > 0).sum()),
            }
            rows.append(row)
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, density in zip(axes, DENSITIES):
        for pair in PAIRS:
            subset = [row for row in rows if tuple(row["pair"]) == pair and
                      row["density"] == density]
            if subset:
                delta = np.mean([row["balanced_accuracy"]["agreement"] -
                                 row["balanced_accuracy"]["mean"] for row in subset])
                ax.bar(pair_name(pair), 100 * delta)
        ax.axhline(0, color="black", lw=1)
        ax.set_title(f"{density:.0%} связей")
        ax.set_ylabel("Agreement − средняя карта, п. п.")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "agreement_vs_mean.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, density in zip(axes, DENSITIES):
        subset = [row for row in rows if row["density"] == density]
        values = []
        for method, label in (("agreement", "Agreement"), ("mean", "Средняя карта")):
            raw = np.mean([row["uncapped_accuracy"][method] for row in subset])
            capped = np.mean([row["balanced_accuracy"][method] for row in subset])
            values.extend((raw, capped))
            ax.plot((0, 1), (raw, capped), "o-", label=label)
        ax.set_xticks((0, 1), ("Без ограничения", "С ограничением"))
        ax.set_ylim(min(values) - .004, max(values) + .004)
        ax.set_title(f"{density:.0%} связей")
        ax.set_ylabel("Сбалансированная точность")
        ax.grid(axis="y", alpha=.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "cap_interaction.png", dpi=160)
    plt.close(fig)
    lines = ["# Ночной запуск MNIST8m", "",
             "Сбалансированная точность после нового обучения MLP; "
             "маски одинаковой плотности, 4 инициализации на маску.", "",
             "| Пара | Плотность | Опор | Agreement | Средняя | Прямая | Случайная | Плотная | Δ, п. п. | IoU до/после |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pair in PAIRS:
        for density in DENSITIES:
            subset = [row for row in rows if tuple(row["pair"]) == pair and
                      row["density"] == density]
            if not subset:
                continue
            avg = {method: np.mean([row["balanced_accuracy"][method]
                                    for row in subset])
                   for method in ("agreement", "mean", "direct", "random", "dense")}
            iou_rows = [row["alignment_iou"] for row in subset if "alignment_iou" in row]
            iou = (f"{np.mean([row['raw'] for row in iou_rows]):.3f} / "
                   f"{np.mean([row['aligned'] for row in iou_rows]):.3f}"
                   if iou_rows else "—")
            lines.append(f"| {pair[0]}/{pair[1]} | {density:.0%} | {len(subset) // 2} | "
                         f"{avg['agreement']:.4f} | {avg['mean']:.4f} | "
                         f"{avg['direct']:.4f} | {avg['random']:.4f} | "
                         f"{avg['dense']:.4f} | {100 * (avg['agreement'] - avg['mean']):+.2f} | "
                         f"{iou} |")
    lines += ["", "IoU: совпадение средней обучающей карты с отложенными картами "
              "до и после выравнивания столбцов.", "",
              "![Разница agreement со средней картой](agreement_vs_mean.png)", "",
              "[Числа по каждой цифре и опоре](summary.json).", ""]
    lines += ["## Ограничение числа связей на пиксель", "",
              "| Плотность | Agreement без ограничения | Средняя без ограничения | "
              "Agreement с ограничением | Средняя с ограничением |",
              "|---|---:|---:|---:|---:|"]
    for density in DENSITIES:
        subset = [row for row in rows if row["density"] == density]
        vals = [np.mean([row["uncapped_accuracy"]["agreement"] for row in subset]),
                np.mean([row["uncapped_accuracy"]["mean"] for row in subset]),
                np.mean([row["balanced_accuracy"]["agreement"] for row in subset]),
                np.mean([row["balanced_accuracy"]["mean"] for row in subset])]
        lines.append(f"| {density:.0%} | " + " | ".join(f"{value:.4f}" for value in vals) + " |")
    lines += ["", "![Влияние ограничения связей на пиксель](cap_interaction.png)", "",
              "При поиске z BCE считается на масках без ограничения. "
              "Итоговая таблица выше использует ограничение для agreement и средней карты; "
              "поиск z не обучался непосредственно на такой финальной маске.", ""]
    transfer_path = out / "transfer_summary.json"
    if transfer_path.exists():
        transfer = json.loads(transfer_path.read_text())
        lines += ["## Перенос на 10 классов", "",
                  "Одна заранее заданная опора для каждой пары и плотности. "
                  "Исходные задачи были «цифра против остальных», поэтому остальные "
                  "цифры встречались в них как отрицательные примеры; это тест переноса "
                  "на новые положительные классы, а не строгий zero-shot по изображениям.", "",
                  "| Пара | Плотность | Agreement | Средняя | Случайная | Плотная | Полнота других 8: agreement/средняя |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for item in transfer:
            acc = item["accuracy"]
            lines.append(f"| {item['pair'][0]}/{item['pair'][1]} | {item['density']:.0%} | "
                         f"{acc['agreement']:.4f} | {acc['mean']:.4f} | "
                         f"{acc['random']:.4f} | {acc['dense']:.4f} | "
                         f"{item['other_recall']['agreement']:.4f} / "
                         f"{item['other_recall']['mean']:.4f} |")
        lines += ["", "![Качество 10-классовой классификации](transfer_accuracy.png)", "",
                  "[Точность по отдельным цифрам и повторениям](transfer_summary.json).", ""]
    strict_path = out / "strict_holdout_summary.json"
    if strict_path.exists():
        strict_rows = json.loads(strict_path.read_text())
        lines += ["## Строго отложенные цифры", "",
                  "При поиске маски доступны только цифры 0, 1, 2, 3 и 8. "
                  "Цифры 4, 5, 6, 7 и 9 исключены также из отрицательных примеров. "
                  "После фиксации маски обучена новая 10-классовая MLP.", "",
                  "| Плотность | Agreement: полнота на отложенных | Средняя | Случайная | Плотная |",
                  "|---|---:|---:|---:|---:|"]
        for item in strict_rows:
            methods = item["methods"]
            lines.append(f"| {item['density']:.0%} | "
                         f"{methods['agreement']['heldout_recall']:.4f} | "
                         f"{methods['mean']['heldout_recall']:.4f} | "
                         f"{methods['random']['heldout_recall']:.4f} | "
                         f"{methods['dense']['heldout_recall']:.4f} |")
        lines += ["", "![Полнота на отложенных цифрах](strict_holdout.png)", "",
                  "[Общая точность и данные по методам](strict_holdout_summary.json).", ""]
    (out / "report.md").write_text("\n".join(lines))


def transfer_summary(out: Path, configs: list[dict]) -> None:
    rows = []
    for config in configs:
        if not config["transfer"]:
            continue
        path = Path(config["folder"]) / "transfer.pt"
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        plot_path = path.with_name("transfer_masks.png")
        if not plot_path.exists():
            plot_masks(payload, plot_path)
        names = payload["names"]
        accuracies = payload["accuracy"].mean(-1)
        recall = payload["recall"].mean(-1)
        source = tuple(config["pair"])
        random_indices = [i for i, name in enumerate(names) if name.startswith("random")]
        accuracy = {name: float(accuracies[i]) for i, name in enumerate(names)}
        accuracy["random"] = float(accuracies[random_indices].mean())
        source_recall = {name: float(recall[i, list(source)].mean())
                         for i, name in enumerate(names)}
        source_recall["random"] = float(recall[random_indices][:, list(source)].mean())
        other_digits = [d for d in range(10) if d not in source]
        other_recall = {name: float(recall[i, other_digits].mean())
                        for i, name in enumerate(names)}
        other_recall["random"] = float(recall[random_indices][:, other_digits].mean())
        rows.append({"pair": source, "density": config["density"],
                     "anchor_task": config["anchor_task"],
                     "anchor_index": config["anchor_index"],
                     "accuracy": accuracy,
                     "source_recall": source_recall,
                     "other_recall": other_recall,
                     "per_digit_recall": {name: recall[i].tolist()
                                          for i, name in enumerate(names)}})
    (out / "transfer_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, density in zip(axes, DENSITIES):
        subset = [row for row in rows if row["density"] == density]
        x = np.arange(len(subset))
        plotted = []
        for i, method in enumerate(("random", "mean", "agreement", "dense")):
            values = [row["accuracy"][method] for row in subset]
            plotted.extend(values)
            ax.bar(x + (i - 1.5) * .19,
                   values, .18, label=method)
        ax.set_xticks(x, [pair_name(tuple(row["pair"])) for row in subset])
        ax.set_title(f"{density:.0%} связей")
        ax.set_ylabel("Точность, 10 классов")
        ax.set_ylim(max(0, min(plotted) - .02), min(1, max(plotted) + .02))
        ax.grid(axis="y", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .99))
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(out / "transfer_accuracy.png", dpi=160)
    plt.close(fig)


def strict_summary(out: Path, runs: list[dict]) -> None:
    rows = []
    for run in runs:
        path = Path(run["folder"]) / "transfer.pt"
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        plot_path = path.with_name("transfer_masks.png")
        if not plot_path.exists():
            plot_masks(payload, plot_path)
        names = payload["names"]
        accuracy = payload["accuracy"].mean(-1)
        recall = payload["recall"].mean(-1)
        heldout = payload["heldout_digits"]
        random_indices = [i for i, name in enumerate(names) if name.startswith("random")]
        def values(method: str) -> tuple[float, float]:
            indices = random_indices if method == "random" else [names.index(method)]
            return (float(accuracy[indices].mean()),
                    float(recall[indices][:, heldout].mean()))
        rows.append({"density": run["density"], "pair": payload["pair"],
                     "visible_digits": payload["visible_digits"],
                     "heldout_digits": heldout, "anchor": run["anchor"],
                     "methods": {method: {"accuracy": values(method)[0],
                                           "heldout_recall": values(method)[1]}
                                 for method in ("agreement", "mean", "vae0", "vae1",
                                                "direct0", "direct1", "random", "dense")}})
    (out / "strict_holdout_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(rows))
        plotted = []
        for index, method in enumerate(("random", "mean", "agreement", "dense")):
            values = [row["methods"][method]["heldout_recall"] for row in rows]
            plotted.extend(values)
            ax.bar(x + (index - 1.5) * .19,
                   values, .18, label=method)
        ax.set_xticks(x, [f"{row['density']:.0%}" for row in rows])
        ax.set_ylim(max(0, min(plotted) - .02), min(1, max(plotted) + .02))
        ax.set_ylabel("Средняя полнота на цифрах 4, 5, 6, 7, 9")
        ax.set_xlabel("Плотность связей")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(out / "strict_holdout.png", dpi=160)
        plt.close(fig)


def run_strict(args: argparse.Namespace) -> None:
    pair = (3, 8)
    visible = (0, 1, 2, 3, 8)
    flags = ("--visible-digits", *(str(digit) for digit in visible))
    root = args.out / "strict_holdout"
    source = root / "bank20"
    if not args.dry_run:
        subprocess.run(raw_command(source, pair, .2, "features", "--device", "cpu",
                                   *flags), check=True, stdout=subprocess.DEVNULL)
    jobs = [Job(f"strict_bank_t{task}",
                raw_command(source, pair, .2, "bank", "--task-index", str(task),
                            "--bank-candidates", str(args.bank_candidates),
                            "--bank-steps", str(args.bank_steps), *flags),
                (source / f"bank_task{task}.pt",),
                args.out / "logs" / "strict" / f"bank_t{task}.log")
            for task in (0, 1)]
    run_jobs(jobs, args.gpu_ids, "strict bank", dry_run=args.dry_run)
    if not args.dry_run:
        for task, digit in enumerate(pair):
            payload = torch.load(source / f"bank_task{task}.pt", map_location="cpu",
                                 weights_only=True, mmap=True)
            settings = payload["settings"]
            if (payload["digit"] != digit or settings.get("visible_digits") != list(visible)
                    or settings.get("candidates") != args.bank_candidates
                    or settings.get("max_steps") != args.bank_steps):
                raise ValueError(f"strict bank protocol mismatch: task {task}")
    jobs = []
    for density in DENSITIES:
        target = root / "prepared" / density_name(density)
        jobs.append(Job(f"strict_prepare_{density_name(density)}",
                        (sys.executable, "-m", "pattern.prepare_mnist8m_5pct",
                         "--source", str(source), "--out", str(target),
                         "--density", str(density)),
                        (target / "bank_task0.pt", target / "bank_task1.pt",
                         target / "features.pt"),
                        args.out / "logs" / "strict" /
                        f"prepare_{density_name(density)}.log"))
    run_jobs(jobs, args.gpu_ids, "strict prepare", dry_run=args.dry_run,
             cpu_only=True)
    if not args.dry_run:
        for density in DENSITIES:
            for task, digit in enumerate(pair):
                path = root / "prepared" / density_name(density) / f"bank_task{task}.pt"
                payload = torch.load(path, map_location="cpu", weights_only=True,
                                     mmap=True)
                if (payload["digit"] != digit or
                        payload["settings"].get("visible_digits") != list(visible) or
                        payload["settings"].get("derived_connections") != round(
                            density * 784 * 64)):
                    raise ValueError(f"strict prepared bank mismatch: {path}")
    if args.dry_run and not all((root / "prepared" / density_name(d) /
                                 "bank_task1.pt").exists() for d in DENSITIES):
        print("strict: later stages need prepared banks", flush=True)
        return
    runs = []
    for density in DENSITIES:
        prepared = root / "prepared" / density_name(density)
        task, index = anchors(prepared, 1)[0]
        folder = root / "runs" / density_name(density) / f"anchor_t{task}_i{index}"
        runs.append({"density": density, "folder": str(folder), "anchor": [task, index]})
    manifest = {"pair": pair, "visible_digits": visible, "runs": runs,
                "bank_candidates": args.bank_candidates, "bank_steps": args.bank_steps,
                "vae_epochs": args.vae_epochs, "search_steps": args.search_steps,
                "evaluation_steps": args.evaluation_steps,
                "transfer_steps": args.transfer_steps}
    manifest_path = root / "manifest.json"
    if not args.dry_run:
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(
                json.dumps(manifest)):
            raise ValueError(f"strict resume settings differ from {manifest_path}")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = []
    for run in runs:
        density = run["density"]
        folder = Path(run["folder"])
        prepared = root / "prepared" / density_name(density)
        task, index = run["anchor"]
        jobs.append(Job(f"strict_align_{density_name(density)}",
                        (sys.executable, "-m", "pattern.align_mnist8m_importance_bank",
                         "--source", str(prepared), "--out", str(folder),
                         "--reference-task", str(task), "--reference-index", str(index)),
                        (folder / "bank_task0.pt", folder / "bank_task1.pt",
                         folder / "features.pt", folder / "alignment_diagnostics.json"),
                        args.out / "logs" / "strict" /
                        f"align_{density_name(density)}.log"))
    run_jobs(jobs, args.gpu_ids, "strict align", dry_run=args.dry_run,
             cpu_only=True)
    jobs = []
    for run in runs:
        folder = Path(run["folder"])
        for task in (0, 1):
            jobs.append(Job(f"strict_vae_{density_name(run['density'])}_t{task}",
                            raw_command(folder, pair, run["density"], "vae",
                                        "--task-index", str(task),
                                        "--vae-epochs", str(args.vae_epochs), *flags),
                            (folder / f"vae_task{task}.pt",),
                            args.out / "logs" / "strict" /
                            f"vae_{density_name(run['density'])}_t{task}.log"))
    run_jobs(jobs, args.gpu_ids, "strict VAE", dry_run=args.dry_run)
    jobs = []
    for run in runs:
        folder = Path(run["folder"])
        for objective in ("own", "shared"):
            for coefficient in (0, 1, 10):
                prefix = "search_shared" if objective == "shared" else "search"
                jobs.append(Job(f"strict_{density_name(run['density'])}_"
                                f"{objective}{coefficient}",
                                raw_command(folder, pair, run["density"], "search",
                                            "--objective", objective, "--coefficient",
                                            str(coefficient), "--search-steps",
                                            str(args.search_steps), *flags),
                                (folder / f"{prefix}_lambda{coefficient}.pt",),
                                args.out / "logs" / "strict" /
                                f"search_{density_name(run['density'])}_"
                                f"{objective}{coefficient}.log"))
    run_jobs(jobs, args.gpu_ids, "strict search", dry_run=args.dry_run)
    jobs = []
    for run in runs:
        folder = Path(run["folder"])
        jobs.append(Job(f"strict_evaluate_{density_name(run['density'])}",
                        raw_command(folder, pair, run["density"], "evaluate",
                                    "--evaluation-steps", str(args.evaluation_steps), *flags),
                        (folder / "evaluation.pt", folder / "report.md",
                         folder / "full_masks.png", folder / "search_convergence.png"),
                        args.out / "logs" / "strict" /
                        f"evaluate_{density_name(run['density'])}.log"))
    run_jobs(jobs, args.gpu_ids, "strict evaluate", dry_run=args.dry_run)
    jobs = []
    for run in runs:
        folder = Path(run["folder"])
        jobs.append(Job(f"strict_transfer_{density_name(run['density'])}",
                        (sys.executable, "-m", "pattern.mnist8m_night_transfer",
                         "--source", str(folder), "--out", str(folder / "transfer.pt"),
                         "--density", str(run["density"]), "--steps",
                         str(args.transfer_steps)),
                        (folder / "transfer.pt", folder / "transfer_masks.png"),
                        args.out / "logs" / "strict" /
                        f"transfer_{density_name(run['density'])}.log"))
    run_jobs(jobs, args.gpu_ids, "strict transfer", dry_run=args.dry_run)
    if not args.dry_run:
        strict_summary(args.out, runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("pattern/outputs/mnist8m_night_20260927"))
    parser.add_argument("--reuse-pair38", type=Path, default=Path(
        "pattern/outputs/mnist8m_raw_mlp_bce/pair38"))
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--anchors-per-task", type=int, default=4)
    parser.add_argument("--bank-candidates", type=int, default=4096)
    parser.add_argument("--bank-steps", type=int, default=1200)
    parser.add_argument("--vae-epochs", type=int, default=200)
    parser.add_argument("--search-steps", type=int, default=24000)
    parser.add_argument("--evaluation-steps", type=int, default=30000)
    parser.add_argument("--transfer-steps", type=int, default=20000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.anchors_per_task < 1:
        parser.error("--anchors-per-task must be positive")
    if not args.gpu_ids or len(set(args.gpu_ids)) != len(args.gpu_ids):
        parser.error("--gpu-ids must contain distinct GPUs")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    configurations_count = len(PAIRS) * len(DENSITIES) * 2 * args.anchors_per_task
    print(f"Plan: {configurations_count} anchors, {2 * configurations_count} VAE, "
          f"{6 * configurations_count} searches, {configurations_count} mask evaluations, "
          f"{len(PAIRS) * len(DENSITIES)} ten-class transfers; "
          "plus 2 strict holdout controls", flush=True)

    def active(stage: str) -> bool:
        return args.stage in ("all", stage)

    if args.stage == "strict":
        run_strict(args)
        return

    if active("bank"):
        jobs = []
        for pair in PAIRS:
            source = source_folder(args.out, pair, args.reuse_pair38)
            if pair == (3, 8):
                for task, digit in enumerate(pair):
                    path = source / f"bank_task{task}.pt"
                    if not path.exists():
                        raise FileNotFoundError(f"missing reusable bank for {digit}: {path}")
                print(f"bank: reusing {source}", flush=True)
                continue
            if not args.dry_run:
                command = raw_command(source, pair, .2, "features", "--device", "cpu")
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            for task in (0, 1):
                jobs.append(Job(f"{pair_name(pair)}_t{task}",
                                raw_command(source, pair, .2, "bank", "--task-index",
                                            str(task), "--bank-candidates",
                                            str(args.bank_candidates), "--bank-steps",
                                            str(args.bank_steps)),
                                (source / f"bank_task{task}.pt",),
                                args.out / "logs" / "bank" / f"{pair_name(pair)}_t{task}.log"))
        run_jobs(jobs, args.gpu_ids, "bank", dry_run=args.dry_run)
    if args.stage == "bank":
        return
    if active("prepare"):
        jobs = []
        for pair in PAIRS:
            source = source_folder(args.out, pair, args.reuse_pair38)
            for density in DENSITIES:
                target = prepared_folder(args.out, pair, density)
                jobs.append(Job(f"{pair_name(pair)}_{density_name(density)}",
                                (sys.executable, "-m", "pattern.prepare_mnist8m_5pct",
                                 "--source", str(source), "--out", str(target),
                                 "--density", str(density)),
                                (target / "bank_task0.pt", target / "bank_task1.pt",
                                 target / "features.pt"),
                                args.out / "logs" / "prepare" /
                                f"{pair_name(pair)}_{density_name(density)}.log"))
        run_jobs(jobs, args.gpu_ids, "prepare", dry_run=args.dry_run, cpu_only=True)

    # All later stages use the exact same frozen anchor list and configuration.
    if not all((prepared_folder(args.out, pair, density) / "bank_task1.pt").exists()
               for pair in PAIRS for density in DENSITIES):
        if args.dry_run:
            print("Later stages need prepared banks; rerun --dry-run after preparation.")
            return
        raise FileNotFoundError("prepared banks are missing; run --stage bank and prepare")
    validate_banks(args)
    configurations = []
    for pair in PAIRS:
        for density in DENSITIES:
            source = prepared_folder(args.out, pair, density)
            for number, (task, index) in enumerate(anchors(source, args.anchors_per_task)):
                configurations.append({"pair": pair, "density": density,
                                       "anchor_task": task, "anchor_index": index,
                                       "folder": str(anchored_folder(args.out, pair, density,
                                                                     task, index)),
                                       "transfer": number == 0})
    if not args.dry_run:
        save_manifest(args.out, args, configurations)

    if active("align"):
        jobs = []
        for config in configurations:
            pair = tuple(config["pair"])
            density = config["density"]
            folder = Path(config["folder"])
            source = prepared_folder(args.out, pair, density)
            jobs.append(Job(f"{pair_name(pair)}_{density_name(density)}_"
                            f"t{config['anchor_task']}_i{config['anchor_index']}",
                            (sys.executable, "-m", "pattern.align_mnist8m_importance_bank",
                             "--source", str(source), "--out", str(folder),
                             "--reference-task", str(config["anchor_task"]),
                             "--reference-index", str(config["anchor_index"])),
                            (folder / "bank_task0.pt", folder / "bank_task1.pt",
                             folder / "features.pt", folder / "alignment_diagnostics.json"),
                            args.out / "logs" / "align" / f"{folder.name}_{pair_name(pair)}_"
                            f"{density_name(density)}.log"))
        run_jobs(jobs, args.gpu_ids, "align", dry_run=args.dry_run, cpu_only=True)
    if active("vae"):
        jobs = []
        for config in configurations:
            pair = tuple(config["pair"])
            density = config["density"]
            folder = Path(config["folder"])
            for task in (0, 1):
                jobs.append(Job(f"{pair_name(pair)}_{density_name(density)}_"
                                f"{folder.name}_vae{task}",
                                raw_command(folder, pair, density, "vae", "--task-index",
                                            str(task), "--vae-epochs", str(args.vae_epochs)),
                                (folder / f"vae_task{task}.pt",),
                                args.out / "logs" / "vae" / f"{pair_name(pair)}_"
                                f"{density_name(density)}_{folder.name}_t{task}.log"))
        run_jobs(jobs, args.gpu_ids, "vae", dry_run=args.dry_run)
    if active("search"):
        jobs = []
        for config in configurations:
            pair = tuple(config["pair"])
            density = config["density"]
            folder = Path(config["folder"])
            for objective in ("own", "shared"):
                for coefficient in (0, 1, 10):
                    basename = "search_shared" if objective == "shared" else "search"
                    jobs.append(Job(f"{pair_name(pair)}_{density_name(density)}_"
                                    f"{folder.name}_{objective}{coefficient}",
                                    raw_command(folder, pair, density, "search",
                                                "--objective", objective,
                                                "--coefficient", str(coefficient),
                                                "--search-steps", str(args.search_steps)),
                                    (folder / f"{basename}_lambda{coefficient}.pt",),
                                    args.out / "logs" / "search" / f"{pair_name(pair)}_"
                                    f"{density_name(density)}_{folder.name}_"
                                    f"{objective}{coefficient}.log"))
        run_jobs(jobs, args.gpu_ids, "search", dry_run=args.dry_run)
    if active("evaluate"):
        jobs = []
        for config in configurations:
            pair = tuple(config["pair"])
            density = config["density"]
            folder = Path(config["folder"])
            jobs.append(Job(f"{pair_name(pair)}_{density_name(density)}_{folder.name}",
                            raw_command(folder, pair, density, "evaluate",
                                        "--evaluation-steps", str(args.evaluation_steps)),
                            (folder / "evaluation.pt", folder / "report.md",
                             folder / "quality.png", folder / "pixel_maps.png",
                             folder / "full_masks.png", folder / "search_convergence.png"),
                            args.out / "logs" / "evaluate" / f"{pair_name(pair)}_"
                            f"{density_name(density)}_{folder.name}.log"))
        run_jobs(jobs, args.gpu_ids, "evaluate", dry_run=args.dry_run)
    if active("transfer"):
        jobs = []
        for config in configurations:
            if not config["transfer"]:
                continue
            pair = tuple(config["pair"])
            density = config["density"]
            folder = Path(config["folder"])
            jobs.append(Job(f"{pair_name(pair)}_{density_name(density)}",
                            (sys.executable, "-m", "pattern.mnist8m_night_transfer",
                             "--source", str(folder), "--out", str(folder / "transfer.pt"),
                             "--density", str(density), "--steps", str(args.transfer_steps)),
                            (folder / "transfer.pt", folder / "transfer_masks.png"),
                            args.out / "logs" / "transfer" /
                            f"{pair_name(pair)}_{density_name(density)}.log"))
        run_jobs(jobs, args.gpu_ids, "transfer", dry_run=args.dry_run)
        if not args.dry_run:
            transfer_summary(args.out, configurations)
    if active("strict"):
        run_strict(args)
    if active("report") and not args.dry_run:
        if args.stage == "report":
            transfer_summary(args.out, configurations)
            strict_manifest = args.out / "strict_holdout" / "manifest.json"
            if strict_manifest.exists():
                strict_summary(args.out, json.loads(strict_manifest.read_text())["runs"])
        summary(args.out, configurations)
        print(args.out / "report.md", flush=True)


if __name__ == "__main__":
    main()
