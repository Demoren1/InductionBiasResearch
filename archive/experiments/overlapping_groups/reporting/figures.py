"""Build report heatmaps from saved searches; no fitting or latent search."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..io import DEFAULT_BANK, checked_load, save_json, sha256
from ..masks import structural_metrics
from ..models.vae import Decoder, topk
from ..problem import reference


def align(mask, target, scores=None):
    metrics, orders = structural_metrics(mask[None], target)
    order = orders[0]
    return (
        mask[:, order],
        scores[:, order] if scores is not None else None,
        metrics[0],
    )


def panel(axis, values, title):
    result = axis.imshow(
        values,
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    axis.axhline(15.5, color="#c9922d", linewidth=1)
    axis.axhline(31.5, color="#bb5555", linewidth=1)
    axis.set_xticks([0, 3, 7])
    axis.set_yticks([0, 15, 16, 31, 32, 47])
    axis.set_title(title, fontsize=10)
    return result


def build(root, bank, decoder_run, seed=4213, families=(0, 22)):
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    manifest = json.loads((bank / "manifest.json").read_text())
    decoder_path = decoder_run / "decoder.pt"
    payload = checked_load(decoder_path)
    decoder = Decoder(payload["config"]).eval().requires_grad_(False)
    decoder.load_state_dict(payload["model"])
    provenance = {
        "decoder_sha256": sha256(decoder_path),
        "seed": seed,
        "families": {},
        "scope": "CPU inference only. The pre/post pair uses the initial and final z of the same selected restart. Columns matched to the analytical reference only for figures.",
    }
    for family in families:
        directory = root / f"wg_{seed}" / f"family_{family:02d}"
        target = reference(manifest["family_definitions"][family], "cpu")
        selection = checked_load(directory / "selection.pt")
        examples = [("Аналитическая", target, target)]
        info = {
            "groups": manifest["family_definitions"][family]["groups"],
            "methods": {},
        }
        for method, label in [
            ("gumbel", "Gumbel"),
            ("evolution", "Evolution"),
        ]:
            search = checked_load(directory / f"{method}_search.pt")
            index = search["selected"]
            for tag, z in [
                ("до", search["initial_z"][index]),
                ("после", search["z"][index]),
            ]:
                with torch.no_grad():
                    logits = decoder(z[None])[0]
                    mask = topk(logits)
                aligned, soft, metric = align(mask, target, logits.sigmoid())
                examples.append(
                    (f"{label}: {tag}\nстарт №{index}", aligned, soft)
                )
                info["methods"][f"{method}_{tag}"] = metric
            info["methods"][method + "_restart"] = index
            info["methods"][method + "_initial_fitness"] = float(
                search["initial_scores"][index]
            )
            info["methods"][method + "_final_fitness"] = float(
                search["scores"][index]
            )
        with torch.no_grad():
            zero = decoder(torch.zeros(1, payload["config"]["latent_dim"]))[0]
        aligned, soft, metric = align(topk(zero), target, zero.sigmoid())
        examples.insert(1, ("Декодер: z = 0", aligned, soft))
        fig, axes = plt.subplots(
            2, len(examples), figsize=(15, 9.6), constrained_layout=True
        )
        for column, (label, mask, scores) in enumerate(examples):
            metric = structural_metrics(mask[None], target)[0][0]
            soft_image = panel(axes[0, column], scores, label)
            hard_image = panel(
                axes[1, column], mask, f"Top-32; IoU={metric['iou']:.3f}"
            )
            axes[1, column].set_xlabel("Скрытый нейрон")
        axes[0, 0].set_ylabel("Оценки связей σ(logit)\nСтрока = вход")
        axes[1, 0].set_ylabel("Бинарная маска\nСтрока = вход")
        fig.colorbar(
            soft_image,
            ax=axes[0, :],
            fraction=0.015,
            pad=0.01,
            label="Оценка связи; не калиброванная вероятность",
        )
        fig.colorbar(
            hard_image,
            ax=axes[1, :],
            fraction=0.015,
            pad=0.01,
            label="0 — связи нет; 1 — связь есть",
        )
        fig.suptitle(
            f"Декодер до и после поиска z: семейство {family}, W/G seed {seed}\nОдин и тот же выбранный старт в каждой паре. 0–15: общие входы; 16–31: переменные; 32–47: шум."
        )
        fig.savefig(
            output / f"decoder_before_after_family_{family:02d}.png", dpi=170
        )
        plt.close(fig)

        imp = checked_load(root / "imp_controls" / f"family_{family:02d}.pt")
        batch = imp["masks"].flatten(0, 1)
        metrics, orders = structural_metrics(batch, target)
        average = (
            torch.stack([mask[:, order] for mask, order in zip(batch, orders)])
            .float()
            .mean(0)
        )
        single = batch[0][:, orders[0]]
        panels = [
            ("Аналитическая", target, 1.0),
            ("MLP без маски\nВсе 384 связи", torch.ones_like(target), None),
            ("IMP: функция 6\nповтор 0", single, metrics[0]["iou"]),
            (
                f"Средний IMP\n{len(batch)} отдельных сетей",
                average,
                float(np.mean([m["iou"] for m in metrics])),
            ),
        ]
        for method, label in [
            ("source_frequency", "Частотная маска"),
            ("best_initial", "Лучший исходный z"),
            ("gumbel", "Gumbel после поиска"),
            ("evolution", "Evolution после поиска"),
            ("random_budget", "Случайные z"),
            ("training_bank", "Выбор из IMP-банка"),
        ]:
            mask, _, metric = align(selection["masks"][method], target)
            panels.append((label, mask, metric["iou"]))
        fig, axes = plt.subplots(
            2, 5, figsize=(13, 10.5), constrained_layout=True
        )
        for axis, (label, mask, iou) in zip(axes.flat, panels):
            suffix = "" if iou is None else f"\nIoU={iou:.3f}"
            im = panel(axis, mask, label + suffix)
            axis.set_xlabel("Скрытый нейрон")
        for axis in axes[:, 0]:
            axis.set_ylabel("Вход: общие / переменные / шум")
        fig.colorbar(
            im,
            ax=axes,
            fraction=0.015,
            pad=0.01,
            label="Для среднего IMP — частота сохранения связи",
        )
        fig.suptitle(
            f"Что находят методы: семейство {family}, W/G seed {seed}\nМаски поиска общие для новых функций. IMP строится отдельно на каждой оценочной функции."
        )
        fig.savefig(output / f"methods_family_{family:02d}.png", dpi=170)
        plt.close(fig)
        info["selection_sha256"] = sha256(directory / "selection.pt")
        provenance["families"][str(family)] = info
    save_json(root / "report_figures.json", provenance)
    print(
        json.dumps(
            {
                "figures": str(output),
                "families": list(families),
                "training": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--decoder-run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--families", type=int, nargs="+", default=[0, 22])
    args = parser.parse_args()
    build(args.root, args.bank, args.decoder_run, args.seed, args.families)
