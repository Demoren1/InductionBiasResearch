"""Extra fixed-sample and degree figures for the other-generators experiment.

This is a report helper.  It does not train, select, or alter any generator;
it reads the completed seed-4100 artefacts and writes figures beside them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_other_generators"
DEFAULT_CANONICAL = ROOT / "outputs/deepsets_vaae/20261001_converged_functional_vae"
SEED = 4100
F, H, K = 784, 32, 5018
GENERATOR_METHODS = ("diffusion", "flow_matching", "gnn_flow", "set_transformer_flow", "gan")
CONTROLS = ("functional_mean_large", "functional_vae_large", "random", "dense")
DISPLAY_METHODS = CONTROLS + GENERATOR_METHODS


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _finite(value: torch.Tensor, label: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{label}: expected shape {shape}, got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label}: non-finite values")
    return value


def _hard_topk(values: torch.Tensor, k: int = K) -> torch.Tensor:
    if values.ndim != 2 or tuple(values.shape) != (F, H):
        raise ValueError("top-k expects one [784,32] map")
    result = torch.zeros_like(values)
    return result.flatten().scatter(0, values.flatten().topk(k).indices, 1.).reshape_as(values)


def _sample_iou_20pct(maps: torch.Tensor, pairs: int = 2048) -> tuple[float, int]:
    """Deterministic sampled-pair IoU, avoiding an accidental O(N²F) report job."""
    flat = _finite(maps, "source train maps").reshape(-1, F, H)
    if len(flat) < 2:
        raise ValueError("need at least two source maps for pair IoU")
    generator = torch.Generator(device="cpu").manual_seed(82731)
    left = torch.randint(len(flat), (pairs,), generator=generator)
    right = torch.randint(len(flat) - 1, (pairs,), generator=generator)
    right += (right >= left).long()
    masks = torch.stack([_hard_topk(item) for item in flat]).bool()
    values = []
    for start in range(0, pairs, 128):
        a, b = masks[left[start:start + 128]], masks[right[start:start + 128]]
        intersection = (a & b).sum((1, 2))
        union = (a | b).sum((1, 2)).clamp_min(1)
        values.append((intersection / union).float())
    return float(torch.cat(values).mean()), pairs


def _load(out: Path, canonical: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    source = canonical / f"seed_{SEED}" / "functional" / "functional_vae_arrays.npz"
    if not source.is_file():
        raise FileNotFoundError(f"Missing canonical aligned functional maps: {source}")
    with np.load(source) as arrays:
        train = _finite(torch.from_numpy(arrays["function_train_aligned"]), "function_train_aligned", (4, 205, F, H))
        valid = _finite(torch.from_numpy(arrays["function_validation_aligned"]), "function_validation_aligned", (4, 51, F, H))
    mask_path = out / f"seed_{SEED}" / "masks.pt"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing combined masks: {mask_path}")
    saved_masks = torch.load(mask_path, map_location="cpu", weights_only=True)
    if not isinstance(saved_masks, dict):
        raise ValueError(f"Expected mask dictionary: {mask_path}")
    masks: dict[str, torch.Tensor] = {}
    for method in DISPLAY_METHODS:
        value = _finite(torch.as_tensor(saved_masks.get(method)), f"mask/{method}", (4, F, H))
        if not bool(torch.all((value == 0) | (value == 1))):
            raise ValueError(f"mask/{method} is not binary")
        expected = F * H if method == "dense" else 7526
        if not bool(torch.all(value.sum((1, 2)) == expected)):
            raise ValueError(f"mask/{method} does not preserve its declared cardinality")
        masks[method] = value
    fits: dict[str, dict[str, Any]] = {}
    samples: dict[str, dict[str, torch.Tensor]] = {}
    for method in GENERATOR_METHODS:
        folder = out / f"seed_{SEED}" / method
        fits[method] = _json(folder / "fit.json")
        payload = torch.load(folder / "samples.pt", map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"{folder}/samples.pt must be a dictionary")
        samples[method] = {
            "samples": _finite(torch.as_tensor(payload.get("samples")), f"samples/{method}", (4, 32, F, H)),
            "score": _finite(torch.as_tensor(payload.get("score")), f"score/{method}", (F, H)),
            "masks": _finite(torch.as_tensor(payload.get("masks")), f"generated masks/{method}", (4, F, H)),
        }
        if not bool(torch.all((samples[method]["masks"] == 0) | (samples[method]["masks"] == 1))):
            raise ValueError(f"generated masks/{method} are not binary")
    return train, valid, masks, fits, samples


def _sample_map_figure(out: Path, train: torch.Tensor, valid: torch.Tensor,
                       samples: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    first_train, first_valid = train[0, 0], valid[0, 0]
    train_mean = train[0].mean(0)
    score_limit = float(torch.stack([train_mean, *[samples[name]["score"] for name in GENERATOR_METHODS]]).max())
    if not np.isfinite(score_limit) or score_limit <= 0:
        raise ValueError("score scale must be positive and finite")
    fig, axes = plt.subplots(len(GENERATOR_METHODS), 5, figsize=(17, 13), constrained_layout=True)
    for row, method in enumerate(GENERATOR_METHODS):
        entries = ((first_train, "source train[task0,map0]", "raw"),
                   (first_valid, "source held-out[task0,map0]", "raw"),
                   (samples[method]["samples"][0, 0], f"{method} sample[task0,0]", "raw"),
                   (train_mean, "source task0 train mean", "score"),
                   (samples[method]["score"], f"{method} score", "score"))
        for ax, (value, title, kind) in zip(axes[row], entries):
            vmax = 1. if kind == "raw" else score_limit
            image = ax.imshow(value.T.numpy(), aspect="auto", cmap="viridis", vmin=0., vmax=vmax,
                              interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("pixel (784)")
            ax.set_ylabel("hidden (32)")
            if row == 0 and kind == "raw":
                fig.colorbar(image, ax=ax, shrink=.75, label="raw normalized map [0,1]")
            if row == 0 and kind == "score":
                fig.colorbar(image, ax=ax, shrink=.75, label="mean/score, shared max")
    _save(fig, out, "fixed_generated_maps_vs_source")
    return {"source_first_train_index": [0, 0], "source_first_heldout_index": [0, 0],
            "generated_index": [0, 0], "raw_colormap_range": [0., 1.],
            "mean_score_common_max": score_limit}


def _degree_maps(out: Path, masks: dict[str, torch.Tensor]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pixel = {method: masks[method].mean((0, 2)).reshape(28, 28).numpy() for method in DISPLAY_METHODS}
    hidden = {method: masks[method].mean((0, 1)).numpy() for method in DISPLAY_METHODS}
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for ax, method in zip(axes.flat, DISPLAY_METHODS):
        image = ax.imshow(pixel[method], cmap="viridis", vmin=0., vmax=1., interpolation="nearest")
        ax.set_title(method, fontsize=9); ax.set_xlabel("pixel x"); ax.set_ylabel("pixel y")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.75, label="mean allowed-edge fraction over hidden and replicas")
    _save(fig, out, "input_pixel_degree")
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), constrained_layout=True)
    for ax, method in zip(axes.flat, DISPLAY_METHODS):
        ax.plot(np.arange(H), hidden[method], marker="o", markersize=2.5, linewidth=1.1)
        ax.axhline(float(np.mean(hidden[method])), color="#777777", linestyle="--", linewidth=.8)
        ax.set_ylim(0., 1.); ax.set_title(method, fontsize=9); ax.set_xlabel("hidden neuron (32)")
        ax.set_ylabel("allowed-edge fraction over pixels")
        ax.grid(alpha=.2)
    _save(fig, out, "hidden_neuron_degree")
    return pixel, hidden


def build(out: Path, canonical: Path) -> dict[str, Any]:
    out = Path(out)
    figure_out = out / "sample_figures"
    figure_out.mkdir(parents=True, exist_ok=False)
    train, valid, masks, fits, samples = _load(out, Path(canonical))
    sample_display = _sample_map_figure(figure_out, train, valid, samples)
    pixel, hidden = _degree_maps(figure_out, masks)
    source_variance = float(train.reshape(-1, F, H).var(dim=0, unbiased=False).mean())
    source_iou, source_pairs = _sample_iou_20pct(train)
    model_metrics = {method: dict(fits[method].get("sample_metrics", {})) for method in GENERATOR_METHODS}
    values = {
        "source_train_coordinate_variance_mean": source_variance,
        "source_train_top20pct_pairwise_iou_sampled_mean": source_iou,
        "source_train_top20pct_pair_count": source_pairs,
        "model_sample_metrics_as_recorded_in_fit_json": model_metrics,
        "fixed_sample_display": sample_display,
    }
    np.savez_compressed(figure_out / "sample_figures_arrays.npz",
                        source_train_first=train[0, 0].numpy(), source_heldout_first=valid[0, 0].numpy(),
                        source_train_mean=train[0].mean(0).numpy(),
                        generated_first=np.stack([samples[name]["samples"][0, 0].numpy() for name in GENERATOR_METHODS]),
                        generated_scores=np.stack([samples[name]["score"].numpy() for name in GENERATOR_METHODS]),
                        input_pixel_degree=np.stack([pixel[name] for name in DISPLAY_METHODS]),
                        hidden_neuron_degree=np.stack([hidden[name] for name in DISPLAY_METHODS]),
                        methods=np.asarray(DISPLAY_METHODS), generator_methods=np.asarray(GENERATOR_METHODS))
    (figure_out / "sample_figures_metrics.json").write_text(json.dumps(values, indent=2, ensure_ascii=False,
                                                               allow_nan=False) + "\n", encoding="utf-8")
    captions = """# Подписи к дополнительным фигурам

## Фиксированные сэмплы и score

`fixed_generated_maps_vs_source.png` показывает строго фиксированные элементы: source train `task=0,map=0`, source held-out `task=0,map=0` и у каждого генератора `task=0,sample=0`. Никакого выбора по качеству, nearest-neighbour или target-метрике здесь нет. Первые три колонки используют общую сырую шкалу `[0,1]`; две последние используют отдельный общий максимум для train mean и generator score. Поэтому карта отдельного сэмпла показывает разнообразие распределения, а mean/score — его усреднённую или отобранную структуру; они не обязаны выглядеть одинаково.

## Степени пикселей и hidden-нейронов

`input_pixel_degree.png` — для каждой финальной binary mask доля разрешённых связей каждого 28×28 input-пикселя, усреднённая по 32 hidden-нейронам и четырём replicas. `hidden_neuron_degree.png` — та же доля по каждому hidden-нейрону, усреднённая по пикселям и replicas. Общая плотность фиксирована (30% для sparse masks), но эти фигуры показывают, как метод перераспределяет этот бюджет по пикселям или hidden units. Они не позволяют вывести межреберные корреляции, причинный эффект архитектуры или перенос на новые target-задачи только по пространственному рисунку.

## Числа

`sample_figures_metrics.json` содержит среднюю покоординатную variance source-bank и sampled pairwise IoU при top-20% для 2048 детерминированных пар source train-карт. Метрики генераторов записаны ровно как сохранены их `fit.json` (полные 32 samples на source task по протоколу запуска); это не новая оптимизация и не новая выборка.
"""
    (figure_out / "CAPTIONS.md").write_text(captions, encoding="utf-8")
    return {"status": "PASS", "out": str(figure_out), **values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    args = parser.parse_args()
    result = build(args.out, args.canonical)
    print(json.dumps({"status": result["status"], "out": result["out"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
