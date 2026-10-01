"""Compare retrained MNIST8m MLPs at several mask densities."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mds.build_mnist8m_mask_control import interval


ROOT = Path("pattern/outputs/mnist8m_raw_mlp_bce/mask_control")
EXTENDED = Path("pattern/outputs/mnist8m_raw_mlp_bce/mask_control_extended")
REPORT = Path("mds/MNIST8M_SPARSITY_CONTROL_2026-09-26.md")
ASSET = Path("mds/assets/2026-09-26/mnist8m_sparsity_control.png")
DENSITIES = (.02, .05, .2)


def payload(density: float, task: int) -> dict:
    suffix = (f"_importance_{density:.4f}".replace(".", "p")
              if density < .2 else "")
    filename = f"task{task}{suffix}.pt"
    source = EXTENDED / filename if (EXTENDED / filename).exists() else ROOT / filename
    return torch.load(source, map_location="cpu",
                      weights_only=True)


def main() -> None:
    rows = []
    for density in DENSITIES:
        tasks = [payload(density, task) for task in (0, 1)]
        assert all(item["settings"]["pairs"] == 410 and
                   item["settings"]["repeats"] == 4 for item in tasks)
        row = {"density": density, "connections": tasks[0]["settings"].get(
            "connections", round(density * 784 * 64)), "steps": [
                item["settings"]["actual_steps"] for item in tasks],
               "late_best": [
                   float((item["best_step"] > .9 * item["settings"]["actual_steps"])
                         .float().mean()) if "best_step" in item else None
                   for item in tasks]}
        for key, scale in (("balanced_accuracy", 100), ("balanced_bce", 1000)):
            differences = np.concatenate([
                (item[key][0] - item[key][1]).numpy().mean(1) * scale
                for item in tasks])
            row[key] = {
                "selected": float(np.mean([item[key][0].mean().item()
                                            for item in tasks])),
                "random": float(np.mean([item[key][1].mean().item()
                                          for item in tasks])),
                "delta_ci": interval(differences, 38000 + int(density * 1000)),
                "per_task": [
                    interval((item[key][0] - item[key][1]).numpy().mean(1) * scale,
                             39000 + int(density * 1000) + task)
                    for task, item in enumerate(tasks)],
            }
        rows.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title, unit in (
        (axes[0], "balanced_accuracy", "Сбалансированная точность", "п. п."),
        (axes[1], "balanced_bce", "BCE", "×10⁻³"),
    ):
        estimates = [row[key]["delta_ci"] for row in rows]
        center = np.array([x[0] for x in estimates])
        lower = center - np.array([x[1] for x in estimates])
        upper = np.array([x[2] for x in estimates]) - center
        x = np.arange(len(rows))
        ax.errorbar(x, center, yerr=np.stack((lower, upper)), fmt="o-",
                    capsize=4, color="#246399")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(x, [f"{100 * density:g}%" for density in DENSITIES])
        ax.set_ylabel(f"Лучшие − случайные, {unit}")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET, dpi=170)
    plt.close(fig)

    lines = ["# MNIST8m: проверка плотности масок", "",
             "Та же бинарная классификация цифр 3 и 8. Для плотности 2% и 5% "
             "из карт лучших MLP банка оставлены top-K связей по |W·M|. "
             "При 20% используется исходная маска MLP; она точно совпадает "
             "с top-K её карты важности. На каждой плотности "
             "сравнили 410 отобранных и 410 случайных масок на задачу: "
             "четыре новых обучения MLP с одинаковыми начальными весами "
             "и пакетами внутри каждой пары. Оценка — на отдельном тестовом "
             "блоке.", "",
             "| Плотность | Связей | Top-K: точность | Случайная: точность | "
             "Разница, п. п. [95% ДИ] |",
             "|---:|---:|---:|---:|---:|"]
    for row in rows:
        data = row["balanced_accuracy"]
        mean, low, high = data["delta_ci"]
        lines.append(f"| {row['density'] * 100:g}% | {row['connections']} | "
                     f"{data['selected']:.4f} | {data['random']:.4f} | "
                     f"{mean:+.3f} [{low:+.3f}; {high:+.3f}] |")
    lines.extend(["", "| Плотность | Top-K: BCE | Случайная: BCE | "
                  "Разница BCE, ×10⁻³ [95% ДИ] |",
                  "|---:|---:|---:|---:|"])
    for row in rows:
        data = row["balanced_bce"]
        mean, low, high = data["delta_ci"]
        lines.append(f"| {row['density'] * 100:g}% | {data['selected']:.5f} | "
                     f"{data['random']:.5f} | "
                     f"{mean:+.3f} [{low:+.3f}; {high:+.3f}] |")
    lines.extend(["", "![Парное сравнение по плотности]"
                  "(assets/2026-09-26/mnist8m_sparsity_control.png)", "",
                  "Средние значения даны по двум задачам. Интервалы — "
                  "парный бутстрэп по маскам при фиксированных изображениях; "
                  "они не показывают разброс между другими задачами.", "",
                  "| Плотность | Разница точности: цифра 3, п. п. | "
                  "Разница точности: цифра 8, п. п. | Шаги для 3/8 | "
                  "Поздний лучший чекпоинт, 3/8 |",
                  "|---:|---:|---:|---:|---:|"])
    for row in rows:
        a, b = row["balanced_accuracy"]["per_task"]
        late = "/".join("—" if value is None else f"{100 * value:.1f}%"
                        for value in row["late_best"])
        lines.append(f"| {row['density'] * 100:g}% | {a[0]:+.3f} | "
                     f"{b[0]:+.3f} | {row['steps'][0]}/{row['steps'][1]} | "
                     f"{late} |")
    gain_2 = rows[0]["balanced_accuracy"]["delta_ci"][0]
    gain_5 = rows[1]["balanced_accuracy"]["delta_ci"][0]
    lines.extend(["", "**Вывод.** При 20% маски лучших MLP не превосходили "
                  "случайные. После отбора сильнейших связей преимущество "
                  f"появилось на обеих задачах: в среднем {gain_5:+.2f} п. п. "
                  f"при 5% и {gain_2:+.2f} п. п. при 2%. "
                  "Дополнительные шаги обучения не "
                  "устранили разницу. Для следующей проверки VAE разумно "
                  "взять 5%: это 2509 связей и более высокая абсолютная "
                  "точность, чем при 2%.", "",
                  "[Код проверки](../pattern/mnist8m_bank_mask_control.py) · "
                  "[основные результаты](../pattern/outputs/mnist8m_raw_mlp_bce/mask_control/) · "
                  "[продлённые прогоны](../pattern/outputs/mnist8m_raw_mlp_bce/mask_control_extended/)", ""])
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
