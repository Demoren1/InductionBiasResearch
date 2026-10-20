"""Build report figures from the preserved October 17–18 result tables."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/induction-report-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "tmp/research_20261017/reports/RESULTS.md"
CVAE_SOURCE = ROOT / "tmp/research_20261017/reports/04_multimask_cvae.md"
BLUE = "#2463a6"
ORANGE = "#d17b26"
GREEN = "#26836b"
TEXT = "#233044"
LABELS = ["12E / 1D", "1E / 12D", "1E / 1D", "cVAE surrogate", "cVAE ELBO"]
KINDS = ["original", "reverse", "shared", "cvae", "cvae_elbo"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": "#c7cfda",
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "savefig.facecolor": "white",
})


def table(section):
    """Read one Markdown table, preserving the snapshot's numeric precision."""
    lines = [line for line in section.splitlines() if line.startswith("|")]
    columns = [cell.strip() for cell in lines[0].strip("|").split("|")]
    return [
        dict(zip(columns, [cell.strip() for cell in line.strip("|").split("|")]))
        for line in lines[2:]
    ]


def source_table(name):
    section = SOURCE.read_text().split(f"## {name}\n", 1)[1]
    return table(section.split("\n## ", 1)[0])


def decorate(ax, title, xlabel, ticks=None):
    ax.set_title(title, loc="left", pad=15, weight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#e4e9ef", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    if ticks is not None:
        ax.set_yticks(np.arange(len(ticks)), ticks)
        ax.invert_yaxis()


def save(fig, name, note):
    fig.text(0.02, 0.025, note, fontsize=9, color="#536176", va="bottom")
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


architecture = source_table("architecture")
transfer = [
    row for row in source_table("transfer")
    if row["seed"] == "4213" and row["method"] == "optimized"
]
pattern = source_table("pattern_controls")
agop = {
    method: source_table(f"agop_{method}_overall")
    for method in ["gumbel", "evolution"]
}
paired = {
    method: source_table(f"agop_{method}_paired")
    for method in ["gumbel", "evolution"]
}
cvae_text = CVAE_SOURCE.read_text()
coverage_section = cvae_text.split("Точные варианты среди 512 prior-сэмплов:", 1)[1]
coverage = table(coverage_section.split("**Вывод:**", 1)[0])

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
fig.subplots_adjust(left=0.15, right=0.96, bottom=0.22, top=0.86, wspace=0.52)
y = np.arange(len(architecture))
params = [int(row["parameters"]) / 1e6 for row in architecture]
axes[0].barh(y, params, color=[BLUE, ORANGE, "#98a6b8", GREEN, GREEN], height=0.58)
decorate(axes[0], "Размер моделей", "Миллионы параметров", LABELS)
axes[0].set_xlim(0, 12)
for pos, value in zip(y, params):
    axes[0].text(value + 0.15, pos, f"{value:.2f}", va="center")
means = [float(row["test_iou_mean"]) for row in architecture]
sds = [float(row["test_iou_sd"]) for row in architecture]
axes[1].errorbar(means, y, xerr=sds, fmt="o", color=BLUE, capsize=4, markersize=7)
decorate(axes[1], "Восстановление масок банка", "IoU, среднее ± SD по трём seeds", LABELS)
axes[1].axvline(1, color="#98a6b8", linestyle="--", linewidth=1)
axes[1].set_xlim(0.3, 1.13)
for pos, value in zip(y, means):
    axes[1].text(value, pos - 0.18, f"{value:.4f}", ha="center", fontsize=10)
save(fig, "01_architectures", "Bank-test: новые учителя исходных задач. SD не является доверительным интервалом.\nУ полностью общей модели один seed остановился рано; размеры моделей и функции потерь различаются.")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
fig.subplots_adjust(left=0.15, right=0.96, bottom=0.23, top=0.85, wspace=0.48)
for ax, metric, title, xlabel in [
    (axes[0], "iou", "Итоговая структура", "IoU на новых задачах"),
    (axes[1], "excess_bce", "Превышение эталонного риска", "Test BCE − эталонная BCE"),
]:
    for family, offset, color in [("A", -0.12, BLUE), ("B", 0.12, ORANGE)]:
        values = [
            float(next(row[metric] for row in transfer
                       if row["kind"] == kind and row["family"] == family))
            for kind in KINDS
        ]
        if metric == "excess_bce":
            values = [max(0, value) for value in values]
        ax.scatter(values, y + offset, color=color, label=f"Семья {family}", s=48)
        for pos, value in zip(y + offset, values):
            ax.annotate(f"{value:.4f}" if metric == "iou" else f"{value:.5f}",
                        (value, pos), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=color)
    decorate(ax, title, xlabel, LABELS)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower left", bbox_to_anchor=(0.15, 0.105),
           ncol=2, frameon=False)
axes[0].set_xlim(0.85, 1.055)
axes[0].axvline(1, color="#98a6b8", linestyle="--", linewidth=1)
axes[1].set_xlim(-0.001, 0.022)
save(fig, "02_transfer", "Полный Gumbel-поиск: seed 4213, восемь стартов, 80 шагов. Это один полный перенос на модель.\nЗначения excess BCE порядка −10⁻⁸ показаны как ноль; выбор маски использует только исходные задачи.")

controls = ["trained_zero", "trained_prior", "uniform_random_32",
            "permuted_reference_columns", "permuted_reference_rows",
            "uncanonicalized_frequency", "canonical_frequency"]
control_labels = ["Обученный D(0)", "Обученный D(z), z ∼ N(0, I)",
                  "Случайные 32 связи", "Переставлены колонки эталона",
                  "Переставлены строки эталона", "Частоты без канонизации",
                  "Частоты после канонизации"]
selected = [next(row for row in pattern if row["kind"] == kind
                 and (kind != "trained_prior" or row["scale"] == "1.0"))
            for kind in controls]
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
fig.subplots_adjust(left=0.26, right=0.97, bottom=0.28, top=0.85, wspace=0.4)
yp = np.arange(len(selected))
for key, offset, color, label in [
    ("mean_raw_iou", -0.16, BLUE, "Без выравнивания"),
    ("mean_aligned_iou", 0.16, ORANGE, "После выравнивания"),
]:
    axes[0].barh(yp + offset, [float(row[key]) for row in selected],
                 height=0.28, color=color, label=label)
decorate(axes[0], "Что меняет выравнивание", "Средний IoU", control_labels)
axes[0].set_xlim(0, 1.05)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower left", bbox_to_anchor=(0.26, 0.105),
           frameon=False, fontsize=9, ncol=2)
prior_rows = [next(row for row in pattern if row["kind"] == "trained_zero")]
prior_rows += [row for row in pattern if row["kind"] == "trained_prior"]
percent = [100 * float(row["raw_exact_fraction"]) for row in prior_rows]
axes[1].bar(np.arange(5), percent, color=["#98a6b8", BLUE, GREEN, BLUE, BLUE], width=0.58)
axes[1].set_title("Точный ответ уже доступен в prior", loc="left", pad=15, weight="bold")
axes[1].set_xticks(np.arange(5), ["z = 0", "0,1", "1", "3", "10"])
axes[1].set_xlabel("Масштаб σ в z ∼ N(0, σ²I)")
axes[1].set_ylabel("Точных масок, %")
axes[1].set_ylim(0, 112)
axes[1].grid(axis="y", color="#e4e9ef")
axes[1].set_axisbelow(True)
for pos, value in enumerate(percent):
    axes[1].text(pos, value + 2, f"{value:.2f}%", ha="center", fontsize=10)
save(fig, "03_pattern", "По 4096 prior-сэмплов на каждый ненулевой масштаб; z = 0 — одна маска. Частоты: 9000 train-масок.\nВыравнивание переставляет только скрытые колонки. Для обученного декодера raw и aligned IoU совпадают.")

coverage_labels = [f"{model}\nseed {seed}" for model in ["surrogate", "ELBO"]
                   for seed in [4213, 4214, 4215]]
exact = np.array([[float(row["Точных"].strip("%")) for row in coverage[i:i + 2]]
                  for i in range(0, len(coverage), 2)])
modes = np.array([[int(row["Вариантов из 4"]) for row in coverage[i:i + 2]]
                  for i in range(0, len(coverage), 2)])
fig, axes = plt.subplots(1, 2, figsize=(10.5, 7))
fig.subplots_adjust(left=0.18, right=0.9, bottom=0.18, top=0.88, wspace=0.6)
for ax, data, title, vmax, cmap in [
    (axes[0], exact, "Точные prior-маски", 60, "Blues"),
    (axes[1], modes, "Покрытие допустимых вариантов", 4, "YlGn"),
]:
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_title(title, loc="left", pad=16, weight="bold")
    ax.set_xticks([0, 1], ["Семья A", "Семья B"])
    ax.set_yticks(np.arange(6), coverage_labels)
    ax.tick_params(length=0)
    for i in range(6):
        for j in range(2):
            label = f"{data[i, j]:.1f}%" if vmax == 60 else f"{data[i, j]:.0f} / 4"
            ax.text(j, i, label, ha="center", va="center", fontsize=12,
                    color="white" if data[i, j] > vmax * 0.6 else TEXT)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.04)
    colorbar.set_label("%" if vmax == 60 else "Число вариантов")
    if vmax == 4:
        colorbar.set_ticks([0, 1, 2, 3, 4])
save(fig, "04_cvae_coverage", "512 prior-сэмплов для каждой пары модель / seed / семья. Частоты округлены в исходном отчёте до 0,1%.\nВысокая частота одного точного ответа не означает покрытия всех четырёх равноценных структур.")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.24, top=0.82, wspace=0.58)
for method, offset, color, label in [
    ("gumbel", -0.12, BLUE, "Gumbel"),
    ("evolution", 0.12, ORANGE, "Evolution"),
]:
    rows = agop[method]
    axes[0].errorbar([float(row["iou_mean"]) for row in rows],
                     np.arange(3) + offset,
                     xerr=[float(row["iou_sd"]) for row in rows], fmt="o",
                     capsize=4, markersize=6, color=color, label=label)
    rows = [row for row in paired[method] if row["control"] == "branch_zero"]
    values = [float(row["agop_minus_control_iou"]) for row in rows]
    axes[1].bar(np.arange(3) + offset, values, width=0.22, color=color, label=label)
    for pos, value in zip(np.arange(3) + offset, values):
        label_offset = (5 if method == "gumbel" else 18) if value >= 0 else -13
        axes[1].annotate(f"{value:+.4f}", (pos, value),
                         xytext=(0, label_offset),
                         textcoords="offset points", ha="center", fontsize=9)
decorate(axes[0], "Среднее качество переноса", "IoU, среднее ± SD по трём seeds",
         ["Полный AGOP", "Карта F", "Нулевая ветка"])
axes[0].set_xlim(0.88, 1.035)
axes[0].axvline(1, color="#98a6b8", linestyle="--", linewidth=1)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower left", bbox_to_anchor=(0.15, 0.11),
           ncol=2, frameon=False)
axes[1].set_title("Эффект AGOP по отдельным seeds", loc="left", pad=15, weight="bold")
axes[1].set_xticks(np.arange(3), ["4213", "4214", "4215"])
axes[1].set_xlabel("Seed инициализации генератора")
axes[1].set_ylabel("IoU AGOP − IoU нулевого контроля")
axes[1].set_ylim(-0.05, 0.06)
axes[1].axhline(0, color="#98a6b8", linewidth=1)
axes[1].grid(axis="y", color="#e4e9ef")
axes[1].set_axisbelow(True)
save(fig, "05_agop", "Четыре новые задачи, три seeds, один фиксированный benchmark. Все три ветки имеют одинаковое число параметров.\nОба поиска: восемь стартов и 80 шагов; число fitness-оценок и объём вычислений различаются.")

data = {
    "architecture": architecture,
    "transfer_seed_4213": transfer,
    "pattern_controls": pattern,
    "cvae_coverage": coverage,
    "agop_overall": agop,
    "agop_paired": paired,
    "sources": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [SOURCE, CVAE_SOURCE]
    },
}
(OUT / "chart_data.json").write_text(
    json.dumps(data, ensure_ascii=False, indent=2) + "\n"
)
print("Saved five figures in PNG/SVG and their source data.")
