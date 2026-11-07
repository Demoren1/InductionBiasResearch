"""Report proper joint fusion against the unchanged G4 competitor cohort."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.darts_structured_tasks_v1.data import save_json


def report(root):
    plan = json.loads((root / "protocol.json").read_text())
    spec = json.loads((root / "confirmation_registration.json").read_text())
    choice = json.loads((root / "hpo_frozen.json").read_text())
    files = sorted((root / "confirmation").glob("*/audit.json"))
    cases = [json.loads(p.read_text()) for p in files]
    if len(cases) != len(spec["jobs"]):
        raise RuntimeError("Report needs the complete frozen additional cohort")
    output = root / "report"
    output.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for case in cases:
        groups[(case["job"]["regime"], case["job"]["n"])].append(case)
    rows = []
    for (regime, n), group in sorted(groups.items()):
        issued = [c for c in group if c["returned_index"]]
        gaps, primal, dual = [], [], []
        for case in issued:
            i = cases.index(case)
            meta = json.loads((files[i].parent / "model.json").read_text())
            with np.load(files[i].parent / "model.npz") as a:
                step = int(a["selected_step"][0, meta["selected_lambda_index"]])
            point = next(h for h in meta["history"] if h["step"] == step)
            gaps.append(point["relative_primal_dual_gap"][0])
            primal.append(point["primal_residual"][0])
            dual.append(point["dual_residual"][0])
        row = {
            "method": "joint_adaptive_signed_fusion",
            "regime": regime,
            "n": n,
            "families": len(group),
            "issued": len(issued),
            "exact": sum(c["recovery"]["exact_I"] for c in issued),
            "mask_f1": float(np.mean([c["recovery"]["mask_f1"] for c in issued])),
            "max_selected_relative_primal_dual_gap": max(gaps),
            "max_selected_primal_residual": max(primal),
            "max_selected_dual_residual": max(dual),
        }
        for support in spec["transfer"]["support"]:
            row[f"mse_{support}"] = float(
                np.mean([c["transfer"][str(support)]["mean_mse"] for c in issued])
            )
        rows.append(row)
    with (
        Path(plan["g4_root"]) / "report/validation_selected_comparison.csv"
    ).open() as f:
        previous = list(csv.DictReader(f))
    names = {
        "yeh__native_or_decoded": "Yeh, native MAP",
        "lc__native_or_decoded": "LC, native I",
        "lutq__native_or_decoded": "LUT-Q, native I",
        "noise_whitened_pca": "Noise-whitened PCA + readout",
        "codebook_crossfit": "Codebook-crossfit F graph",
        "oscar_prox": "Earlier independent OSCAR",
        "joint_adaptive_signed_fusion": "Joint adaptive signed fusion",
    }
    comparison = [
        r
        for r in previous
        if r["method"] in names and (r["regime"], int(r["n"])) in groups
    ] + rows
    save_json(
        output / "summary.json",
        {
            "rows": rows,
            "comparison": comparison,
            "selected_settings": choice["settings"],
            "stage": spec["stage"],
            "independent_geometries": 12,
            "scope": "Additional known control on already-opened G4 cohort, not new independent confirmation. Native/equality graph, no k-means readout.",
        },
    )
    keys = sorted(set().union(*(r.keys() for r in comparison)))
    with (output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(comparison)
    regimes = [
        ("primary", 8192),
        ("primary", 131072),
        ("low_noise", 131072),
        ("high_noise", 131072),
    ]
    methods = list(names)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(regimes))
    width = 0.8 / len(methods)
    for number, method in enumerate(methods):
        selected = [
            next(
                (
                    r
                    for r in comparison
                    if r["method"] == method
                    and r["regime"] == regime
                    and int(r["n"]) == n
                ),
                None,
            )
            for regime, n in regimes
        ]
        axes[0].bar(
            x + (number - (len(methods) - 1) / 2) * width,
            [
                int(r["exact"]) / int(r["families"]) if r is not None else np.nan
                for r in selected
            ],
            width=width,
            label=names[method],
        )
        axes[1].plot(
            x,
            [
                float(r["mse_4"]) if r is not None and r.get("mse_4") else np.nan
                for r in selected
            ],
            marker=".",
            label=names[method],
        )
    labels = ["noise1\nn8192", "noise1\nn131072", "noise.2\nn131072", "noise2\nn131072"]
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Exact I fraction; nonissue = failure")
    axes[1].set_ylabel("Fresh-C support4 population clean MSE")
    axes[1].set_yscale("log")
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(output / "joint_fusion_comparison.png", dpi=170)
    plt.close(fig)
    selected_cases = [
        (i, c)
        for i, c in enumerate(cases)
        if c["job"]["family"]["id"] == 7101 and c["job"]["regime"] == "primary"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for i, case in selected_cases:
        folder = files[i].parent
        meta = json.loads((folder / "model.json").read_text())
        history = meta["history"]
        steps = [h["step"] for h in history]
        axes[0].plot(
            steps,
            [h["source_mse"][0] for h in history],
            label=f"source n={case['job']['n']}",
        )
        axes[0].plot(
            steps,
            [h["validation_mse"][0] for h in history],
            marker=".",
            linestyle="none",
            label=f"validation, eligible I, n={case['job']['n']}",
        )
        axes[1].plot(
            steps,
            np.array(case["trace_mask_recall"])[:, 0],
            label=f"n={case['job']['n']}",
        )
        axes[2].plot(
            steps,
            [max(h["relative_primal_dual_gap"][0], 1e-14) for h in history],
            label=f"n={case['job']['n']}",
        )
    axes[0].set_ylabel("Observed MSE")
    axes[0].set_title("Source / independent source validation", fontsize=10)
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Native constraint mask recall (diagnostic)")
    axes[2].set_ylabel("Feasible relative primal-dual objective gap")
    axes[2].set_yscale("log")
    axes[2].axhline(
        plan["solver"]["tolerance"], color="black", linestyle="--", linewidth=1
    )
    for ax in axes:
        ax.set_xlabel("Actual ADMM steps along increasing lambda path")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "joint_fusion_loss_recall_gap.png", dpi=170)
    plt.close(fig)
    low = next(files[i].parent for i, c in selected_cases if c["job"]["n"] == 8192)
    high = next(files[i].parent for i, c in selected_cases if c["job"]["n"] == 131072)
    with np.load(low / "aligned_audit_only.npz") as a:
        small, truth = a["index"], a["truth"]
    with np.load(high / "aligned_audit_only.npz") as a:
        large = a["index"]
    with np.load(low / "model.npz") as a:
        numerical = a["selected_weights"][0]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
    for ax, value, title in zip(
        axes,
        (numerical, small, large, truth),
        (
            "Actual numerical fusion W, task0",
            "Native I, n8192",
            "Native I, n131072",
            "True I (audit only)",
        ),
    ):
        image = ax.imshow(
            value,
            cmap="coolwarm",
            vmin=-float(np.abs(numerical).max()) if ax == axes[0] else -3,
            vmax=float(np.abs(numerical).max()) if ax == axes[0] else 3,
        )
        ax.set_title(title, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle("First registered geometry7101, primary noise1; source-only choices")
    fig.tight_layout()
    fig.savefig(output / "joint_fusion_indexators.png", dpi=170)
    plt.close(fig)
    lines = [
        "# G12: полноценный совместный fusion-контроль",
        "",
        "Это добавление известного objective-level PACS/convex-clustering контроля к уже открытым G4 данным, не новая независимая confirmation и не новый наш метод. Профили всех128 функций фьюзятся совместно; нули/суммы/разности имеют group-L2 penalties. Используется полный sample Gram. Native I читается по точным нулям proximal переменных, без k-means. Числа C на validation/transfer обучаются заново при выбранных равенствах.",
        "",
        "## Выбор и численная достоверность",
        "",
        "На4 прежних pilot-геометриях×2n проверены gamma0/1/2/4/8 и r0=.1/1, семь lambda. По mean source-validation выбраны gamma8,r0=.1. Каждое новое source-наблюдение выбирает lambda и сходящийся checkpoint по тем же32 full-C support4/query128 функциям, что G4. Учитель/target в выборе не участвуют. Все48 дополнительных source-моделей заморожены до structural scoring; парный перенос использует те же100 новых коэффициентных функций, что G4.",
        "",
        "У выбранных состояний проверены primal/dual residual и feasible relative primal-dual gap<=1e−6. Глобальная оптимальность выпуклой цели не означает истинность I. Adaptive rho — численная эвристика; gap служит прямым ограничением погрешности цели, а не обещанием сходимости изменяющегося rho.",
        "",
        "| Режим | Exact I | MSE переноса, support4 |",
        "|---|---:|---:|",
    ]
    for regime, n in regimes:
        row = next(r for r in rows if r["regime"] == regime and r["n"] == n)
        lines.append(
            f"| {regime}, n{n} | {row['exact']}/{row['families']} | {row['mse_4']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Диагноз",
            "",
            "Приn8192 pilot0/4: удалены все14 слабых позиций каждого conv и3 каждого signed. Приn131072 pilot4/4. Прямая/двойственная цель уже близки; потеря группы не объясняется лишь незавершённым solver. На6101,n8192 true-equality squaredRMS≈.001338 почти совпадает с estimated noise≈.001318; слабый код против0 после коррекции имеет≈.000642. Raw adaptive distances содержат substantial noise-floor.",
            "",
            "По этому диагнозу THEORY§20 до G13 выводит noise-corrected fusion-веса. Нужны отдельные raw shrinking-floor и Gaussian-kernel контроли. Грубая lambda-сетка остаётся ограничением этой реализации: нельзя выдавать G12 за отрицание всех хорошо настроенных PACS/convex-clustering вариантов. Приоритет общего noise-correction/embedding+fusion принципа не заявляется; PCMF и другие noisy-clustering предшественники должны учитываться.",
            "",
            "## Сравнение и ограничения",
            "",
            "Joint-контроль сильнее прежнего независимого OSCAR при большом n, но не превосходит val-selected LC/Yeh/LUT-Q и whitening-PCA. Более сильный solver/адаптивные веса/совместная норма изменены одновременно; различие с прежним OSCAR нельзя приписать одной joint-норме. Вся серия остаётся линейной. Independent nonlinear native comparison, robust noise и приоритет собственного компонента ещё не закрыты.",
            "",
            "- [Сравнение с неизменёнными G4 конкурентами](joint_fusion_comparison.png)",
            "- [Реальные ADMM loss/recall/gap кривые](joint_fusion_loss_recall_gap.png)",
            "- [Численные W и native I](joint_fusion_indexators.png)",
            "- [Таблица](comparison.csv), [summary](summary.json), [pilot-диагноз](../pilot_diagnosis.json).",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"rows": rows, "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)


if __name__ == "__main__":
    main()
