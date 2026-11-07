"""Frozen G17 native LC comparison including the same-data warm-up control."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.darts_structured_tasks_v1.data import digest, save_json

LABELS = {
    "primary_lc": "Native LC: penalty from start",
    "primary_penalty": "Penalty only: from start",
    "primary_dense_gn": "Dense GN + explicit DP readout",
    "warmup_lc": "Native LC: 24 B-training solves first",
    "warmup_penalty": "Penalty only: 24 B-training solves first",
}


def read_models(root, stage):
    frozen = json.loads((root / "models_frozen.json").read_text())
    audited = json.loads((root / "audit.json").read_text())["records"]
    result = []
    for row in audited:
        if not row["selected_global_setting"]:
            continue
        path = root / "models" / f"job_{row['job_id']:03d}"
        for suffix in (".npz", ".json"):
            candidate = path.with_suffix(suffix)
            if digest(candidate) != frozen["sha256"][str(candidate.relative_to(root))]:
                raise RuntimeError("Frozen G17 source changed")
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path.with_suffix(".npz")) as a:
            arrays = dict(a)
        with np.load(path.parent / (path.name + "_audit_only.npz")) as a:
            audit_arrays = dict(a)
        result.append(
            {
                **row,
                "name": stage + "_" + row["job"]["method"],
                "meta": meta,
                "arrays": arrays,
                "audit_arrays": audit_arrays,
            }
        )
    return result


def generate(root):
    report = root / "report"
    report.mkdir(exist_ok=False)
    plan = json.loads((root / "protocol.json").read_text())
    results = read_models(root, "primary") + read_models(
        root / "same_data_warmup", "warmup"
    )
    summary = []
    for name in LABELS:
        for initial in plan["initializations"]:
            selected = [
                row
                for row in results
                if row["name"] == name and row["job"]["initialization"] == initial
            ]
            checkpoints = [
                row["meta"]["records"][row["meta"]["selected_checkpoint"]]
                for row in selected
            ]
            summary.append(
                {
                    "method": name,
                    "initialization": initial,
                    "exact": sum(row["recovery"]["exact_I"] for row in selected),
                    "cases": len(selected),
                    "mean_I_errors": float(
                        np.mean(
                            [row["recovery"]["categorical_errors"] for row in selected]
                        )
                    ),
                    "mean_validation_mse": float(
                        np.mean([row["selected_validation_mse"] for row in selected])
                    ),
                    "mean_clean_validation_mse": float(
                        np.mean(
                            [row["selected_clean_validation_mse"] for row in selected]
                        )
                    ),
                    "mean_raw_validation_mse_at_selected": float(
                        np.mean([row["validation_mse_raw"] for row in checkpoints])
                    ),
                    "mean_L_solves": float(
                        np.mean([row["meta"]["l_solves"] for row in selected])
                    ),
                    "mean_refit_solves": float(
                        np.mean([row["meta"]["refit_solves"] for row in selected])
                    ),
                    "mean_seconds": float(
                        np.mean([row["meta"]["seconds"] for row in selected])
                    ),
                }
            )
    with (report / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    save_json(
        report / "summary.json",
        {
            "comparisons": summary,
            "primary_freeze_sha256": digest(root / "models_frozen.json"),
            "warmup_freeze_sha256": digest(
                root / "same_data_warmup/models_frozen.json"
            ),
            "scope": "80 source-fit, source-only selection,4 opened geometries and2 references. Known controls; no own-method superiority, independent confirmation, nonlinear coverage or transfer claim.",
        },
    )
    plot_curves(report, results)
    plot_indexators(report, results, plan)
    plot_weights(report, results, plan)
    write_report(report, summary)


def chosen(results, name, family, initial="cold"):
    return next(
        row
        for row in results
        if row["name"] == name
        and row["job"]["family"] == family
        and row["job"]["initialization"] == initial
    )


def plot_curves(report, results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    names = ["primary_lc", "warmup_lc", "primary_dense_gn", "warmup_penalty"]
    for line, family in enumerate((8401, 8402)):
        for name in names:
            row = chosen(results, name, family)
            trace = row["meta"]["records"]
            x = [point["total_solves"] for point in trace]
            axes[line, 0].plot(
                x, [point["source_mse_polished"] for point in trace], label=LABELS[name]
            )
            axes[line, 1].plot(
                x, [point["validation_mse"] for point in trace], label=LABELS[name]
            )
            axes[line, 2].plot(x, row["trace_mask_recall"], label=LABELS[name])
            selected = row["meta"]["selected_checkpoint"]
            axes[line, 1].scatter(
                x[selected], trace[selected]["validation_mse"], s=50, marker="x"
            )
        for ax in axes[line]:
            ax.set_xlabel(
                "Actual GN linear solves (learning + fixed-I numerical refit)"
            )
        axes[line, 0].set_ylabel(
            f"Family {family}, cold\nSource MSE (mean of 2 labels)"
        )
        axes[line, 1].set_ylabel("Independent noisy validation MSE")
        axes[line, 2].set_ylabel("True-mask recall (audit only)")
        axes[line, 2].set_ylim(0.3, 1.03)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "G17: source loss, validation selection and recall; known controls on the same B"
    )
    fig.savefig(report / "nonlinear_lc_loss_recall.png", dpi=180)
    plt.close(fig)


def plot_indexators(report, results, plan):
    names = ["primary_lc", "primary_dense_gn", "warmup_lc", "warmup_penalty"]
    fig, axes = plt.subplots(4, 5, figsize=(14, 11), constrained_layout=True)
    for line, family in enumerate(plan["families"]):
        models = [chosen(results, name, family) for name in names]
        images = [
            models[0]["audit_arrays"]["truth"],
            *[row["audit_arrays"]["selected_aligned"] for row in models],
        ]
        titles = [
            "True I",
            "LC from start",
            "Dense GN + DP",
            "LC after B warm-up",
            "Penalty after B warm-up",
        ]
        for col, (data, title) in enumerate(zip(images, titles, strict=True)):
            im = axes[line, col].imshow(data, cmap="RdBu_r", vmin=-4, vmax=4)
            if line == 0:
                axes[line, col].set_title(title, fontsize=9)
            axes[line, col].set_xticks([0, 7, 15])
            axes[line, col].set_yticks([0, 7, 15])
            if col == 0:
                axes[line, col].set_ylabel(f"Family {family}\nInput coordinate")
            if line == 3:
                axes[line, col].set_xlabel("Hidden neuron (audit aligned)")
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.7,
        ticks=range(-4, 5),
        label="Signed code ID; 0 = disconnected",
    )
    fig.suptitle(
        "G17 chosen indexators: all four cold cases; source-validation choices frozen first\n"
        "Code names/signs/hidden columns matched only for audit"
    )
    fig.savefig(report / "nonlinear_lc_indexators.png", dpi=180)
    plt.close(fig)


def plot_weights(report, results, plan):
    fig, axes = plt.subplots(2, 5, figsize=(14, 7), constrained_layout=True)
    shape = (plan["inputs"], plan["hidden"])
    for line, family in enumerate((8401, 8402)):
        primary = chosen(results, "primary_lc", family)
        warm = chosen(results, "warmup_lc", family)
        truth = primary["audit_arrays"]["truth_weights"]
        checkpoint = warm["meta"]["selected_checkpoint"]
        arrays = [
            truth,
            primary["arrays"]["reference_theta"][:-2].reshape(shape),
            primary["arrays"]["selected_theta"][:-2].reshape(shape),
            warm["arrays"]["trace_latent_theta"][checkpoint, :-2].reshape(shape),
            warm["arrays"]["selected_theta"][:-2].reshape(shape),
        ]
        limit = max(float(np.abs(data).max()) for data in arrays)
        for col, (data, title) in enumerate(
            zip(
                arrays,
                [
                    "True W",
                    "Initial W",
                    "LC from start: hard W",
                    "LC warm-up: latent W",
                    "LC warm-up: hard W",
                ],
                strict=True,
            )
        ):
            if col:
                costs = ((data[:, :, None] - truth[:, None, :]) ** 2).sum(0)
                source, target = linear_sum_assignment(costs)
                data = data[:, source[np.argsort(target)]]
            im = axes[line, col].imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[line, col].set_title(title, fontsize=9)
            axes[line, col].set_xlabel("Hidden neuron")
        axes[line, 0].set_ylabel(f"Family {family}, cold\nInput coordinate")
        fig.colorbar(
            im, ax=axes[line].tolist(), shrink=0.8, label="Actual numerical weight"
        )
    fig.suptitle(
        "Actual numerical weights; hidden permutations matched only for visualization"
    )
    fig.savefig(report / "nonlinear_lc_weights.png", dpi=180)
    plt.close(fig)


def write_report(report, summary):
    lines = [
        "# G17: native LC и сильный плотный контроль на nonlinear DeepSets",
        "",
        "Полноценный известный LC-контроль после короткого обучения на той жеB восстанавливает4/4 холодных и4/4 тёплых I. Плотный Gauss–Newton с явным DP-readout также4/4+4/4. Ранний LC без этой подготовки даёт3/4 cold. **Новый собственный метод, превосходящий эти контроли, не получен.**",
        "",
        "## Протокол и адаптация",
        "",
        "THEORY.md§24 до primary48fit.4 открытые G9/G16 геометрии8401..8404, поcold/first-warm(moment_pairs8388608),16inputs/16ReLU,4signed-кода+fixed0,общий positive bias+offset. SourceB1048576 входов, по2Gaussian noisy labels sigma.2:2097152 labels. Эти наблюдения и validation16384 точно те же, чтоG16, проверены SHA. Reference обучался наA; дорогая warm-moment стоимость сохраняется. Это8reference условий/4геометрии, не8 независимых структур и не новая confirmation.",
        "",
        "LC — objective-level адаптация [Carreira-Perpiñán–Idelbayev2017](https://arxiv.org/abs/1707.04319): augmented Lagrangian/scaleddual, true-ReLU backtracking вGN L-step, exact signed scalar-DP C-step. Это native индексатор из собственного C-step; дополнительный внешний kmeans не добавлен. Penalty-only задаётu=0. DenseGN+DP — явно дополнительный readout. Все fixed-I checkpoint получают одинаковый source-only fit4C+b+d до12GN-solves; этот polishing не меняетI и не возвращается в latent trajectory. Это не неизменённые авторские скрипты и не finite-run гарантия глобального optimum.",
        "",
        "Primary48:8reference×3метода×2settings. LC/penalty24mu stages от1e−6/1e−4 до10·maxeig(H0), max6L-solves/stage. Dense имеет тот жеmax144 L-solves иdamping1e−10/1e−4. Всеcheckpoint выбираются по observed validation; одна global setting/метод — по среднему8source-ошибок. Wholefreeze/HPO до нового trueI аудита. Source-MSE относится к среднему2labels сnoisevariance.02, validation — к single-label variance.04; их абсолютные уровни нельзя напрямую читать как overfit-gap.",
        "",
        "После primary обнаружено: denseGN с малымdamping4/4cold, LC3/4. THEORY§24.5 до дополнительного запуска фиксирует4stage(mu=u=0) наB, затем20positive-stage. Это до24 unpenalized L-solves **внутри того жеmax144**, не добавочный оптимизационный бюджет. Same_data_warmup содержит32 новых LC/penalty fit; old dense16 не повторяются. Сохраняется и primaryрезультат. Изменение распределяет шаги между warm-up иpenaltyпутём и меняет сеткуpositive-mu; это не чистая изоляция одного скаляра и не доказательство единственной причины сбоя.",
        "",
        "## Результаты source-validation selection",
        "",
        "|Метод|Cold exact I|First-warm exact I|Cold mean observed validation MSE|Cold mean clean validation MSE (audit)|",
        "|---|---:|---:|---:|---:|",
    ]
    for name, label in LABELS.items():
        cold = next(
            row
            for row in summary
            if row["method"] == name and row["initialization"] == "cold"
        )
        warm = next(
            row
            for row in summary
            if row["method"] == name and row["initialization"] != "cold"
        )
        lines.append(
            f"|{label}|{cold['exact']}/{cold['cases']}|{warm['exact']}/{warm['cases']}|"
            f"{cold['mean_validation_mse']:.8f}|{cold['mean_clean_validation_mse']:.8g}|"
        )
    lines.extend(
        [
            "",
            "G16 на тех жеB/reference: independent one-step+scalarDP2/4cold,4/4first-warm. Новый результат плотного контроля показывает: **один frozen-Jacobian шаг был недостаточен; повторная линеаризация и численное обучение наB исправляют все4cold**. Отсюда не следует, что исходнаяA-модель находилась в доказанном строгом local minimum, или что любой Adam/SGD обязан провалиться. Это также не новый structural метод — dense control с явным readout.",
            "",
            "Warm-up усиливает LC до того же4/4; нельзя объявлять новую схему лучше LC по сравнению только с его ранним более слабым запуском. Penalty-only послеwarm-up тоже4/4: отдельный выигрышdual-переменной здесь не показан. Все выигравшие global settings —row0. Индивидуальные raw/polished losses, actual learning/refit solve counts, acceptedupdates, mu/primal residual иteacher recall сохранены. Полный путь не обязан монотонно улучшать validation; выбирается checkpoint, не последнийstage.",
            "",
            "## Область вывода и следующий вопрос",
            "",
            "80 новых source-fit, все6свободныхGPU0/1/2/3/6/7; чужие4/5не затрагивались. Software tests не создавались/не запускались. Данные иmaxчислоL-solves согласованы, фактическое числоGN/refit ивремя отличаются и вынесенывcomparison.csv; FLOP/time-matched победа не заявлена. Noisy G9 primarygates/abstentions не менялись. G17 — один underlying source-function на геометрию, нет нового target-C/transfer или независимого multi-function common-I подтверждения.",
            "",
            "Простое исправление численного этапа уже снимает рассматриваемые cold ошибки. Следующая собственная схема должна показывать самостоятельный эффект сверх сильного dense/nativeLC, а не приписывать им неизбежный optimizer failure. Для большой цели остаются новый механизм извлечения общей структуры, независимые геометрии/разныеsource-C, misspecified noise и перенос толькоI. Дальнейший вариант должен сначала объяснить, какую наблюдаемую структурную информацию он использует, и пройти проверку близкой литературы.",
            "",
            "![Loss и recall](nonlinear_lc_loss_recall.png)",
            "",
            "![Выбранные индексаторы](nonlinear_lc_indexators.png)",
            "",
            "![Численные веса](nonlinear_lc_weights.png)",
        ]
    )
    (report / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)


if __name__ == "__main__":
    main()
