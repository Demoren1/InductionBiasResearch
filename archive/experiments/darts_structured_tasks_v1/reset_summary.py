"""Freeze source-only cycle choices and summarize their later transfer results."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import digest, save_json
from .report import categorical, matched_columns, save_figure
from .reset_report import LABELS


def baseline_integrity(frozen_method):
    method = json.loads((frozen_method / "method.json").read_text())
    origin = Path(method["origin"])
    for name, expected in method["origin_sha256"].items():
        if digest(origin / name) != expected:
            raise RuntimeError(f"Original baseline changed: {name}")
    for name, expected in method["copied_sha256"].items():
        if digest(frozen_method / name) != expected:
            raise RuntimeError(f"Archived baseline changed: {name}")
    return {"checked_utc": datetime.now(UTC).isoformat(), "all_hashes_match": True}


def freeze(output, roots, frozen_method):
    candidates = []
    for root in roots:
        if (root / "transfer_confirmation").exists():
            raise RuntimeError("Combined choices must freeze before target generation")
        choices = json.loads((root / "family_selection.json").read_text())["families"]
        for family, row in choices.items():
            candidates.append(
                {
                    "root": str(root.resolve()),
                    "family": family,
                    "selection_mse": row["selection_mse"],
                    "configuration": row["configuration"],
                    "source_frozen_sha256": digest(root / "source_frozen.json"),
                    "selection_sha256": digest(root / "family_selection.json"),
                }
            )
    baseline = next(row for row in candidates if row["family"] == "consensus_baseline")
    proposed = min(
        (
            row
            for row in candidates
            if row["family"] in ("plain_adapt_reset", "consensus_adapt_reset")
        ),
        key=lambda row: row["selection_mse"],
    )
    paired = next(
        row
        for row in candidates
        if row["root"] == proposed["root"]
        and row["family"] == proposed["family"].replace("_reset", "_keep")
    )
    output.mkdir(exist_ok=False)
    save_json(
        output / "choices_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "rules": [
                "Compare both source block lengths before generating any new transfer target.",
                "Choose proposed reset by mean source-selection MSE, never teacher recovery or heldout/target losses.",
                "Paired keep control uses the same block-length experiment; its own split/tau was selected by source pilot.",
                "Short and long share source data and seeds; they are paired ablations, not independent task replications.",
            ],
            "choices": {
                "baseline": baseline,
                "proposed_reset": proposed,
                "paired_keep": paired,
            },
            "candidates": candidates,
            "frozen_method": str(frozen_method.resolve()),
            "baseline_integrity": baseline_integrity(frozen_method),
        },
    )


def report(output, reference):
    frozen = json.loads((output / "choices_frozen.json").read_text())
    rows = {}
    for name, choice in frozen["choices"].items():
        root = Path(choice["root"])
        for file, field in (
            ("source_frozen.json", "source_frozen_sha256"),
            ("family_selection.json", "selection_sha256"),
        ):
            if digest(root / file) != choice[field]:
                raise RuntimeError(f"Source choice changed after freeze: {file}")
        source = json.loads((root / "summary.json").read_text())["families"][
            choice["family"]
        ]
        transfer = json.loads(
            (root / "transfer_confirmation/summary.json").read_text()
        )["aggregate"]
        rows[name] = {
            "choice": choice,
            "source": source,
            "source_mean_test_nmse": float(
                np.mean([r["test_nmse"] for r in source["per_seed"]])
            ),
            "transfer": transfer[choice["family"] + "/learned"],
            "random_sharing": transfer[choice["family"] + "/random_sharing"],
        }
    save_json(
        output / "baseline_integrity.json",
        baseline_integrity(Path(frozen["frozen_method"])),
    )
    save_json(output / "summary.json", rows)
    with np.load(
        Path(rows["baseline"]["choice"]["root"]) / "aligned_states.npz"
    ) as archive:
        truth, order = archive["true_indices"], archive["row_order"]
    figure, axes = plt.subplots(1, 4, figsize=(17.5, 5), constrained_layout=True)
    image = categorical(axes[0], truth[order], 4, "Истинный индексатор")
    for axis, name, title in zip(
        axes[1:3],
        ("baseline", "proposed_reset"),
        ("Согласование, сохранить Z", "Дообучить C, затем Z=C[I]"),
    ):
        row = rows[name]
        with np.load(Path(row["choice"]["root"]) / "aligned_states.npz") as archive:
            indexator = archive[f"row_{row['source']['primary']['row']}_indices"]
        image = categorical(
            axis,
            indexator[order],
            4,
            f"{title}\nточный I: {row['source']['exact_indexators']}/3; ошибок выбранного I: {row['source']['primary']['recovery']['categorical_errors']}",
        )
    figure.colorbar(
        image,
        ax=list(axes[:3]),
        location="bottom",
        ticks=range(-4, 5),
        fraction=0.04,
        shrink=0.6,
    )
    with np.load(
        Path(rows["baseline"]["choice"]["root"]) / "teacher_audit_only.npz"
    ) as archive:
        values = np.r_[0.0, archive["numeric"][:4]]
    with np.load(reference / "selected_states.npz") as archive:
        dense = matched_columns(
            archive["dense_shared_adam_weights"], values[np.abs(truth)] * np.sign(truth)
        )
    maximum = np.abs(dense).max()
    image = axes[3].imshow(dense[order], cmap="RdBu_r", vmin=-maximum, vmax=maximum)
    axes[3].set_title("Плотный Adam: численные веса")
    axes[3].set_xlabel("Скрытый нейрон")
    figure.colorbar(image, ax=axes[3], shrink=0.7)
    save_figure(figure, output / "figures/selected_indexators")
    figure, axes = plt.subplots(2, 4, figsize=(18, 10), constrained_layout=True)
    image = categorical(axes[0, 0], truth[order], 4, "Истинный индексатор")
    for row_number, name, title in (
        (0, "baseline", "Согласование"),
        (1, "proposed_reset", "C-fit, затем Z=C[I]"),
    ):
        source = rows[name]["source"]
        with np.load(
            Path(rows[name]["choice"]["root"]) / "aligned_states.npz"
        ) as archive:
            for axis, record in zip(axes[row_number, 1:], source["per_seed"]):
                errors = record["recovery"]["categorical_errors"]
                image = categorical(
                    axis,
                    archive[f"row_{record['row']}_indices"][order],
                    4,
                    f"{title}, seed {record['seed']}\nошибок I: {errors}/256; test NMSE: {record['test_nmse']:.3g}",
                )
    figure.colorbar(
        image,
        ax=list(axes[0]) + list(axes[1, 1:]),
        location="bottom",
        ticks=range(-4, 5),
        shrink=0.55,
        fraction=0.03,
    )
    image = axes[1, 0].imshow(dense[order], cmap="RdBu_r", vmin=-maximum, vmax=maximum)
    axes[1, 0].set_title("Плотный Adam: численные веса")
    axes[1, 0].set_xlabel("Скрытый нейрон")
    figure.colorbar(image, ax=axes[1, 0], shrink=0.7)
    save_figure(figure, output / "figures/per_seed_indexators")
    lines = [
        "# Дообучение C и циклический сброс Z: итог проверки",
        "",
        "Исходный успешный вариант [зафиксирован отдельно](../../../frozen_methods/deepsets_consensus_exact_v1/README.md). Проверено предложение: обучение Z → точная проекция I/C → обучение C по задаче при фиксированном I → Z=C[I] → следующий цикл. Поиск ведётся по всем 256 позициям с четырьмя свободными кодами и связанными отрицательными двойниками, без знания истинной маски и групп.",
        "",
        "Выбор лучшего нового режима и длины блока выполнен по source-selection до генерации новых функций переноса; см. choices_frozen.json. В короткой серии 39 конфигураций, в длинной 37. Обе используют одинаковые исходные наблюдения, pilot 341/342 и confirmation 343–345. Бюджет каждого запуска: 2000 шагов разогрева и 10000 дальнейших обновлений по задаче. В короткой серии 200 циклов по 50 шагов, в длинной 10 циклов по 1000 шагов; исходный контроль всегда 200×50. Число градиентных проходов одинаково, FLOPs могут различаться. При удлинении блока меняется также частота проекций и сбросов: эффект числа шагов C отдельно от частоты не изолирован.",
        "",
        "| Режим, выбранный по source | Z/C за цикл | Mean source test NMSE, 3 seed | Точный I | Mean fresh 32, та же функция | Перенос, query-selected | Перенос, все 8 стартов |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    titles = {
        "baseline": "Исходное согласование",
        "proposed_reset": "Дообучить C и присвоить Z=C[I]",
        "paired_keep": "Чередовать Z/C, сохранить Z",
    }
    for name in ("baseline", "proposed_reset", "paired_keep"):
        row, source = rows[name], rows[name]["source"]
        config, target = source["settings"], row["transfer"]
        lines.append(
            f"| {titles[name]} | {config['z_steps']}/{config['c_steps']} | {row['source_mean_test_nmse']:.6g} | {source['exact_indexators']}/3 | {source['fresh']['mean_test_nmse']:.6g} | {target['mean_query_selected_test_nmse']:.6g} | {target['mean_all_restarts_test_nmse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "![Все три запуска выбранных режимов и Adam](figures/per_seed_indexators.png)",
            "",
            "Для каждого seed показан его собственный снимок, выбранный по source-selection; провальные запуски не удалены. Численный test относится к той же исходной функции. Следующая компактная хитмапа показывает только лучшие по selection структуры, используемые в переносе.",
            "",
            "![Выбранные для переноса индексаторы и Adam](figures/selected_indexators.png)",
            "",
            "## Все source-абляции",
            "",
        ]
    )
    roots = list(dict.fromkeys(row["root"] for row in frozen["candidates"]))
    for root_name in roots:
        root = Path(root_name)
        source = json.loads((root / "summary.json").read_text())["families"]
        lines.extend(
            [
                "| Блок | Семейство | Z/C | Selection NMSE | Mean source test NMSE | Точный I |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, row in source.items():
            config = row["settings"]
            period = config.get("cycle_updates", 50)
            lines.append(
                f"| {period} | {LABELS[name]} | {config['z_steps']}/{config['c_steps']} | {row['selection_nmse_mean']:.6g} | {np.mean([r['test_nmse'] for r in row['per_seed']]):.6g} | {row['exact_indexators']}/3 |"
            )
        lines.extend(
            [
                "",
                f"[Полная таблица, кривые и индексаторы {root.name}](../{root.name}/REPORT.md).",
                "",
            ]
        )
    control_root = Path(rows["baseline"]["choice"]["root"])
    targets = json.loads(
        (control_root / "transfer_confirmation/summary.json").read_text()
    )["aggregate"]
    lines.extend(
        [
            "## Контроли и ограничения",
            "",
            "Перенос: шесть новых функций с новыми коэффициентами, seed-offset 15100000. Исходный I фиксирован, все численные параметры заново инициализируются; 8 стартов × 3000 обновлений, 6 численных настроек. Query выбирает старт, mean selection — настройку; затем открывается test из 8192 примеров. Обе серии оцениваются на одинаковых задачах. Dense-контроль длинной серии повторно используется только после проверки идентичности данных и протокола.",
            "",
        ]
    )
    for name in ("oracle", "dense_adam"):
        row = targets[name]
        lines.append(
            f"{name}: mean query-selected test NMSE {row['mean_query_selected_test_nmse']:.6g}; mean всех 8 стартов {row['mean_all_restarts_test_nmse']:.6g}."
        )
    lines.extend(
        [
            "",
            "На новых функциях отдельные численные старты срываются даже при истинном I; почти нулевой результат после выбора по query не означает успех всех восьми стартов. Средние по всем стартам приведены отдельно. У эквивалентных точных I одна представимость; небольшие различия ошибок связаны с численным обучением, а не преимуществом над истинной структурой.",
        ]
    )
    lines.extend(
        [
            "",
            "Длинный предложенный цикл восстановил точный I в 1/3 подтверждающих запусков, короткий — в 0/3; исходное согласование — в 3/3. Поэтому нельзя считать предложенный цикл неработающим, но он не улучшил воспроизводимость исходного варианта. Выбранный по валидации успешный I нового метода точен; его качество переноса следует отличать от вероятности получить такой I одним запуском поиска.",
            "",
            "Циклический сброс по построению обнуляет численный разрыв Z−C[I] в момент присваивания. Он при этом удаляет непрерывные различия между позициями, получившими один код. В этой серии устранение разрыва само по себе не обеспечило устойчивого восстановления структуры. Сохранение Z оставляет эти различия доступными последующим шагам поиска. Это объяснение согласуется с измерениями, но не является доказательством для всех расписаний.",
            "",
            "Полные счётчики Z/C/head, ошибок присваивания, числа проекций и разных колонок находятся в cycle_diagnostics.json каждой серии; train/query кривые — в их figures. Исходный архив и старые результаты проверены по SHA256 и не изменены. Программные тесты не создавались и не запускались; Ruff прошёл. Результат относится к синтетическому DeepSets одной скрытой структуры, новым численным коэффициентам и проверенным расписаниям. Новые запуски на Pattern/синусах/изображениях здесь не выполнялись.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))
    (output / "reporting_code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    save_json(
        output / "complete.json",
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "sha256": {
                str(path.relative_to(output)): digest(path)
                for path in output.rglob("*")
                if path.is_file() and path.name != "complete.json"
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", choices=("freeze", "report"), required=True)
    parser.add_argument("--sources", nargs=2, type=Path)
    parser.add_argument("--frozen-method", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.root, args.sources, args.frozen_method)
    else:
        report(args.root, args.reference)


if __name__ == "__main__":
    main()
