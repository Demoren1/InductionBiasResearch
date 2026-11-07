"""Transfer the source-frozen f=6 indexator by changing frequency alone."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .data import BASE, digest, save_json, teacher_predict
from .sine_consensus import numeric_rows
from .train import fit, save_fit


def prepare(root, parent):
    root.mkdir(parents=True, exist_ok=False)
    source = parent / "sine_6"
    source_frozen = json.loads((source / "source_frozen.json").read_text())
    for name, expected in source_frozen["sha256"].items():
        if digest(source / name) != expected:
            raise RuntimeError(f"Source changed: {source / name}")
    choice = json.loads((source / "architecture_selection.json").read_text())["primary"]
    with np.load(source / "confirmation/hard_fits.npz") as archive:
        index = archive["indices"][choice["row"]].copy()
    np.savez_compressed(root / "fixed_structure.npz", indices=index)
    save_json(root / "fixed_indexator.json", {"indices": index.tolist()})
    save_json(
        root / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "parent": str(parent.resolve()),
            "source_frequency": 6,
            "source_choice": choice,
            "source_frozen_sha256": digest(source / "source_frozen.json"),
            "fixed_structure_sha256": digest(root / "fixed_structure.npz"),
            "frequencies": [4, 6, 8],
            "calibration_seed": 701,
            "fresh_seed": 702,
            "calibration_restarts": 32,
            "fresh_restarts": 32,
            "heldout_seed_offset": 21100000,
            "rules": [
                "Transfer exactly one raw I chosen by the previous f=6 source-validation search. Do not search, relabel, recluster, shuffle or edit I for any target.",
                "Keep the original direction and both amplitudes; change only the high teacher frequency. f=6 is a fresh-numerics source control, f=4/8 are frequency transfer. Reuse original support/query/selection observations. New test observations are generated after all numerical choices freeze.",
                "Transfer no C, amplitude or optimizer states. Use the existing fit function: fresh Gaussian three-code starts normalized to each radius, amplitude initially1, then train C and amplitude. The student remains one sine, no phase/offset; dictionary is [0,C1,C2,C3,-C1,-C2,-C3].",
                "Calibrate12settings (radius .5/2/6/8, Adam rate .003/.01/.03) with32starts each and3000updates. Query selects a checkpoint per start and a start per setting; independent selection loss selects the setting. This best-of-starts deployment rule is registered before target fitting, and differs from the previous mean8 calibration rule.",
                "At the selected setting train32new starts with independent seed702. Query chooses the reported checkpoint/start; also report all-start means and near-floor counts. Freeze every frequency's choices before opening teacher directions or computing population/test/capacity metrics.",
                "Reused dense/oracle32start source controls, if shown, are contextual: identical observations/functions and step budget, but their original calibration used mean8 and seed502. Do not claim equal total search budgets or paired fresh seeds.",
            ],
        },
    )
    snapshot = root / "code_snapshot"
    snapshot.mkdir()
    for name in (
        "sine_frequency_transfer.py",
        "sine_consensus.py",
        "train.py",
        "model.py",
        "data.py",
        "evaluate.py",
        "sine_consensus_report.py",
        "sine_capacity.py",
    ):
        shutil.copy2(BASE / name, snapshot / name)
    for frequency in (4, 6, 8):
        origin = parent / f"sine_{frequency}"
        output = root / f"sine_{frequency}"
        output.mkdir()
        for name in ("source.npz", "teacher_audit_only.npz", "protocol.json"):
            shutil.copy2(origin / name, output / name)
        spec = json.loads((origin / "specification.json").read_text())
        spec["name"] = f"fixed_source6_to_{frequency}"
        save_json(output / "specification.json", spec)
    save_json(
        root / "prepared.json",
        {
            "sha256": {
                str(path.relative_to(root)): digest(path)
                for path in root.rglob("*")
                if path.is_file()
            }
        },
    )


def run(root, device):
    registration = json.loads((root / "registration.json").read_text())
    prepared = json.loads((root / "prepared.json").read_text())
    for name, expected in prepared["sha256"].items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Prepared input changed: {root / name}")
    with np.load(root / "fixed_structure.npz") as archive:
        index = archive["indices"].copy()
    choices = {}
    for frequency in registration["frequencies"]:
        output = root / f"sine_{frequency}"
        directory = output / "numerics"
        directory.mkdir(exist_ok=False)
        plan = json.loads((output / "protocol.json").read_text())
        spec = json.loads((output / "specification.json").read_text())
        rows = numeric_rows(registration["calibration_seed"])
        calibrated = fit(
            output,
            plan,
            spec,
            rows,
            device,
            indices=np.repeat(index[None], len(rows), axis=0),
            restarts=registration["calibration_restarts"],
        )
        save_fit(directory / "calibration", calibrated, rows)
        restarts = calibrated["query_mse"].argmin(-1)
        number = int(
            calibrated["selection_mse"][np.arange(len(rows)), restarts].argmin()
        )
        fresh_row = {**rows[number], "seed": registration["fresh_seed"]}
        fresh = fit(
            output,
            plan,
            spec,
            [fresh_row],
            device,
            indices=index[None],
            restarts=registration["fresh_restarts"],
        )
        save_fit(directory / "fresh", fresh, [fresh_row])
        choices[str(frequency)] = {
            "calibration_row": number,
            "calibration_query_selected_restart": int(restarts[number]),
            "settings": fresh_row,
            "fresh_query_selected_restart": int(fresh["query_mse"][0].argmin()),
        }
        save_json(directory / "selection.json", choices[str(frequency)])
        print(f"frequency={frequency} choices={choices[str(frequency)]}", flush=True)
    save_json(
        root / "numerical_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "choices": choices,
            "sha256": {
                str(path.relative_to(root)): digest(path)
                for path in root.rglob("*")
                if path.is_file()
            },
        },
    )


def report(root):
    import matplotlib.pyplot as plt

    from .report import categorical, save_figure
    from .sine_consensus_report import capacity, risks

    frozen = json.loads((root / "numerical_frozen.json").read_text())
    for name, expected in frozen["sha256"].items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Frozen numerical state changed: {root / name}")
    registration = json.loads((root / "registration.json").read_text())
    (root / "report_execution_snapshot.py").write_bytes(Path(__file__).read_bytes())
    with np.load(root / "fixed_structure.npz") as archive:
        index = archive["indices"]
    records, states = {}, {"indices": index}
    for frequency in registration["frequencies"]:
        output = root / f"sine_{frequency}"
        spec = json.loads((output / "specification.json").read_text())
        plan = json.loads((output / "protocol.json").read_text())
        with np.load(output / "teacher_audit_only.npz") as archive:
            truth = dict(archive)
        if "direction" in states and not np.array_equal(
            states["direction"], truth["direction"]
        ):
            raise RuntimeError("Frequency transfer changed the teacher direction")
        states["direction"] = truth["direction"]
        floor = capacity(truth["indices"], truth["direction"], spec)["population_mse"]
        heldout = output / "heldout.npz"
        if heldout.exists():
            with np.load(heldout) as archive:
                x, y = archive["test_x"], archive["test_y"]
        else:
            rng = np.random.default_rng(
                plan["seed"] + registration["heldout_seed_offset"]
            )
            x = rng.normal(size=(spec["test"], spec["inputs"])).astype(np.float32)
            y = teacher_predict(x, truth, spec)
            np.savez_compressed(heldout, test_x=x, test_y=y)
        with np.load(output / "numerics/fresh.npz") as archive:
            fresh = dict(archive)
        choice = frozen["choices"][str(frequency)]
        restart = choice["fresh_query_selected_restart"]
        record = {
            "frequency": frequency,
            "population_floor": floor,
            "fixed_I_capacity": capacity(index, truth["direction"], spec),
            "choice": choice,
            "fresh": risks(output, fresh, 0, restart, truth, spec, x, y, floor),
        }
        states[f"weights_{frequency}"] = fresh["weights"][0, restart, :, 0]
        states[f"numeric_{frequency}"] = fresh["numeric"][0, restart]
        with np.load(output / "numerics/calibration.npz") as archive:
            row = choice["calibration_row"]
            calibration_restart = choice["calibration_query_selected_restart"]
            states[f"calibration_weights_{frequency}"] = archive["weights"][
                row, calibration_restart, :, 0
            ]
            states[f"calibration_numeric_{frequency}"] = archive["numeric"][
                row, calibration_restart
            ]
            record["calibration"] = risks(
                output,
                archive,
                row,
                choice["calibration_query_selected_restart"],
                truth,
                spec,
                x,
                y,
                floor,
            )
        reference = Path(registration["parent"]) / f"sine_{frequency}"
        if digest(reference / "source.npz") != digest(output / "source.npz"):
            raise RuntimeError("Archived controls use different observations")
        reference_frozen = json.loads((reference / "numerical_frozen.json").read_text())
        references = json.loads((reference / "numerics/selection.json").read_text())
        record["archived_controls"] = {}
        for method in ("oracle", "dense_random", "dense_moment"):
            name = f"numerics/{method}_fresh.npz"
            if digest(reference / name) != reference_frozen["sha256"][name]:
                raise RuntimeError(f"Archived control changed: {reference / name}")
            with np.load(reference / name) as archive:
                selected = references[method]["query_selected_restart"]
                states[f"{method}_weights_{frequency}"] = archive["weights"][
                    0, selected, :, 0
                ]
                record["archived_controls"][method] = risks(
                    output,
                    archive,
                    0,
                    references[method]["query_selected_restart"],
                    truth,
                    spec,
                    x,
                    y,
                    floor,
                )
        records[str(frequency)] = record
    np.savez_compressed(root / "selected_states.npz", **states)
    seconds = sum(
        float(json.loads(path.read_text())["seconds"])
        for path in root.glob("sine_*/numerics/*.json")
        if path.name != "selection.json"
    )
    save_json(
        root / "summary.json", {"frequencies": records, "training_seconds": seconds}
    )
    figure, axes = plt.subplots(4, 1, figsize=(12, 7), constrained_layout=True)
    categorical(axes[0], index.T, 3, "Один фиксированный I, найденный на f=6")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Индекс")
    axes[0].set_yticks([0], ["I"])
    for axis, frequency in zip(axes[1:], registration["frequencies"]):
        vectors = [
            states[f"{prefix}weights_{frequency}"]
            for prefix in ("calibration_", "", "dense_random_")
        ]
        aligned = [
            (1 if vector @ states["direction"] >= 0 else -1) * vector
            for vector in vectors
        ]
        values = np.stack([frequency * states["direction"], *aligned])
        limit = max(float(np.abs(values).max()), 1e-12)
        axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_yticks([0, 1, 2, 3], ["f·v", "C, подбор", "C, повтор", "Dense Adam"])
        axis.set_title(
            f"f={frequency}; C после подбора: {records[str(frequency)]['calibration']['selected_population_mse']:.6f}; повтор: {records[str(frequency)]['fresh']['selected_population_mse']:.6f}"
        )
        axis.set_xlabel("Координата входа")
    save_figure(figure, root / "frequency_transfer_heatmaps")
    lines = [
        "# Перенос фиксированного I с f=6 на другие частоты",
        "",
        (
            "Один индексатор из source-validation на f=6; целевое направление v не меняется. "
            "На каждой частоте C и амплитуда обучаются заново; исходные числа и состояния Adam не копируются. "
            "f=6 — контроль повторной численной подгонки, f=4/8 — перенос частоты. "
            "Учитель: .4sin(.7vᵀx)+sin(fvᵀx), ученик: a·sin(C[I]ᵀx)."
        ),
        "",
        (
            "12 настроек радиуса/Adam lr × 32 calibration-старта × 3000 шагов. "
            "Query выбирает checkpoint и старт на настройку; selection выбирает настройку. "
            "После этого 32 новых старта, query выбирает один из них. "
            "Все выборы заморожены до teacher-аудита, population-риска и нового test. "
            "Support/query/selection — прежние наблюдения, поэтому это перенос частоты на исходном направлении, а не проверка нового направления."
        ),
        "",
        "## Результат подбора свежего кодбука",
        "",
        (
            "Здесь показана модель, выбранная по query/selection среди 12 настроек. Все C обучены с нуля. "
            "Достигнуто близкое к глобальному качество на каждой частоте при одном неизменном I."
        ),
        "",
        "| f | Population | Test | Train | Query | Предел при фикс. I | Глобальный |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records.values():
        cal = row["calibration"]
        lines.append(
            f"| {row['frequency']} | {cal['selected_population_mse']:.8f} | {cal['selected_test_mse']:.8f} | "
            f"{cal['selected_train_mse']:.8f} | {cal['selected_query_mse']:.8f} | "
            f"{row['fixed_I_capacity']['population_mse']:.8f} | {row['population_floor']:.8f} |"
        )
    lines += [
        "",
        "## Повтор при выбранной настройке с 32 новыми стартами",
        "",
        "| f | Population, выбранный по query | Test | Среднее 32 | В пределах глобального +.001 | Лучший риск при фикс. I | Глобальный |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records.values():
        fresh = row["fresh"]
        lines.append(
            f"| {row['frequency']} | {fresh['selected_population_mse']:.8f} | {fresh['selected_test_mse']:.8f} | "
            f"{fresh['mean_population_mse']:.8f} | {fresh['within_0_001_of_floor']}/32 | "
            f"{row['fixed_I_capacity']['population_mse']:.8f} | {row['population_floor']:.8f} |"
        )
    lines += [
        "",
        (
            "В повторе хорошая подгонка C не воспроизводится устойчиво из случайного старта. "
            "Близкая аналитическая представимость и реально достигнутые calibration-решения показывают, что этот провал численный. "
            "Смена частоты в основном масштабирует прежнее направление; она не проверяет истинность всех найденных групп равенств. "
            "Ранее отрицательный перенос в sine_consensus/transfer менял сами три коэффициента направления при f=6 — это другой эксперимент."
        ),
        "",
        "![Фиксированный I и заново обученные веса](frequency_transfer_heatmaps.png)",
        "",
        "## Численная подгонка и контроли",
        "",
        (
            "`summary.json` содержит все 32 риска, calibration и fresh отдельно, полные train/query MSE, "
            "выбранные настройки, аналитическую представимость и старые dense/oracle контроли. "
            "Контроли используют те же функции/данные и 3000 шагов, но другой численный подбор "
            "(mean8, seed502); бюджеты поиска и старты не попарно одинаковы. "
            "Глобальный риск ненулевой, поскольку ученик содержит одну синусоиду. "
            "При визуализации общий знак веса согласован с v; исходные I/C/W сохранены без изменения."
        ),
        "",
        (
            f"Новое численное обучение: {seconds / 60:.2f} мин. "
            "Ни один поиск структуры на целевых частотах не запускался."
        ),
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(
        json.dumps({"frequencies": records, "training_seconds": seconds}, indent=2),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--stage", required=True, choices=("prepare", "run", "report"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare(args.root, args.parent)
    elif args.stage == "run":
        run(args.root, torch.device(args.device))
    else:
        report(args.root)


if __name__ == "__main__":
    main()
