"""Open new query responses only after the complete numerical freeze."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.reporting.evaluate import (
    align_index,
    write_csv,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.theorem_pair_recovery_v1.recover import canonical_partition

LABELS = {
    "current": "G34",
    "signature": "Weighted SVD: geometric first",
    "dense": "Dense Adam + GN",
    "signature_random": "Shuffled signature",
    "dense_adam": "Dense Adam",
}


def score(output):
    freeze = json.loads((output / "prediction_freeze.json").read_text())
    if digest(output / "plan.json") != freeze["plan_sha256"]:
        raise RuntimeError("The frozen target plan changed")
    if digest(output / "inputs_frozen.json") != freeze["inputs_frozen_sha256"]:
        raise RuntimeError("The input commitment changed")
    plan = json.loads((output / "plan.json").read_text())
    root = Path(plan["root"])
    commitments = json.loads((output / "inputs_frozen.json").read_text())
    for path, expected in commitments["public_sha256"].items():
        if digest(root / path) != expected:
            raise RuntimeError(f"Public input changed: {path}")
    predictions, receipts = {}, {}
    compact = output / "predictions"
    compact.mkdir()
    for job_id, item in freeze["predictions"].items():
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise RuntimeError(f"Prediction changed: {job_id}")
        receipt_path = path.with_name("access_receipt.json")
        if digest(receipt_path) != item["receipt_sha256"]:
            raise RuntimeError(f"Access receipt changed: {job_id}")
        receipts[job_id] = json.loads(receipt_path.read_text())
        with np.load(path) as stored:
            predictions[job_id] = dict(stored)
        np.savez_compressed(
            compact / f"{job_id}.npz",
            **{
                key: value
                for key, value in predictions[job_id].items()
                if "prediction_" in key
            },
        )
    save_json(output / "access_receipts.json", receipts)
    save_json(
        output / "query_opening.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "prediction_freeze_sha256": digest(output / "prediction_freeze.json"),
            "all_predictions_verified_before_opening": True,
        },
    )
    private = {}
    for name, expected in commitments["private_sha256"].items():
        path = root / "private" / name
        if digest(path) != expected:
            raise RuntimeError(f"Sealed query input changed: {name}")
        with np.load(path) as stored:
            private[Path(name).stem] = dict(stored)
    np.savez_compressed(
        output / "scoring_answers.npz",
        **{f"{case}_query_y": arrays["query_y"] for case, arrays in private.items()},
    )
    transfer, recovery, structure_arrays = [], [], {}
    for assignment in plan["assignments"]:
        case, condition, method = [
            assignment[key] for key in ("case", "condition", "method")
        ]
        folder = root / "targets" / f"{case}_c{condition}"
        truth = private[case]["index"]
        structure_arrays[f"{case}_truth"] = truth
        if method != "dense":
            with np.load(folder / "structures.npz") as stored:
                index = stored[method]
            aligned, metrics = align_index(index, truth, 4)
            structure_arrays[f"{case}_c{condition}_{method}"] = aligned
            recovery.append(
                {
                    **{
                        key: assignment[key]
                        for key in (
                            "case",
                            "condition",
                            "geometry",
                            "stratum",
                            "method",
                        )
                    },
                    **metrics,
                    "unsigned_exact": canonical_partition(np.abs(index), 4).tobytes()
                    == canonical_partition(np.abs(truth), 4).tobytes(),
                }
            )
        result = predictions[assignment["job"]]
        for budget in (64, 256, 1024):
            fields = [(method, f"prediction_{budget}")]
            if method == "dense":
                fields.append(("dense_adam", f"adam_prediction_{budget}"))
            for label, field in fields:
                errors = (result[field] - private[case]["query_y"]) ** 2
                if not np.isfinite(errors).all():
                    raise RuntimeError(f"Nonfinite predictions: {assignment['job']}")
                for task, mse in enumerate(errors.mean((1, 2))):
                    transfer.append(
                        {
                            **{
                                key: assignment[key]
                                for key in (
                                    "case",
                                    "condition",
                                    "geometry",
                                    "stratum",
                                    "outputs",
                                )
                            },
                            "method": label,
                            "budget": budget,
                            "task": task,
                            "mse": float(mse),
                            "job": assignment["job"],
                            "first_values": 256 if label.startswith("dense") else 4,
                            "task_parameters": (256 if label.startswith("dense") else 4)
                            + 16
                            + 16 * assignment["outputs"]
                            + assignment["outputs"],
                        }
                    )
    write_csv(output / "transfer_per_task.csv", transfer)
    write_csv(output / "recovery.csv", recovery)
    np.savez_compressed(output / "recovery_structures.npz", **structure_arrays)
    source_records = json.loads((output / "structure_freeze.json").read_text())["cases"]
    paired = []
    for geometry in ("local", "quartet_mix", "random"):
        for condition in (0, 1):
            records = [
                r
                for r in source_records
                if r["geometry"] == geometry and r["condition"] == condition
            ]
            available = [r["case"] for r in records if r["signature_available"]]
            record = {
                "geometry": geometry,
                "condition": condition,
                "available_strata": len(available),
                "total_strata": len(records),
            }
            for method in LABELS:
                rows = [
                    r
                    for r in transfer
                    if r["case"] in available
                    and r["condition"] == condition
                    and r["method"] == method
                    and r["budget"] == 1024
                ]
                record[method] = (
                    float(np.mean([r["mse"] for r in rows])) if rows else None
                )
            paired.append(record)
    write_csv(output / "matched_primary.csv", paired)
    summary = {
        "primary_budget": 1024,
        "source_conditions": 12,
        "signature_available": sum(r["signature_available"] for r in source_records),
        "new_target_functions": 48,
        "unique_target_fits": len(plan["jobs"]),
        "method_conditions": len(plan["assignments"]),
        "matched_primary": paired,
        "recovery": {
            method: {
                "exact": sum(r["exact"] for r in recovery if r["method"] == method),
                "mask_exact": sum(
                    r["mask_exact"] for r in recovery if r["method"] == method
                ),
                "available": sum(r["method"] == method for r in recovery),
                "total_conditions": 12,
            }
            for method in ("current", "signature", "signature_random")
        },
        "prediction_freeze_utc": freeze["created_utc"],
        "query_opening_utc": json.loads((output / "query_opening.json").read_text())[
            "created_utc"
        ],
        "access_receipts": len(receipts),
        "violations": sum(len(r["violations"]) for r in receipts.values()),
        "target_fleet_wall_seconds": freeze["wall_seconds"],
        "scope": "Fresh numeric tasks on six previously known source structures; exploratory, no independent source confirmation",
    }
    save_json(output / "summary.json", summary)
    lines = [
        "# Взвешенные сигнатуры: перенос на общей синтетике",
        "",
        "Первый геометрический кандидат взвешенной SVD сравнивается с G34 на тех же сохранённых source-моделях. Структуры зафиксированы до генерации 48 новых численных target-функций. Все их численные параметры обучаются заново. Source-структуры и исторические исходы уже известны; это продолжение на прежних шести структурных семействах, не независимое подтверждение по новым source-семействам.",
        "",
        "## Основной результат: N=1024",
        "",
        "Средняя query MSE по восьми новым функциям на страту. В таблице все методы оценены на одинаковых доступных стратах. Шум относится только к исходным задачам; target-ответы чистые. Dense означает Adam + тот же GN. В одном шумном random-условии сигнатурного кандидата нет: строка 1/2 описывает только оставшуюся страту и не заменяет среднее по обеим.",
        "",
        "| Геометрия | Source σ | Страт | G34 | Взвешенная SVD | Dense | Перемешанные сигнатуры |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in paired:
        values = " | ".join(
            f"{record[method]:.6g}"
            for method in ("current", "signature", "dense", "signature_random")
        )
        lines.append(
            f"| {record['geometry']} | {0.2 * record['condition']:.1f} | {record['available_strata']}/{record['total_strata']} | {values} |"
        )
    lines += [
        "",
        "## Каждая страта, включая отсутствующее построение",
        "",
        "Страта 0: один выход и общий генерирующий bias; страта 1: два выхода и независимые bias. В fitted-моделях bias свободные. Эти два различия не изолированы причинно.",
        "",
        "| Case | Source σ | G34 | Сигнатуры | Dense | Точный I G34 / сигнатур |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for record in source_records:
        case, condition = record["case"], record["condition"]
        values, exact = [], []
        for method in ("current", "signature", "dense"):
            rows = [
                r
                for r in transfer
                if r["case"] == case
                and r["condition"] == condition
                and r["method"] == method
                and r["budget"] == 1024
            ]
            values.append(
                f"{np.mean([r['mse'] for r in rows]):.6g}" if rows else "нет кандидата"
            )
            if method != "dense":
                entries = [
                    r
                    for r in recovery
                    if r["case"] == case
                    and r["condition"] == condition
                    and r["method"] == method
                ]
                exact.append(
                    ("да" if entries[0]["exact"] else "нет") if entries else "отказ"
                )
        lines.append(
            f"| {case} ({record['geometry']}) | {0.2 * condition:.1f} | {' | '.join(values)} | {' / '.join(exact)} |"
        )
    lines += ["", "## Полнота и восстановление", ""]
    for method in ("current", "signature"):
        item = summary["recovery"][method]
        lines.append(
            f"- {LABELS[method]}: точный signed I {item['exact']}/12; точная маска {item['mask_exact']}/12; построено {item['available']}/12."
        )
    lines += [
        "",
        "## Границы вывода и воспроизводимость",
        "",
        "Во всех методах одинаковый adapter: 18 Adam-стартов/2000 обновлений и четыре GN-старта/48 обновлений. Поддержка N=64/256/1024, отдельная numerical selection N/4, query2048. Число параметров на задачу: 37/54 для четырёх кодов при одном/двух выходах, 289/306 для Dense. Совпадающие канонические структуры используют одни предсказания.",
        "",
        "Общего среднего raw MSE по разным геометриям не рассчитываем. Две страты не дают надёжной общей оценки вариации исходных семейств; восемь новых численных функций внутри одной структуры не считаются восемью независимыми открытиями структуры. Перемешивание сохраняет плотность и signed code counts, но не отделяет вклады маски, знаков и sharing.",
        "",
        f"Завершено {summary['unique_target_fits']} уникальных target-подгонок; receipts {summary['access_receipts']}, нарушений {summary['violations']}. Freeze: {summary['prediction_freeze_utc']}; первое открытие ответов: {summary['query_opening_utc']}.",
        "",
        "Полные per-task значения и меньшие N — transfer_per_task.csv; matched_primary.csv — основная сопоставимая таблица; recovery.csv — поздний структурный аудит. Точные предсказания и ответы сохранены в predictions/ и scoring_answers.npz. Полные численные состояния и кривые остаются локально по пути из plan.json. Исторические результаты и статья не изменены.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    for name in ("PROTOCOL.md", "README.md"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.output.resolve())


if __name__ == "__main__":
    main()
