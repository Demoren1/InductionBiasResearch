"""Score frozen paired episodes and document the source model lineage."""

import json
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import t

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from . import ANCHORS, DENSE, OUTPUT
from .transfer import COHORTS, SETTINGS, development_rows

LABELS = {"mnist16": "MNIST, 16", "mnist100": "MNIST, 100+50"}


def interval(values):
    return float(
        t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )


def audit_sources():
    with np.load(DENSE) as archive:
        original = archive["parameters"]
    rows = []
    for anchor in ANCHORS:
        name = f"anchor_{anchor:02d}"
        folder = OUTPUT / "source_fits" / name
        record = json.loads((folder / "complete.json").read_text())
        if set(record["adapted_tasks"]) != set(ANCHORS) - {anchor}:
            raise ValueError(f"Incorrect descendant list: {name}")
        initial_path = folder / "initial_hidden.npy"
        if digest(initial_path) != record["hidden_sha256"]:
            raise ValueError(f"Changed initial layer: {name}")
        with np.load(folder / "models.npz") as archive:
            params = archive["parameters"]
        if len(params) != 10 or not np.array_equal(
            params[anchor], original[anchor]
        ):
            raise ValueError(f"Incorrect anchor in source cohort: {name}")
        initial = np.load(initial_path)
        if not np.array_equal(initial, original[anchor, : len(initial)]):
            raise ValueError(f"Initial layer does not match anchor: {name}")
        bank = json.loads((OUTPUT / "banks" / f"{name}.json").read_text())
        if (
            bank["models_per_bank"] != 10
            or bank["neuron_correspondence"] != "inherited"
            or bank["dense_sha256"] != digest(folder / "models.npz")
        ):
            raise ValueError(f"Incorrect bank lineage: {name}")
        rows.append(record)
    return rows


def run():
    result_path = OUTPUT / "results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    choices_path = OUTPUT / "choices_frozen.json"
    frozen = json.loads(choices_path.read_text())
    plan = json.loads((OUTPUT / "target_jobs.json").read_text())
    folder = OUTPUT / "target_fits"
    if digest(choices_path) != plan["selection_sha256"]:
        raise ValueError("Changed frozen selection")
    if digest(OUTPUT / "frozen_indexators.npz") != plan["indexators_sha256"]:
        raise ValueError("Changed frozen indexators")
    checked_sources = audit_sources()
    prediction_hashes = {}
    allowed = {
        *(Path(panel["path"]).resolve() for panel in plan["panels"].values()),
        (OUTPUT / "frozen_indexators.npz").resolve(),
    }
    for number, job in enumerate(plan["jobs"]):
        path = folder / f"{number:04d}.json"
        prediction = path.with_suffix(".npz")
        record = json.loads(path.read_text())
        actual = digest(prediction)
        if record["job"] != job or record["prediction_sha256"] != actual:
            raise ValueError(f"Changed target job or predictions: {number}")
        prediction_hashes[prediction.name] = actual
        allowed.update((path.resolve(), prediction.resolve()))
    access = sorted(folder.glob("access_*.json"))
    if not access:
        raise ValueError("Missing target data access records")
    for path in access:
        receipt = json.loads(path.read_text())
        unauthorized = {
            opened
            for opened in map(Path, receipt["opened"])
            if "artifacts" in opened.parts and opened not in allowed
        }
        if unauthorized or receipt["violations"]:
            raise ValueError(f"Unexpected target fitting access: {path}")
    save_json(OUTPUT / "predictions_frozen.json", prediction_hashes)

    accuracies, losses, arrays = {}, {}, {}
    for key, panel in plan["panels"].items():
        path = panel["answer_path"]
        if digest(path) != panel["answer_sha256"]:
            raise ValueError(f"Changed target answers: {key}")
        with np.load(path) as archive:
            arrays[key.replace("/", "_") + "_query_y"] = archive["y"]
    for number, job in enumerate(plan["jobs"]):
        setting, cohort = job["setting"], job["cohort"]
        key = f"{setting}_{cohort}"
        with np.load(folder / f"{number:04d}.npz") as data:
            probability = data["probability"]
        truth = arrays[f"{key}_query_y"]
        if probability.shape != truth.shape or len(truth) != 384:
            raise ValueError(f"Unexpected prediction dimensions: {key}")
        accuracy = (probability.argmax(-1) == truth.argmax(-1)).mean(-1)
        ce = -(np.log(probability.clip(1e-30)) * truth).sum(-1).mean(-1)
        for policy in job["policies"]:
            accuracies[setting, cohort, policy] = accuracy
            losses[setting, cohort, policy] = ce
            arrays[f"{key}_{policy}_accuracy"] = accuracy
            arrays[f"{key}_{policy}_ce"] = ce
    rows = {}
    for setting in SETTINGS:
        rows[setting] = {}
        for name in (*COHORTS, "pooled"):
            cohorts = COHORTS if name == "pooled" else (name,)
            independent = np.concatenate(
                [
                    accuracies[setting, cohort, "independent"]
                    for cohort in cohorts
                ]
            )
            rows[setting][name] = {}
            for policy in frozen["policies"][setting]:
                accuracy = np.concatenate(
                    [accuracies[setting, cohort, policy] for cohort in cohorts]
                )
                delta = accuracy - independent
                rows[setting][name][policy] = {
                    "accuracy": float(accuracy.mean()),
                    "accuracy_half_ci": interval(accuracy),
                    "ce": float(
                        np.concatenate(
                            [
                                losses[setting, cohort, policy]
                                for cohort in cohorts
                            ]
                        ).mean()
                    ),
                    "delta": float(delta.mean()),
                    "delta_half_ci": interval(delta),
                    "episodes": len(delta),
                }
    np.savez_compressed(OUTPUT / "scoring.npz", **arrays)
    results = {
        "query": rows,
        "development": development_rows(),
        "selection": frozen,
        "source_anchor_runs": checked_sources,
        "target_access_audited": True,
    }
    save_json(result_path, results)
    report(results)


def report(results):
    registration = json.loads((OUTPUT / "registration.json").read_text())
    pairs = registration["source_pairs"]
    line = [
        "# aSRR с общей инициализацией исходных моделей",
        "",
        "18 ноября 2026. MNIST, десять исходных пар классов.",
        "Для каждого якоря строим два индексатора из одних и тех же десяти",
        "моделей: самого якоря и девяти моделей, дообученных из его первого",
        "слоя. В каждом режиме выбираем якорь на 96 development-эпизодах.",
        "Номера нейронов в якорных наборах наследуются от старта. Для",
        "независимых моделей применено обычное сопоставление нейронов.",
        "",
        "Бюджеты, семейства и численные правила фиксированы по прошлому",
        "подбору. В обеих ветках построение использует по одной лучшей",
        "Dense-модели каждой исходной задачи. Прежний предварительный",
        "скрининг не участвует в отборе. Переобучение кодбуков на новых",
        "задачах с нуля. Две когорты по 384 эпизода на режим.",
        "",
        "| Режим | Коды и семейство | Независимые, % | Общий старт, % | Якорь | Разность, п.п. (95% ДИ) |",
        "|---|---|---:|---:|---|---:|",
    ]
    choices = results["selection"]["policies"]
    for setting in SETTINGS:
        pool = results["query"][setting]["pooled"]
        cold, warm = pool["independent"], pool["warm"]
        choice = choices[setting]["warm"]
        anchor = int(choice["origin"].split("_")[-1])
        pair = "–".join(map(str, pairs[anchor]))
        family = choice["name"].split("_")[0]
        line.append(
            f"|{LABELS[setting]}|{choice['codes']} {family}|"
            f"{100 * cold['accuracy']:.2f}|{100 * warm['accuracy']:.2f}|{pair}|"
            f"{100 * warm['delta']:+.2f} ± {100 * warm['delta_half_ci']:.2f}|"
        )
    line += [
        "",
        "Разности парные по эпизодам. Интервалы условны на сохранённые",
        "исходные выборки и разбиение классов. 100+50 обозначает примеры",
        "для подгонки и для численной валидации на класс.",
        "",
        "| Режим | Когорта | Вариант | Accuracy, % | CE | Разность к независимым, п.п. |",
        "|---|---|---|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        for cohort, policies in results["query"][setting].items():
            for policy, row in policies.items():
                line.append(
                    f"|{LABELS[setting]}|{cohort}|{policy}|"
                    f"{100 * row['accuracy']:.2f}|{row['ce']:.4f}|"
                    f"{100 * row['delta']:+.2f} ± {100 * row['delta_half_ci']:.2f}|"
                )
    line += [
        "",
        "Вариант single_anchor использует лучшую исходную модель по H-loss",
        "как якорь ещё до development. Он помогает сравнить один заранее",
        "выбранный старт с отбором из десяти.",
        "",
        "| Стартовая пара | Валидация Dense, % | Косинус соответствующих столбцов | Development 16, % | Development 100+50, % |",
        "|---|---:|---:|---:|---:|",
    ]
    for origin in ("independent", *(f"anchor_{i:02d}" for i in ANCHORS)):
        label = (
            "независимые"
            if origin == "independent"
            else "–".join(map(str, pairs[int(origin.split("_")[-1])]))
        )
        bank = json.loads((OUTPUT / "banks" / f"{origin}.json").read_text())
        score = [
            next(
                row["accuracy"]
                for row in results["development"]
                if row["setting"] == setting and row["origin"] == origin
            )
            for setting in SETTINGS
        ]
        line.append(
            f"|{label}|{100 * np.mean(bank['numerical_validation_accuracy']):.2f}|"
            f"{bank['mean_corresponding_source_cosine']:.3f}|"
            f"{100 * score[0]:.2f}|{100 * score[1]:.2f}|"
        )
    line += [
        "",
        "Косинусы рассчитаны по номерам столбцов после обычного сопоставления",
        "для независимых источников и по унаследованным номерам для якорных.",
        "",
        "В комплекте сохранены выбранные индексаторы, расчёт по эпизодам,",
        "протокол и записи происхождения исходных моделей.",
        "",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(line))
    shutil.copy2(
        Path(__file__).with_name("PROTOCOL.md"), OUTPUT / "PROTOCOL.md"
    )


if __name__ == "__main__":
    run()
