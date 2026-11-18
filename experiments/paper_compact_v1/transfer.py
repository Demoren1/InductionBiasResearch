"""Fixed-budget comparison on shared Omniglot episodes and HAR subjects."""

import argparse
import shutil
from pathlib import Path

import numpy as np

from experiments.asrr_anchor_init_v1.report import interval
from experiments.asrr_anchor_init_v1.transfer import COHORTS, public_panel
from experiments.asrr_tuning_v1.data import SETTINGS
from experiments.asrr_tuning_v1.study import run as run_fits
from experiments.paper_structure_benchmark_v1.fitting.real_bank_fit import (
    candidates as native_candidates,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.wsr_omniglot_v1.source import OUTPUT as WSR_OMNIGLOT

from .source import BUDGETS, OMNIGLOT, OUTPUT, ROOT, TUNING, read_json


METHODS = ("dense", "g34", "asrr", "lutq", "yeh_map", "yeh_soft", "msr", "wsr")
NATIVE = METHODS[3:7]
VARIANT = "wide24_500"
TOP4 = f"{VARIANT}_top4"
OLD_HAR = OUTPUT.parent / "paper_recovery_extension_2026-11-14/har"


def save_model(folder, name, codes, *, index=None, basis=None, scale=1.0):
    path = folder / f"{name}.npz"
    if index is None:
        if basis.ndim != 3 or basis.shape[-1] != codes:
            raise ValueError(f"Wrong basis dimension: {name}")
        np.savez_compressed(path, basis=basis.astype(np.float32))
    else:
        if np.abs(index).max() > codes or not np.any(index):
            raise ValueError(f"Wrong signed assignment: {name}")
        np.savez_compressed(
            path, index=index.astype(np.int16), codes=codes, scale=scale
        )
    return {"name": name, "codes": codes, "model_sha256": digest(path)}


def import_model(path, folder, name, codes):
    with np.load(path) as model:
        if "index" in model and model["index"].size:
            scale = float(model["scale"]) if "scale" in model else 1.0
            return save_model(
                folder, name, codes, index=model["index"], scale=scale
            )
        return save_model(folder, name, codes, basis=model["basis"])


def prepare_development(setting, *, yeh=False):
    out = OUTPUT / setting
    out.mkdir(exist_ok=True)
    queue = "development_yeh" if yeh else "development"
    if (out / f"{queue}_jobs.json").exists():
        raise FileExistsError(out / f"{queue}_jobs.json")
    folder = ROOT / "study" / setting / "panels"
    folder.mkdir(parents=True, exist_ok=True)
    dataset, _, _, _ = SETTINGS[setting]
    models = ROOT / "banks" / dataset / "models"
    models.mkdir(parents=True, exist_ok=True)
    tuned = TUNING / "banks" / dataset / "models"
    codes = BUDGETS[setting]
    if setting == "omniglot10":
        phase = "development"
        old = read_json(WSR_OMNIGLOT / "development_panel.json")
        dev_path = Path(old["path"])
        if digest(dev_path) != old["sha256"]:
            raise ValueError("Changed Omniglot development panel")
        banks = {}
        if not yeh:
            banks["g34"] = [
                import_model(
                    tuned / "g34_refined_m1024.npz", models, "g34", codes
                )
            ]
        methods = ("yeh_map", "yeh_soft") if yeh else ("lutq", "msr")
        for method in methods:
            banks[method] = []
            for row, array, _ in native_candidates(OMNIGLOT, method, codes)[
                :8
            ]:
                name = f"{method}_{row}"
                if array.ndim == 2 and np.any(array):
                    scale = np.sqrt(array.size / np.count_nonzero(array))
                    saved = save_model(
                        models, name, codes, index=array, scale=scale
                    )
                elif array.ndim == 3:
                    saved = save_model(models, name, codes, basis=array)
                else:
                    continue
                banks[method].append(saved)
    else:
        phase = "d2"
        dev_path = TUNING / "study/har3/panels/d2.npz"
        banks = {
            "dense": [{"name": "dense", "codes": 8976, "model_sha256": None}],
            "asrr": [
                import_model(
                    tuned / "asrr_global_m128.npz", models, "asrr", codes
                )
            ],
            "g34": [
                import_model(tuned / "g34_m128.npz", models, "g34", codes)
            ],
        }
        for method in NATIVE:
            path = OLD_HAR / "development" / method / "budgets/k128/model.npz"
            banks[method] = [import_model(path, models, method, codes)]
        banks["wsr"] = [
            import_model(path, models, path.stem, codes)
            for path in sorted((ROOT / setting / "wsr").glob("wsr_*.npz"))
        ]
    if any(not rows for rows in banks.values()):
        raise ValueError(f"Empty candidate family in {setting}")
    panel = folder / f"{phase}.npz"
    if panel.exists():
        if digest(panel) != digest(dev_path):
            raise ValueError("Changed development panel")
    else:
        shutil.copyfile(dev_path, panel)
    jobs = [
        {
            "setting": setting,
            "name": row["name"],
            "phase": phase,
            "variant": VARIANT if setting == "omniglot10" else "screen",
            "model_sha256": row["model_sha256"],
            "panel_sha256": digest(panel),
        }
        for rows in banks.values()
        for row in rows
    ]
    save_json(out / ("banks_yeh.json" if yeh else "banks.json"), banks)
    save_json(out / f"{queue}_jobs.json", jobs)


def freeze(setting, *, early=False):
    out = OUTPUT / setting
    suffix = "_early" if early else ""
    choice_file = out / f"choices_frozen{suffix}.json"
    if choice_file.exists():
        raise FileExistsError(choice_file)
    study = ROOT / "study" / setting
    jobs = read_json(out / "development_jobs.json")
    banks = read_json(out / "banks.json")
    if setting == "omniglot10" and not early:
        jobs += read_json(out / "development_yeh_jobs.json")
        banks.update(read_json(out / "banks_yeh.json"))
    fits = {}
    for job in jobs:
        path = (
            study
            / "fits"
            / job["phase"]
            / f"{job['name']}__{job['variant']}.json"
        )
        record = read_json(path)
        if record["job"] != job:
            raise ValueError(f"Changed development job: {path}")
        fits[job["name"]] = record
    policies = {}
    for method, rows in banks.items():
        options = [
            {
                **row,
                **metric,
                "rule": fit["rules"][key],
                "development_key": key,
            }
            for row in rows
            for fit in (fits[row["name"]],)
            for key, metric in fit["metrics"].items()
            if setting != "omniglot10" or key == TOP4
        ]
        policies[method] = min(
            options, key=lambda o: (-o["accuracy"], o["ce"], o["name"])
        )
    if setting == "omniglot10":
        old = read_json(WSR_OMNIGLOT / "choices_frozen.json")["policies"]
        for method, row in old.items():
            policies[method] = {
                **row,
                "codes": 12544 if method == "dense" else BUDGETS[setting],
                "rule": read_json(WSR_OMNIGLOT / "choices_frozen.json")[
                    "numeric_rule"
                ],
                "prior_fit": True,
            }
    expected = (
        set(METHODS) - {"yeh_map", "yeh_soft"} if early else set(METHODS)
    )
    if set(policies) != expected:
        raise ValueError(f"Incomplete frozen method list: {setting}")
    save_json(choice_file, policies)
    print(
        {
            method: (row["name"], row["accuracy"])
            for method, row in policies.items()
        }
    )


def prepare_targets(setting, *, early=False):
    out = OUTPUT / setting
    study = ROOT / "study" / setting
    directory = study / "panels"
    suffix = "_early" if early else ""
    choice_file = out / f"choices_frozen{suffix}.json"
    policies = read_json(choice_file)
    jobs, panels, evaluations = [], {}, {}
    queue_file = out / f"target{suffix}_jobs.json"
    if queue_file.exists():
        raise FileExistsError(queue_file)
    old_panels = read_json(WSR_OMNIGLOT / "target_panels.json")["panels"]
    prior_root = ROOT.parent / WSR_OMNIGLOT.name / "study" / setting
    for count, cohort in enumerate(COHORTS):
        if setting == "omniglot10":
            panel = old_panels[cohort]
            if digest(Path(panel["path"])) != panel["sha256"]:
                raise ValueError("Changed frozen Omniglot observations")
            shutil.copyfile(panel["path"], directory / f"{cohort}.npz")
        else:
            panel = public_panel(
                setting,
                cohort,
                90,
                7611186600 + count,
                directory=directory,
                target=True,
            )
        panels[cohort] = panel
        evaluations[cohort] = {}
        for method, row in policies.items():
            variant = VARIANT if setting == "omniglot10" else "support"
            fit_root = prior_root if row.get("prior_fit") else study
            job = {
                "setting": setting,
                "name": row["name"],
                "variant": variant,
                "phase": cohort,
                "model_sha256": row["model_sha256"],
                "panel_sha256": panel["sha256"],
            }
            if setting == "har3":
                job["rule"] = row["rule"]
            path = fit_root / "fits" / cohort / f"{row['name']}__{variant}.npz"
            if not row.get("prior_fit"):
                if path.exists():
                    if read_json(path.with_suffix(".json"))["job"] != job:
                        raise ValueError(
                            f"Changed previously frozen target fit: {path}"
                        )
                else:
                    jobs.append(job)
            evaluations[cohort][method] = {
                "path": str(path),
                "job": job,
                "key": TOP4 if setting == "omniglot10" else "frozen",
            }
    save_json(
        out / f"target_panels{suffix}.json",
        {
            "panels": panels,
            "evaluations": evaluations,
            "choices_sha256": digest(choice_file),
        },
    )
    save_json(queue_file, jobs)


def score(setting):
    out = OUTPUT / setting
    target = read_json(out / "target_panels.json")
    choice_file = out / "choices_frozen.json"
    if digest(choice_file) != target["choices_sha256"]:
        raise ValueError("Changed frozen choices")
    frozen = {}
    for cohort, methods in target["evaluations"].items():
        frozen[cohort] = {}
        for method, row in methods.items():
            path = Path(row["path"])
            record = read_json(path.with_suffix(".json"))
            if (
                record["job"] != row["job"]
                or record["access_violations"]
                or record["query_answers_used_for_target_fitting"]
                or digest(path) != record["prediction_sha256"]
            ):
                raise ValueError(f"Unverified target predictions: {path}")
            frozen[cohort][method] = record["prediction_sha256"]
    save_json(out / "predictions_frozen.json", frozen)

    records = {m: [] for m in METHODS}
    subjects = []
    for cohort, panel in target["panels"].items():
        path = Path(panel["answer_path"])
        if digest(path) != panel["answer_sha256"]:
            raise ValueError(f"Changed scoring labels: {path}")
        with np.load(path) as data:
            truth = data["y"]
        if setting == "har3":
            subjects.extend(
                row["person"]
                for row in read_json(
                    path.with_name(f"{cohort}_provenance.json")
                )
            )
        for method, row in target["evaluations"][cohort].items():
            with np.load(row["path"]) as data:
                p = data[row["key"]].astype(np.float64)
            if p.shape != truth.shape or not np.isfinite(p).all():
                raise ValueError(f"Invalid predictions: {row['path']}")
            accuracy = (p.argmax(-1) == truth.argmax(-1)).mean(-1)
            ce = -(np.log(np.maximum(p, 1e-30)) * truth).sum(-1).mean(-1)
            records[method].append(np.column_stack((accuracy, ce)))
    units = {method: np.concatenate(rows) for method, rows in records.items()}
    episodes = len(units["dense"])
    if subjects:
        subjects = np.asarray(subjects)
        units = {
            method: np.stack(
                [
                    x[subjects == person].mean(0)
                    for person in np.unique(subjects)
                ]
            )
            for method, x in units.items()
        }
    rows = {}
    for method, choice in read_json(choice_file).items():
        x = units[method]
        delta = x - units["asrr"]
        rows[method] = {
            "codes": choice["codes"],
            "accuracy": float(x[:, 0].mean()),
            "ce": float(x[:, 1].mean()),
            "accuracy_ci95_halfwidth": interval(x[:, 0]),
            "accuracy_minus_asrr": float(delta[:, 0].mean()),
            "difference_ci95_halfwidth": interval(delta[:, 0]),
        }
    save_json(
        out / "results.json",
        {
            "episodes": episodes,
            "units": "subjects" if len(subjects) else "episodes",
            "methods": rows,
        },
    )
    print(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("setting", choices=("omniglot10", "har3"))
    parser.add_argument(
        "action",
        choices=(
            "prepare-development",
            "development",
            "prepare-yeh",
            "development_yeh",
            "freeze",
            "freeze-early",
            "prepare-targets",
            "prepare-targets-early",
            "target",
            "target_early",
            "score",
        ),
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[7])
    args = parser.parse_args()
    if args.action in (
        "development",
        "development_yeh",
        "target",
        "target_early",
    ):
        run_fits(
            args.action, args.gpus, root=ROOT, output=OUTPUT / args.setting
        )
    elif args.action == "prepare-yeh":
        if args.setting != "omniglot10":
            parser.error("prepare-yeh is for Omniglot")
        prepare_development(args.setting, yeh=True)
    elif args.action in ("freeze-early", "prepare-targets-early"):
        if args.setting != "omniglot10":
            parser.error("Early staging is for Omniglot")
        action = freeze if args.action == "freeze-early" else prepare_targets
        action(args.setting, early=True)
    else:
        {
            "prepare-development": prepare_development,
            "freeze": freeze,
            "prepare-targets": prepare_targets,
            "score": score,
        }[args.action](args.setting)
