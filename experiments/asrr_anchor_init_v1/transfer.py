"""Select aSRR indexators on development tasks and fit fresh target values."""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.asrr_tuning_v1.data import episodes
from experiments.paper_recovery_extension_v1.adapter import fit_adam, values
from experiments.paper_recovery_extension_v1.regression_target import (
    TargetBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from . import OUTPUT, ROOT
from .bank import ORIGINS

SETTINGS = ("mnist16", "mnist100")
PRIOR_CHOICES = Path("artifacts/asrr_tuning_2026-11-17/choices_frozen.json")
COHORTS = ("test_a", "test_b")


def rank(row):
    return -row["accuracy"], row["ce"], row["origin"]


def development_rows():
    plan = json.loads((OUTPUT / "development_jobs.json").read_text())
    rows = []
    for number, job in enumerate(plan["jobs"]):
        path = OUTPUT / f"development_fits/{number:04d}.json"
        record = json.loads(path.read_text())
        if record["job"] != job:
            raise ValueError(f"Changed development job {number}")
        rows.append({**job, **record["metrics"]})
    return rows


def public_panel(setting, cohort, count, seed, *, directory=None, target=None):
    if target is None:
        target = cohort in COHORTS
    panel, provenance = episodes(
        setting, "test" if target else "development", count, seed
    )
    if directory is None:
        directory = ROOT / setting
    directory.mkdir(parents=True, exist_ok=True)
    panel_path = directory / f"{cohort}.npz"
    if target:
        private = directory / "private"
        private.mkdir(exist_ok=True)
        generator = np.random.default_rng(seed + 1000)
        order = np.array(
            [
                generator.permutation(panel["query_y"].shape[1])
                for _ in panel["query_y"]
            ]
        )
        batch = np.arange(count)[:, None]
        answers = panel.pop("query_y")[batch, order]
        panel["query_x"] = panel["query_x"][batch, order]
        answer_path = private / f"{cohort}_answers.npz"
        np.savez_compressed(answer_path, y=answers, query_order=order)
        save_json(private / f"{cohort}_provenance.json", provenance)
    else:
        save_json(directory / f"{cohort}_provenance.json", provenance)
    np.savez_compressed(panel_path, **panel)
    record = {
        "path": str(panel_path),
        "sha256": digest(panel_path),
        "count": count,
    }
    if target:
        record.update(
            answer_path=str(answer_path), answer_sha256=digest(answer_path)
        )
    return record


def prepare_development():
    stage = "development"
    path = OUTPUT / "development_jobs.json"
    if path.exists():
        raise FileExistsError(path)
    metadata = {
        origin: json.loads((OUTPUT / "banks" / f"{origin}.json").read_text())
        for origin in ORIGINS
    }
    jobs, panels = [], {}
    previous = json.loads(PRIOR_CHOICES.read_text())["policies"]
    for number, setting in enumerate(SETTINGS):
        frozen = previous[setting]["asrr_accuracy"]
        name = frozen["name"].removeprefix("asrr_")
        panels[f"{setting}/{stage}"] = public_panel(
            setting, stage, 96, 7611182300 + number
        )
        for bank in metadata.values():
            candidate = next(
                row for row in bank["candidates"] if row["name"] == name
            )
            jobs.append(
                {
                    **candidate,
                    "setting": setting,
                    "cohort": stage,
                    "rule": frozen["rule"],
                }
            )
    save_json(
        path,
        {
            "jobs": jobs,
            "panels": panels,
            "bank_sha256": {
                origin: row["bank_sha256"] for origin, row in metadata.items()
            },
            "protocol_sha256": digest(Path(__file__).with_name("PROTOCOL.md")),
            "previous_accuracy_choices_sha256": digest(PRIOR_CHOICES),
        },
    )
    print(json.dumps({"stage": stage, "jobs": len(jobs)}), flush=True)


def freeze():
    path = OUTPUT / "choices_frozen.json"
    if path.exists():
        raise FileExistsError(path)
    results = development_rows()
    with (OUTPUT / "banks/independent.json").open() as stream:
        reference = f"anchor_{json.load(stream)['alignment_reference']:02d}"
    choices, indexes = {}, {}
    plan_path = OUTPUT / "development_jobs.json"
    plan = json.loads(plan_path.read_text())
    for setting in SETTINGS:
        ranked = sorted(
            (row for row in results if row["setting"] == setting), key=rank
        )
        independent = next(
            row for row in ranked if row["origin"] == "independent"
        )
        warm = next(row for row in ranked if row["origin"] != "independent")
        chosen = {
            "independent": independent,
            "warm": warm,
            "single_anchor": next(
                row for row in ranked if row["origin"] == reference
            ),
        }
        choices[setting] = {}
        for policy, row in chosen.items():
            model = f"{row['origin']}__{row['name']}"
            choices[setting][policy] = {**row, "model": model}
            if model in indexes:
                continue
            bank_path = OUTPUT / "banks" / f"{row['origin']}.npz"
            if digest(bank_path) != plan["bank_sha256"][row["origin"]]:
                raise ValueError(f"Changed bank: {bank_path}")
            with np.load(bank_path) as archive:
                indexes[model] = archive[row["name"]]
    bank_path = OUTPUT / "frozen_indexators.npz"
    np.savez_compressed(bank_path, **indexes)
    save_json(
        path,
        {
            "policies": choices,
            "single_anchor": reference,
            "indexators_sha256": digest(bank_path),
            "development_jobs_sha256": digest(plan_path),
            "selection": "Development accuracy, CE, anchor number",
        },
    )


def prepare_targets():
    path = OUTPUT / "target_jobs.json"
    if path.exists():
        raise FileExistsError(path)
    choices_path = OUTPUT / "choices_frozen.json"
    frozen = json.loads(choices_path.read_text())
    jobs, panels = [], {}
    for setting_number, (setting, policies) in enumerate(
        frozen["policies"].items()
    ):
        for cohort_number, cohort in enumerate(COHORTS):
            panels[f"{setting}/{cohort}"] = public_panel(
                setting,
                cohort,
                384,
                7611183000 + setting_number * 10 + cohort_number,
            )
            unique = {}
            for policy, row in policies.items():
                if row["model"] in unique:
                    unique[row["model"]]["policies"].append(policy)
                else:
                    unique[row["model"]] = {
                        **row,
                        "setting": setting,
                        "cohort": cohort,
                        "policies": [policy],
                    }
            jobs.extend(unique.values())
    save_json(
        path,
        {
            "jobs": jobs,
            "panels": panels,
            "selection_sha256": digest(choices_path),
            "indexators_sha256": frozen["indexators_sha256"],
        },
    )
    print(json.dumps({"stage": "target", "jobs": len(jobs)}), flush=True)


def predict(panel, index, codes, rule, seed):
    x, y, qx = (panel[key] for key in ("train_x", "train_y", "query_x"))
    basis = {
        "index": torch.as_tensor(index, device=x.device, dtype=torch.long),
        "codes": codes,
        "scale": 1.0,
    }
    if rule["variant"] == "support":
        options = {
            "fixed_step": rule["steps"],
            "keep": rule["numeric_starts"],
            "preserve_setting_order": True,
        }
        settings = [
            {**rule["global_setting"], "restart": r}
            for r in range(rule["numeric_starts"])
        ]
    elif rule["variant"] == "wide24_1000":
        options = {"steps": rule["steps"], "keep": rule["ensemble"]}
        settings = [
            {"lr": lr, "quantile": quantile, "restart": r}
            for lr in (0.001, 0.003, 0.01, 0.03)
            for quantile in (0.1, 0.5, 0.9)
            for r in range(2)
        ]
    else:
        raise ValueError(f"Unexpected frozen numerical rule: {rule}")
    params, _ = fit_adam(
        x,
        y,
        panel.get("valid_x"),
        panel.get("valid_y"),
        basis,
        16,
        2,
        loss="cross_entropy",
        seed=seed,
        settings=settings,
        **options,
    )
    with torch.no_grad():
        return (
            values(qx, basis, params[: rule["ensemble"]], 16, 2)
            .softmax(-1)
            .mean(0)
            .cpu()
            .numpy()
        )


def worker(stage, gpu, slot, slots):
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{gpu}")
    plan = json.loads((OUTPUT / f"{stage}_jobs.json").read_text())
    target = stage == "target"
    dest = OUTPUT / f"{stage}_fits"
    dest.mkdir(exist_ok=True)
    boundary = None
    if target:
        boundary = TargetBoundary(ROOT.resolve(), dest.resolve())
        boundary.install()
        bank_path = OUTPUT / "frozen_indexators.npz"
        if digest(bank_path) != plan["indexators_sha256"]:
            raise ValueError("Changed frozen indexators")
        with np.load(bank_path) as archive:
            frozen = dict(archive)
    panels, indexes = {}, {}
    for number in range(slot, len(plan["jobs"]), slots):
        job = plan["jobs"][number]
        result = dest / f"{number:04d}.json"
        if result.exists():
            continue
        started = time.perf_counter()
        setting, cohort = job["setting"], job["cohort"]
        key = f"{setting}/{cohort}"
        if key not in panels:
            spec = plan["panels"][key]
            if digest(spec["path"]) != spec["sha256"]:
                raise ValueError(f"Changed observations: {key}")
            with np.load(spec["path"]) as archive:
                if target and "query_y" in archive.files:
                    raise ValueError("Query answers in a fitting panel")
                panels[key] = {
                    field: torch.as_tensor(archive[field], device=device)
                    for field in archive.files
                }
        if target:
            index = frozen[job["model"]]
        else:
            origin = job["origin"]
            if origin not in indexes:
                bank_path = OUTPUT / "banks" / f"{origin}.npz"
                if digest(bank_path) != plan["bank_sha256"][origin]:
                    raise ValueError(f"Changed bank: {origin}")
                with np.load(bank_path) as archive:
                    indexes[origin] = dict(archive)
            index = indexes[origin][job["name"]]
        seed = 7611182400 + ("development", *COHORTS).index(cohort)
        probability = predict(
            panels[key], index, job["codes"], job["rule"], seed
        )
        if not np.isfinite(probability).all():
            raise ValueError(f"Nonfinite predictions: {job}")
        record = {
            "job": job,
            "gpu": gpu,
            "seconds": time.perf_counter() - started,
        }
        if target:
            prediction_path = result.with_suffix(".npz")
            np.savez_compressed(prediction_path, probability=probability)
            record["prediction_sha256"] = digest(prediction_path)
        else:
            truth = panels[key]["query_y"].cpu().numpy()
            record["metrics"] = {
                "accuracy": float(
                    (probability.argmax(-1) == truth.argmax(-1)).mean()
                ),
                "ce": float(
                    -(np.log(probability.clip(1e-30)) * truth).sum(-1).mean()
                ),
            }
        save_json(result, record)
        if target or stage == "development":
            print(
                json.dumps({"stage": stage, "gpu": gpu, "job": number}),
                flush=True,
            )
    if boundary is not None:
        save_json(
            dest / f"access_{slot}.json",
            {
                "violations": boundary.violations,
                "opened": sorted(boundary.opened),
            },
        )


def run(stage, gpus):
    def partition(slot):
        log = OUTPUT / f"{stage}_gpu{gpus[slot]}.log"
        with log.open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    __spec__.name,
                    "worker",
                    stage,
                    "--gpu",
                    str(gpus[slot]),
                    "--slot",
                    str(slot),
                    "--slots",
                    str(len(gpus)),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(partition, range(len(gpus))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "freeze", "run", "worker")
    )
    parser.add_argument("stage", choices=("development", "target"), nargs="?")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--slots", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.stage == "target":
            prepare_targets()
        else:
            prepare_development()
    elif args.action == "freeze":
        freeze()
    elif args.action == "run":
        run(args.stage, args.gpus)
    else:
        worker(args.stage, args.gpu, args.slot, args.slots)
