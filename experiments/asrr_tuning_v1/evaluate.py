"""Freeze primary/CE policies and score two prespecified fresh cohorts."""

import argparse
import json

import numpy as np
from scipy.stats import t

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .data import SETTINGS, episodes
from .replicate import OUTPUT, ROOT


def freeze(settings):
    path = OUTPUT / "choices_frozen.json"
    if path.exists():
        raise RuntimeError("Tuned choices already frozen")
    jobs = json.loads((OUTPUT / "d2_jobs.json").read_text())
    for number in range(len(jobs)):
        receipt = OUTPUT / f"d2_jobs/{number:04d}.json"
        if (
            not receipt.exists()
            or json.loads(receipt.read_text())["exit_code"]
        ):
            raise RuntimeError(f"D2 is incomplete: {receipt}")
    choices = {}
    for setting in settings:
        dataset = SETTINGS[setting][0]
        models = {
            r["name"]: r
            for r in json.loads(
                (ROOT / "banks" / dataset / "models.json").read_text()
            )
        }
        options = []
        for p in (ROOT / "study" / setting / "fits/d2").glob("*.json"):
            fit = json.loads(p.read_text())
            model = models[fit["job"]["name"]]
            families = {
                "asrr" if family.startswith("asrr") else family
                for family in [
                    model["family"],
                    *model.get("family_aliases", []),
                ]
            }
            for key, metric in fit["metrics"].items():
                for family in families:
                    options.append(
                        {
                            "name": model["name"],
                            "family": family,
                            "codes": model["codes"],
                            "key": key,
                            "rule": fit["rules"][key],
                            "metrics": metric,
                            "fit_metadata_sha256": digest(p),
                            "model_sha256": model.get("model_sha256"),
                        }
                    )
        policies = {}
        for criterion in ("accuracy", "ce"):
            for family in ("asrr", "g34", "dense"):
                candidates = [
                    row for row in options if row["family"] == family
                ]
                ordering = (
                    (
                        lambda r: (
                            -r["metrics"]["accuracy"],
                            r["metrics"]["ce"],
                            r["codes"],
                        )
                    )
                    if criterion == "accuracy"
                    else (lambda r: (r["metrics"]["ce"], r["codes"]))
                )
                policies[f"{family}_{criterion}"] = min(
                    candidates, key=ordering
                )
        choices[setting] = policies
    save_json(
        path,
        {
            "policies": choices,
            "query_answers_generated": False,
            "primary_criterion": "Development accuracy; CE and fewer codes break ties",
        },
    )
    print(
        json.dumps(
            {
                setting: {
                    policy: {
                        "name": row["name"],
                        "rule": row["rule"],
                        "metrics": row["metrics"],
                    }
                    for policy, row in policies.items()
                }
                for setting, policies in choices.items()
            }
        ),
        flush=True,
    )


def prepare_targets():
    frozen = json.loads((OUTPUT / "choices_frozen.json").read_text())
    jobs = []
    panel_hashes = {}
    answer_hashes = {}
    for setting, policies in frozen["policies"].items():
        root = ROOT / "study" / setting
        (root / "private").mkdir()
        for cohort in ("test_a", "test_b"):
            seed = (
                7611174000
                + 100 * list(SETTINGS).index(setting)
                + (1 if cohort == "test_a" else 2)
            )
            panel, provenance = episodes(
                setting, "test", 90 if setting == "har3" else 512, seed
            )
            np.savez_compressed(
                root / f"private/{cohort}_answers.npz", y=panel.pop("query_y")
            )
            np.savez_compressed(root / f"panels/{cohort}.npz", **panel)
            save_json(root / f"private/{cohort}_provenance.json", provenance)
            panel_hashes[f"{setting}/{cohort}"] = digest(
                root / f"panels/{cohort}.npz"
            )
            answer_hashes[f"{setting}/{cohort}"] = digest(
                root / f"private/{cohort}_answers.npz"
            )
            unique = {}
            mapping = {}
            for policy, row in policies.items():
                rule = row["rule"]
                if rule["variant"] == "support":
                    fingerprint = json.dumps(
                        {"name": row["name"], "rule": rule}, sort_keys=True
                    )
                    variant = f"support_{len(unique)}"
                    prediction_key = "frozen"
                else:
                    fingerprint = row["name"] + "__" + rule["variant"]
                    variant = rule["variant"]
                    prediction_key = row["key"]
                if fingerprint not in unique:
                    job = {
                        "setting": setting,
                        "name": row["name"],
                        "phase": cohort,
                        "variant": variant,
                        "rule": rule,
                        "model_sha256": row.get("model_sha256"),
                        "panel_sha256": panel_hashes[f"{setting}/{cohort}"],
                    }
                    unique[fingerprint] = job
                    jobs.append(job)
                job = unique[fingerprint]
                mapping[policy] = {
                    "file": job["name"] + "__" + job["variant"] + ".npz",
                    "key": prediction_key,
                    "choice": row,
                }
            save_json(OUTPUT / f"{setting}_{cohort}_mapping.json", mapping)
    save_json(OUTPUT / "target_jobs.json", jobs)
    save_json(
        OUTPUT / "target_registration.json",
        {
            "choices_sha256": digest(OUTPUT / "choices_frozen.json"),
            "jobs_sha256": digest(OUTPUT / "target_jobs.json"),
            "cohorts": ["test_a", "test_b"],
            "retuning_between_cohorts": False,
            "panel_sha256": panel_hashes,
            "answer_sha256": answer_hashes,
        },
    )


def half(values):
    return float(
        t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )


def score():
    if (OUTPUT / "tuned_results.json").exists():
        raise RuntimeError("Tuned queries already scored")
    registration = json.loads(
        (OUTPUT / "target_registration.json").read_text()
    )
    if (
        digest(OUTPUT / "choices_frozen.json")
        != registration["choices_sha256"]
        or digest(OUTPUT / "target_jobs.json") != registration["jobs_sha256"]
    ):
        raise RuntimeError("Target registration changed")
    choices = json.loads((OUTPUT / "choices_frozen.json").read_text())[
        "policies"
    ]
    frozen = {}
    for setting in choices:
        for cohort in ("test_a", "test_b"):
            mapping = json.loads(
                (OUTPUT / f"{setting}_{cohort}_mapping.json").read_text()
            )
            for spec in mapping.values():
                path = (
                    ROOT / "study" / setting / "fits" / cohort / spec["file"]
                )
                receipt = json.loads(path.with_suffix(".json").read_text())
                if receipt.get("access_violations"):
                    raise RuntimeError(f"Held-out violation {path}")
                if digest(path) != receipt["prediction_sha256"]:
                    raise RuntimeError(f"Changed prediction {path}")
                frozen[str(path)] = digest(path)
    save_json(
        OUTPUT / "tuned_predictions_frozen.json",
        {"predictions": frozen, "query_answers_opened": False},
    )
    rows, paired, arrays = [], [], {}
    for setting, policies in choices.items():
        outcomes = {}
        for cohort in ("test_a", "test_b"):
            root = ROOT / "study" / setting
            answer_path = root / f"private/{cohort}_answers.npz"
            if (
                digest(answer_path)
                != registration["answer_sha256"][f"{setting}/{cohort}"]
            ):
                raise RuntimeError("Changed registered query answers")
            truth = np.load(answer_path)["y"]
            mapping = json.loads(
                (OUTPUT / f"{setting}_{cohort}_mapping.json").read_text()
            )
            arrays[f"{setting}_{cohort}_truth"] = truth
            outcomes[cohort] = {}
            for policy, spec in mapping.items():
                probability = np.load(root / "fits" / cohort / spec["file"])[
                    spec["key"]
                ]
                if not np.isfinite(probability).all():
                    raise RuntimeError("Nonfinite target predictions")
                accuracy = (probability.argmax(-1) == truth.argmax(-1)).mean(
                    -1
                )
                ce = (
                    -(np.log(np.maximum(probability, 1e-30)) * truth)
                    .sum(-1)
                    .mean(-1)
                )
                outcomes[cohort][policy] = (accuracy, ce)
                unit = (
                    accuracy.reshape(10, 9).mean(0)
                    if setting == "har3"
                    else accuracy
                )
                rows.append(
                    {
                        "setting": setting,
                        "cohort": cohort,
                        "policy": policy,
                        "name": spec["choice"]["name"],
                        "codes": spec["choice"]["codes"],
                        "rule": spec["choice"]["rule"],
                        "accuracy": float(accuracy.mean()),
                        "accuracy_half_ci": half(unit),
                        "ce": float(ce.mean()),
                        "episodes": len(accuracy),
                    }
                )
                arrays[f"{setting}_{cohort}_{policy}_probability"] = (
                    probability
                )
            for criterion in ("accuracy", "ce"):
                left = outcomes[cohort][f"asrr_{criterion}"][0]
                for other in ("g34", "dense"):
                    delta = left - outcomes[cohort][f"{other}_{criterion}"][0]
                    if setting == "har3":
                        delta = delta.reshape(10, 9).mean(0)
                    paired.append(
                        {
                            "setting": setting,
                            "cohort": cohort,
                            "criterion": criterion,
                            "asrr_minus": other,
                            "difference": float(delta.mean()),
                            "half_ci": half(delta),
                        }
                    )
        for policy, choice in policies.items():
            accuracy = np.concatenate(
                [
                    outcomes[cohort][policy][0]
                    for cohort in ("test_a", "test_b")
                ]
            )
            ce = np.concatenate(
                [
                    outcomes[cohort][policy][1]
                    for cohort in ("test_a", "test_b")
                ]
            )
            unit = (
                accuracy.reshape(20, 9).mean(0)
                if setting == "har3"
                else accuracy
            )
            rows.append(
                {
                    "setting": setting,
                    "cohort": "pooled_fixed",
                    "policy": policy,
                    "name": choice["name"],
                    "codes": choice["codes"],
                    "rule": choice["rule"],
                    "accuracy": float(accuracy.mean()),
                    "accuracy_half_ci": half(unit),
                    "ce": float(ce.mean()),
                    "episodes": len(accuracy),
                }
            )
        for criterion in ("accuracy", "ce"):
            for other in ("g34", "dense"):
                delta = np.concatenate(
                    [
                        outcomes[cohort][f"asrr_{criterion}"][0]
                        - outcomes[cohort][f"{other}_{criterion}"][0]
                        for cohort in ("test_a", "test_b")
                    ]
                )
                if setting == "har3":
                    delta = delta.reshape(20, 9).mean(0)
                paired.append(
                    {
                        "setting": setting,
                        "cohort": "pooled_fixed",
                        "criterion": criterion,
                        "asrr_minus": other,
                        "difference": float(delta.mean()),
                        "half_ci": half(delta),
                    }
                )
    np.savez_compressed(OUTPUT / "tuned_scoring.npz", **arrays)
    save_json(
        OUTPUT / "tuned_results.json",
        {
            "rows": rows,
            "paired": paired,
            "scope": "Prespecified fresh episodes; fixed previously studied source/classes/subjects; no test-based tuning",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("freeze", "prepare-targets", "score")
    )
    parser.add_argument("--settings", nargs="+")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.settings)
    elif args.phase == "prepare-targets":
        prepare_targets()
    else:
        score()
