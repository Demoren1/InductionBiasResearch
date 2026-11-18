"""Register, select on development characters, freeze, then predict fresh tasks."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from experiments.heterogeneous_index_v1.model import scalar_seed
from experiments.paper_structure_benchmark_v1.methods.scalable_projection import assign
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .model import HIDDEN, INPUTS, align, dense_descriptor, numerical_fit, project

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "artifacts/local/omniglot_prior_2026-11-15"
OUTPUT = PROJECT / "artifacts/omniglot_prior_2026-11-15"
IMAGES = PROJECT / "artifacts/local/msr_reproduction_2026-11-14/images/omniglot"
SOURCE = (
    PROJECT
    / "artifacts/local/omniglot_single_task_2026-11-14/study/source/dense/models.npz"
)
RETAINED = (
    PROJECT / "artifacts/omniglot_signature_2026-11-12/models/current/model.npz",
    PROJECT / "artifacts/benchmark_2026-11-07/omniglot_pilot/targets/current/model.npz",
)
MSR = PROJECT / "artifacts/benchmark_2026-11-07/omniglot_pilot/targets/msr/model.npz"
PENALTIES = (0.0, 0.25, 1.0, 4.0, 16.0)
DEPTHS = (1, 4)


def stamp(**values):
    return {"created_utc": datetime.now(UTC).isoformat(), **values}


def episodes(split, count, seed):
    data = np.load(IMAGES / f"{split}.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    panels = {
        f"{part}_{key}": []
        for part in ("train", "valid", "query")
        for key in ("x", "y")
    }
    provenance = []
    for _ in range(count):
        chosen = rng.choice(len(data), 5, replace=False)
        rotations = rng.integers(0, 4, 5)
        values = {key: [] for key in panels}
        drawings = []
        for label, (character, rotation) in enumerate(zip(chosen, rotations)):
            order = rng.permutation(20)
            drawings.append(order.tolist())
            images = np.rot90(data[character, order, 0], int(rotation), axes=(-2, -1))
            images = images.reshape(20, -1).astype(np.float32) / 255
            for part, positions in (
                ("train", slice(0, 10)),
                ("valid", slice(10, 15)),
                ("query", slice(15, 20)),
            ):
                selected = images[positions]
                values[f"{part}_x"].append(selected)
                values[f"{part}_y"].append(
                    np.eye(5, dtype=np.float32)[[label] * len(selected)]
                )
        for key, value in values.items():
            panels[key].append(np.concatenate(value))
        provenance.append(
            {
                "classes": chosen.tolist(),
                "rotations": rotations.tolist(),
                "drawings": drawings,
            }
        )
    return {key: np.stack(value) for key, value in panels.items()}, provenance


def source_bank():
    bank = {}
    for name, path in zip(("retained_pruned_m64", "retained_full_m64"), RETAINED):
        with np.load(path) as arrays:
            bank[name] = arrays["index"].astype(np.int16)
    with np.load(SOURCE) as arrays:
        ranking = arrays["numerical_cross_entropy"].min(0).argsort()[:16]
        raw = arrays["parameters"][ranking, : INPUTS * HIDDEN].reshape(
            -1, INPUTS, HIDDEN
        )
        ensembles = arrays["ensemble_parameters"][
            ranking, :, : INPUTS * HIDDEN
        ].reshape(-1, 4, INPUTS, HIDDEN)
    normal = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    matched, errors = [], []
    for reference, ensemble in zip(normal, ensembles):
        deviations = []
        for item in ensemble:
            unit = item / np.maximum(np.linalg.norm(item, axis=0), 1e-12)
            left, right = linear_sum_assignment(-reference.T @ unit)
            deviations.append(unit[:, right[np.argsort(left)]] - reference)
        error = np.maximum(
            np.sqrt(np.mean(np.stack(deviations) ** 2, axis=(0, 1))), 0.0003
        )
        left, right = linear_sum_assignment(-normal[0].T @ reference)
        order = right[np.argsort(left)]
        matched.append(reference[:, order])
        errors.append(error[order])
    matched, errors = np.stack(matched), np.stack(errors)
    for codes in (16, 256):
        for mask_name, mask in (
            ("full", np.ones((INPUTS, HIDDEN), dtype=bool)),
            ("pruned", bank["retained_pruned_m64"] != 0),
        ):
            initial = np.zeros((INPUTS, HIDDEN), dtype=np.int16)
            initial[mask] = scalar_seed(matched[0][mask, None], codes).ravel()
            answer = assign(
                initial,
                mask,
                torch.as_tensor(matched),
                torch.as_tensor(errors),
                codes,
                steps=12,
            )
            if answer is None:
                continue
            index, _, scales, score, iterations = answer
            bank[f"source_{mask_name}_m{codes}"] = (
                index * np.where(scales[0] >= 0, 1, -1)[None]
            ).astype(np.int16)
            print(
                json.dumps(
                    {
                        "stage": "source_proposal",
                        "codes": codes,
                        "mask": mask_name,
                        "score": score,
                        "iterations": iterations,
                    }
                ),
                flush=True,
            )
    return bank


def evaluate(name, panels, *, index=None, basis=None, cohort="development"):
    folder = ROOT / cohort / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "fit.npz"
    if path.exists():
        with np.load(path) as arrays:
            result = dict(arrays)
        result["trace"] = json.loads((folder / "trace.json").read_text())
    else:
        started = time.perf_counter()
        result = numerical_fit(panels, index=index, basis=basis)
        np.savez_compressed(
            path, **{key: value for key, value in result.items() if key != "trace"}
        )
        save_json(folder / "trace.json", result["trace"])
        if index is not None:
            np.savez_compressed(folder / "index.npz", index=index)
        save_json(
            folder / "complete.json",
            stamp(seconds=time.perf_counter() - started, fit_sha256=digest(path)),
        )
    if "query_y" in panels:
        logits, answers = result["prediction"], panels["query_y"]
        loss = (logsumexp(logits, axis=-1) - (logits * answers).sum(-1)).mean(1)
        accuracy = (logits.argmax(-1) == answers.argmax(-1)).mean(1)
        result["evaluation_ce"] = float(loss.mean())
        print(
            json.dumps(
                {
                    "stage": cohort,
                    "method": name,
                    "ce": float(loss.mean()),
                    "accuracy": float(accuracy.mean()),
                }
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "stage": cohort,
                    "method": name,
                    "predictions_saved": len(panels["train_x"]),
                }
            ),
            flush=True,
        )
    return result


def run():
    torch.set_num_threads(2)
    started = time.perf_counter()
    ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registration = OUTPUT / "registered.json"
    if not registration.exists():
        snapshot = OUTPUT / "executed_source"
        snapshot.mkdir()
        for path in Path(__file__).parent.iterdir():
            if path.is_file():
                shutil.copy2(path, snapshot / path.name)
        save_json(
            registration,
            stamp(
                source_files={
                    str(path): digest(path) for path in (SOURCE, *RETAINED, MSR)
                },
                class_splits={
                    split: digest(IMAGES / f"{split}_classes.json")
                    for split in ("train", "val", "test")
                },
                penalties=PENALTIES,
                depths=DEPTHS,
                budgets=[16, 64, 256, INPUTS * HIDDEN],
                development_tasks=48,
                test_tasks=128,
                test_seed=3511158404,
                development_seed=3511158403,
                protocol_sha256=digest(Path(__file__).with_name("PROTOCOL.md")),
            ),
        )
    bank_path = ROOT / "source_bank.npz"
    if bank_path.exists():
        with np.load(bank_path) as arrays:
            bank = dict(arrays)
    else:
        bank = source_bank()
        np.savez_compressed(bank_path, **bank)
        save_json(
            OUTPUT / "source_frozen.json",
            stamp(
                bank_sha256=digest(bank_path),
                entries={
                    name: {
                        "codes": int(np.abs(index).max()),
                        "density": float(np.mean(index != 0)),
                    }
                    for name, index in bank.items()
                },
                test_tasks_generated=False,
            ),
        )
    development_path = ROOT / "development.npz"
    if not development_path.exists():
        panels, provenance = episodes("val", 48, 3511158403)
        np.savez_compressed(development_path, **panels)
        save_json(ROOT / "development_provenance.json", provenance)
    with np.load(development_path) as arrays:
        panels = dict(arrays)
    dense = evaluate("dense", panels)
    normal, uncertainty = dense_descriptor(dense)
    scores = []
    for name, index in bank.items():
        fit = evaluate(name, panels, index=index)
        scores.append(
            {
                "name": name,
                "codes": int(np.abs(index).max()),
                "ce": fit["evaluation_ce"],
            }
        )
    selected = min(scores, key=lambda row: (row["ce"], row["codes"]))
    initial = bank[selected["name"]]
    print(json.dumps({"selected_common_indexator": selected}), flush=True)
    aligned, error = align(initial, normal, uncertainty)
    adaptation_scores = []
    for penalty in PENALTIES:
        for depth in DEPTHS:
            name = f"adapt_l{penalty:g}_d{depth}"
            index = project(initial, aligned, error, penalty, depth)
            fit = evaluate(name, panels, index=index)
            adaptation_scores.append(
                {
                    "name": name,
                    "penalty": penalty,
                    "depth": depth,
                    "ce": fit["evaluation_ce"],
                    "changes": float(np.mean(index != initial)),
                }
            )
    fixed_choice = {
        "name": selected["name"],
        "penalty": 0.0,
        "depth": 0,
        "ce": selected["ce"],
        "changes": 0.0,
    }
    plain = min(
        [fixed_choice] + [row for row in adaptation_scores if row["penalty"] == 0],
        key=lambda row: (row["ce"], row["depth"]),
    )
    prior = min(
        [fixed_choice] + adaptation_scores,
        key=lambda row: (row["ce"], row["depth"], row["penalty"]),
    )
    single_scores = []
    for codes in (16, 64, 256):
        index = np.stack([scalar_seed(weight, codes) for weight in normal])
        name = f"single_m{codes}"
        fit = evaluate(name, panels, index=index)
        single_scores.append({"name": name, "codes": codes, "ce": fit["evaluation_ce"]})
    single = min(single_scores, key=lambda row: (row["ce"], row["codes"]))
    with np.load(MSR) as arrays:
        basis = arrays["basis"].astype(np.float32)
    msr = evaluate("msr_structure", panels, basis=basis)
    choices = {
        "common": selected,
        "plain": plain,
        "prior": prior,
        "single": single,
        "source_grid": scores,
        "adaptation_grid": adaptation_scores,
        "single_grid": single_scores,
        "dense_development_ce": dense["evaluation_ce"],
        "msr_development_ce": msr["evaluation_ce"],
        "source_bank_sha256": digest(bank_path),
    }
    frozen = OUTPUT / "choices_frozen.json"
    if frozen.exists():
        previous = json.loads(frozen.read_text())
        if previous["choices"] != choices:
            raise RuntimeError("Development choices changed after freezing")
    else:
        save_json(frozen, stamp(choices=choices, test_tasks_generated=False))
    target = ROOT / "target_inputs.npz"
    if not target.exists():
        target_panels, provenance = episodes("test", 128, 3511158404)
        private = ROOT / "private"
        private.mkdir(exist_ok=True)
        np.savez_compressed(
            private / "query_answers.npz", y=target_panels.pop("query_y")
        )
        np.savez_compressed(target, **target_panels)
        save_json(private / "provenance.json", provenance)
        save_json(
            OUTPUT / "test_created.json",
            stamp(
                target_inputs_sha256=digest(target),
                answers_sha256=digest(private / "query_answers.npz"),
                choices_sha256=digest(frozen),
                test_query_answers_used=False,
            ),
        )
    with np.load(target) as arrays:
        panels = dict(arrays)
    dense = evaluate("dense", panels, cohort="test")
    normal, uncertainty = dense_descriptor(dense)
    aligned, error = align(initial, normal, uncertainty)
    entries = {
        "dense": "dense",
        "fixed_g34": "fixed_g34",
        "msr_structure": "msr_structure",
        "single": "single",
    }
    evaluate("fixed_g34", panels, index=initial, cohort="test")
    for method, policy in (("adapt", plain), ("prior", prior)):
        index = project(initial, aligned, error, policy["penalty"], policy["depth"])
        evaluate(method, panels, index=index, cohort="test")
        entries[method] = method
        save_json(
            ROOT / "test" / method / "changes.json",
            {
                "fraction": np.mean(index != initial, axis=(1, 2)).tolist(),
                "zero_fraction": np.mean(index == 0, axis=(1, 2)).tolist(),
            },
        )
    index = np.stack([scalar_seed(weight, single["codes"]) for weight in normal])
    evaluate("single", panels, index=index, cohort="test")
    evaluate("msr_structure", panels, basis=basis, cohort="test")
    save_json(
        OUTPUT / "predictions_frozen.json",
        stamp(
            choices_sha256=digest(frozen),
            inputs_sha256=digest(target),
            predictions={
                name: digest(ROOT / "test" / folder / "fit.npz")
                for name, folder in entries.items()
            },
            numerical_curves={
                name: digest(ROOT / "test" / folder / "trace.json")
                for name, folder in entries.items()
            },
            test_evaluation_answers_opened=False,
        ),
    )
    save_json(
        OUTPUT / "fitting_complete.json", stamp(seconds=time.perf_counter() - started)
    )
    print(
        json.dumps(
            {"fitting_complete": True, "seconds": time.perf_counter() - started}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
