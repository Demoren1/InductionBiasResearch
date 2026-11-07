"""Evaluate frozen indexators on genuinely new numerical target functions."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .data import digest, save_json, teacher_predict
from .model import score
from .train import fit, save_fit


def register(output, cohort="transfer", seed_offset=8100000):
    save_json(
        output / f"{cohort}_registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "tasks": 6,
            "cohort": cohort,
            "seed_offset": seed_offset,
            "support": 1024,
            "query": 512,
            "selection": 2048,
            "test": 8192,
            "numeric_restarts": 8,
            "numeric_steps": 3000,
            "radii": [0.5, 1.5],
            "numeric_rates": [0.003, 0.01, 0.03],
            "rules": [
                "Freeze all source-family primary indexators before generating any new target function.",
                "New functions keep the teacher connectivity/grouping, but draw four new signed coefficients independently: magnitudes Uniform[0.4,1.6], equiprobable signs. Bias0.15 and sum readout1 unchanged.",
                "Target generation alone reads the teacher; learner files contain only observations and dimensions. The oracle I is an explicitly isolated numerical control, never used to alter a learned I.",
                "Each family transfers its source-selection primary I, fixed throughout all target fits. All numerical parameters, including bias/readout/offset, initialize freshly.",
                "Controls per family: shuffle group assignments on the same active support, and shuffle all positions while preserving code/sign counts. These are frozen before target fitting.",
                "Also evaluate a true-I diagnostic and full dense Adam. All get the same support/query/selection sizes,3000 updates,8 starts and six rate/radius settings; parameter counts and FLOPs differ.",
                "Query chooses numerical checkpoints and one of eight starts. Mean selection loss chooses hyperparameters within each fixed I/method. Heldout target samples are generated only after those choices freeze.",
                "Report both query-selected and mean-over-eight heldout NMSE for each new target. Never select structures or revise source settings from target results.",
                "New-coefficient target evaluation is structural transfer; fresh initializations on the old function are a separate optimizer diagnostic.",
            ],
        },
    )


def prepare(output, cohort="transfer"):
    frozen = json.loads((output / "source_frozen.json").read_text())
    for name, expected in frozen["sha256"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Source changed before transfer: {name}")
    registration_path = output / f"{cohort}_registration.json"
    registration = json.loads(registration_path.read_text())
    choices = json.loads((output / "family_selection.json").read_text())["families"]
    with np.load(output / "confirmation/hard_fits.npz") as archive:
        source_indices = archive["indices"]
    root = output / cohort
    root.mkdir(exist_ok=False)
    (root / "training_code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    plan = json.loads((output / "protocol.json").read_text())
    spec = json.loads((output / "specification.json").read_text())
    structures, records = [], []
    rng = np.random.default_rng(plan["seed"] + registration["seed_offset"] - 1)
    for family, entry in choices.items():
        primary = min(entry["per_seed"], key=lambda row: row["selection_mse"])
        index = source_indices[primary["row"]].copy()
        shuffled_groups = index.copy()
        shuffled_groups[index != 0] = rng.permutation(index[index != 0])
        shuffled_all = rng.permutation(index.ravel()).reshape(index.shape)
        for suffix, assignment in (
            ("learned", index),
            ("random_sharing", shuffled_groups),
            ("random_all", shuffled_all),
        ):
            records.append(
                {
                    "method": f"{family}/{suffix}",
                    "family": family,
                    "source_row": primary["row"],
                    "variant": suffix,
                }
            )
            structures.append(assignment)
    # This file freezes the transferable learned structures before teacher access.
    np.savez_compressed(root / "learned_structures.npz", indices=np.array(structures))
    save_json(
        root / "source_choices.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "rows": records,
            "structure_sha256": digest(root / "learned_structures.npz"),
            "source_frozen_sha256": digest(output / "source_frozen.json"),
        },
    )
    with np.load(output / "teacher_audit_only.npz") as archive:
        teacher = dict(archive)
    np.savez_compressed(root / "oracle_control_only.npz", indices=teacher["indices"])
    for task in range(registration["tasks"]):
        folder = root / f"task_{task}"
        folder.mkdir()
        task_rng = np.random.default_rng(
            plan["seed"] + registration["seed_offset"] + 100003 * task
        )
        coefficients = task_rng.uniform(0.4, 1.6, spec["codes"]) * task_rng.choice(
            [-1, 1], spec["codes"]
        )
        truth = {
            **teacher,
            "numeric": np.r_[coefficients, 0.15, 1.0, 0.0].astype(np.float32),
        }
        observations = {}
        for split in ("support", "query", "selection"):
            x = task_rng.normal(size=(registration[split], spec["inputs"])).astype(
                np.float32
            )
            observations[split + "_x"] = x
            observations[split + "_y"] = teacher_predict(x, truth, spec)
        # Center using support only; offset remains an ordinary fitted parameter.
        offset = -float(observations["support_y"].mean())
        truth["numeric"][-1] = offset
        for split in ("support", "query", "selection"):
            observations[split + "_y"] += offset
        local_spec = {
            **spec,
            "name": f"transfer_new_coefficients_{task}",
            "target_mean": float(observations["support_y"].mean()),
            "target_variance": float(observations["support_y"].var()),
        }
        np.savez_compressed(folder / "source.npz", **observations)
        np.savez_compressed(folder / "teacher_audit_only.npz", **truth)
        save_json(folder / "specification.json", local_spec)
        save_json(folder / "protocol.json", plan)
    save_json(
        root / "prepared.json",
        {
            "registration_sha256": digest(registration_path),
            "training_code_sha256": digest(root / "training_code_snapshot.py"),
            "tasks": registration["tasks"],
        },
    )


def run(output, device, reference=None, cohort="transfer"):
    root = output / cohort
    registration = json.loads((output / f"{cohort}_registration.json").read_text())
    prepared = json.loads((root / "prepared.json").read_text())
    if digest(Path(__file__)) != prepared["training_code_sha256"]:
        raise RuntimeError("Transfer implementation changed after preparation")
    choice = json.loads((root / "source_choices.json").read_text())
    if digest(root / "learned_structures.npz") != choice["structure_sha256"]:
        raise RuntimeError("Transfer structures changed")
    with np.load(root / "learned_structures.npz") as archive:
        learned = archive["indices"]
    with np.load(root / "oracle_control_only.npz") as archive:
        oracle = archive["indices"]
    methods = choice["rows"] + [{"method": "oracle", "variant": "oracle"}]
    indexators = np.concatenate((learned, oracle[None]), 0)
    results = []
    for task in range(registration["tasks"]):
        folder = root / f"task_{task}"
        plan = json.loads((folder / "protocol.json").read_text())
        spec = json.loads((folder / "specification.json").read_text())
        rows, proposals = [], []
        for method, index in zip(methods, indexators):
            for radius, rate in itertools.product(
                registration["radii"], registration["numeric_rates"]
            ):
                rows.append(
                    {
                        **method,
                        "radius": radius,
                        "numeric_lr": rate,
                        "seed": 301 + task,
                        "aligned_initial_codes": False,
                    }
                )
                proposals.append(index)
        fixed = fit(
            folder,
            plan,
            spec,
            rows,
            device,
            indices=np.array(proposals),
            restarts=registration["numeric_restarts"],
            steps=registration["numeric_steps"],
        )
        fixed["indices"] = np.array(proposals)
        save_fit(folder / "fixed_fits", fixed, rows)
        dense_rows = [
            {
                "method": "dense_adam",
                "radius": radius,
                "numeric_lr": rate,
                "seed": 301 + task,
            }
            for radius, rate in itertools.product(
                registration["radii"], registration["numeric_rates"]
            )
        ]
        if reference is None:
            dense = fit(
                folder,
                plan,
                spec,
                dense_rows,
                device,
                restarts=registration["numeric_restarts"],
                steps=registration["numeric_steps"],
            )
            save_fit(folder / "dense_fits", dense, dense_rows)
        else:
            origin = reference / cohort / f"task_{task}"
            previous = json.loads((origin / "frozen_choices.json").read_text())
            if (
                digest(origin / "dense_fits.npz")
                != previous["sha256"]["dense_fits.npz"]
            ):
                raise RuntimeError("Referenced dense control changed")
            with (
                np.load(origin / "source.npz") as old,
                np.load(folder / "source.npz") as new,
            ):
                if set(old.files) != set(new.files) or any(
                    not np.array_equal(old[key], new[key]) for key in old.files
                ):
                    raise RuntimeError("Dense control observations differ")
            old_plan = json.loads((origin / "protocol.json").read_text())
            old_spec = json.loads((origin / "specification.json").read_text())
            old_rows = json.loads((origin / "dense_fits.json").read_text())["rows"]
            if (
                old_plan["fit"] != plan["fit"]
                or old_plan["seed"] != plan["seed"]
                or old_spec != spec
                or old_rows != dense_rows
            ):
                raise RuntimeError("Dense control protocol differs")
            with np.load(origin / "dense_fits.npz") as archive:
                dense = dict(archive)
            if (
                dense["numeric"].shape[:2]
                != (len(dense_rows), registration["numeric_restarts"])
                or int(dense["history_steps"][-1]) != registration["numeric_steps"]
            ):
                raise RuntimeError("Dense control budget differs")
            for suffix in ("npz", "json"):
                shutil.copy2(
                    origin / f"dense_fits.{suffix}", folder / f"dense_fits.{suffix}"
                )
            save_json(
                folder / "dense_control_reuse.json",
                {
                    "origin": str(origin.resolve()),
                    "sha256": digest(origin / "dense_fits.npz"),
                    "observations_and_protocol_identical": True,
                },
            )
        selected = {}
        for number, row in enumerate(rows):
            value = float(fixed["selection_mse"][number].mean())
            if (
                row["method"] not in selected
                or value < selected[row["method"]]["selection_mse"]
            ):
                selected[row["method"]] = {
                    "row": number,
                    "selection_mse": value,
                    "settings": row,
                    "restart": int(fixed["query_mse"][number].argmin()),
                    "fit": "fixed",
                }
        number = int(dense["selection_mse"].mean(-1).argmin())
        selected["dense_adam"] = {
            "row": number,
            "selection_mse": float(dense["selection_mse"][number].mean()),
            "settings": dense_rows[number],
            "restart": int(dense["query_mse"][number].argmin()),
            "fit": "dense",
        }
        save_json(
            folder / "frozen_choices.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "methods": selected,
                "sha256": {
                    name: digest(folder / name)
                    for name in ("source.npz", "fixed_fits.npz", "dense_fits.npz")
                },
            },
        )
        with np.load(folder / "teacher_audit_only.npz") as archive:
            teacher = dict(archive)
        rng = np.random.default_rng(
            plan["seed"] + registration["seed_offset"] + 5000000 + task
        )
        x = rng.normal(size=(registration["test"], spec["inputs"])).astype(np.float32)
        y = teacher_predict(x, teacher, spec)
        np.savez_compressed(folder / "heldout.npz", test_x=x, test_y=y)
        task_results = {}
        for name, entry in selected.items():
            fits = fixed if entry["fit"] == "fixed" else dense
            row = entry["row"]
            losses = (
                score(
                    torch.tensor(x, device=device),
                    torch.tensor(y, device=device),
                    torch.tensor(fits["weights"][row : row + 1], device=device),
                    torch.tensor(fits["numeric"][row : row + 1], device=device),
                    spec,
                )[0]
                .cpu()
                .numpy()
                / spec["target_variance"]
            )
            task_results[name] = {
                **entry,
                "mean_test_nmse": float(losses.mean()),
                "query_selected_test_nmse": float(losses[entry["restart"]]),
                "all_test_nmse": losses.tolist(),
            }
        save_json(folder / "summary.json", task_results)
        results.append({"task": task, "methods": task_results})
        print(
            json.dumps(
                {
                    "task": task,
                    "selected_test": {
                        name: row["query_selected_test_nmse"]
                        for name, row in task_results.items()
                    },
                }
            ),
            flush=True,
        )
    names = results[0]["methods"].keys()
    aggregate = {
        name: {
            "mean_query_selected_test_nmse": float(
                np.mean(
                    [r["methods"][name]["query_selected_test_nmse"] for r in results]
                )
            ),
            "mean_all_restarts_test_nmse": float(
                np.mean([r["methods"][name]["mean_test_nmse"] for r in results])
            ),
            "per_task_selected_test_nmse": [
                r["methods"][name]["query_selected_test_nmse"] for r in results
            ],
        }
        for name in names
    }
    save_json(root / "summary.json", {"tasks": results, "aggregate": aggregate})
    text = [
        "# Перенос фиксированных индексаторов на новые коэффициенты",
        "",
        "Шесть новых целевых функций с новыми истинными C. Все извлечённые I выбраны на старой функции и фиксированы. На каждой новой функции с нуля обучаются только численные параметры; независимая validation выбирает настройки. Oracle и dense — отдельные диагностические контроли.",
        "",
        "| Метод | Средняя test NMSE, query-selected C | Средняя test NMSE по всем8 C |",
        "| --- | ---: | ---: |",
    ]
    text.extend(
        f"| {name} | {row['mean_query_selected_test_nmse']:.6g} | {row['mean_all_restarts_test_nmse']:.6g} |"
        for name, row in aggregate.items()
    )
    text.extend(
        [
            "",
            "random_sharing переставляет кодовые метки внутри той же маски, random_all перемешивает все позиции с сохранением числа кодов и их частот. Численные скорости/радиусы подбираются отдельно для каждого метода на query/selection, heldout не участвует. Это перенос между коэффициентами одной скрытой структуры, не перенос между архитектурами или датасетами.",
            "",
            f"Полный протокол: ../{cohort}_registration.json. Численные состояния и source-freeze сохранены по каждой задаче. Программные тесты не создавались и не запускались.",
            "",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(text))
    save_json(
        root / "complete.json",
        {
            "sha256": {
                str(p.relative_to(root)): digest(p)
                for p in root.rglob("*")
                if p.is_file() and p.name != "complete.json"
            }
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("register", "prepare", "run"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--cohort", choices=("transfer", "transfer_confirmation"), default="transfer"
    )
    parser.add_argument("--seed-offset", type=int, default=8100000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "register":
        register(args.root, args.cohort, args.seed_offset)
    elif args.stage == "prepare":
        prepare(args.root, args.cohort)
    else:
        run(args.root, torch.device(args.device), args.reference, args.cohort)


if __name__ == "__main__":
    main()
