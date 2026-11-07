"""Extend inner training on structures chosen by source validation loss."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from run import save_json, teacher_tensors, torch_labels
from search import metrics, successful, train_population

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=HERE / "protocol_refit.json"
    )
    args = parser.parse_args()
    extension = json.loads(args.protocol.read_text())
    source = HERE / extension["search_from"]
    output = HERE / extension["output"]
    plan = json.loads((source / "protocol.json").read_text())
    plan["seed"] += extension["seed_offset"]
    plan["search"]["population"] = extension["restarts"]
    plan["search"]["candidate_steps"] = extension["steps"]
    plan["search"]["log_every"] = extension["log_every"]
    if not torch.cuda.is_available():
        raise RuntimeError("Run on an available CUDA GPU outside sandbox")
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    for original, name in [
        (Path(__file__), "refit_snapshot.py"),
        (HERE / "search.py", "search_snapshot.py"),
        (HERE / "run.py", "run_snapshot.py"),
        (args.protocol, "extension_protocol.json"),
    ]:
        shutil.copyfile(original, output / name)
    save_json(output / "effective_protocol.json", plan)
    save_json(
        output / "seal.json",
        {
            "sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in [
                    "refit_snapshot.py",
                    "search_snapshot.py",
                    "run_snapshot.py",
                    "extension_protocol.json",
                    "effective_protocol.json",
                ]
            },
            "search_seal_sha256": hashlib.sha256(
                (source / "seal.json").read_bytes()
            ).hexdigest(),
            "stage": "conditional_refit",
        },
    )
    archive = np.load(source / "source_tasks.npz")
    tasks = [
        {
            key: archive[key][i]
            for key in [
                "parents",
                "kinds_first",
                "kinds_second",
                "kind_output",
            ]
        }
        for i in range(plan["tasks"])
    ]
    teachers = teacher_tensors(tasks, "cuda")
    generator = torch.Generator(device="cuda").manual_seed(plan["seed"] + 4000)
    selection_bits = torch.randint(
        0,
        2,
        (plan["tasks"], plan["search"]["selection_examples"], 64),
        generator=generator,
        device="cuda",
    ).float()
    data = {
        "teachers": teachers,
        "selection_x": 2 * selection_bits - 1,
        "selection_y": torch_labels(selection_bits, teachers),
    }
    results = []
    for method in plan["methods"]:
        destination = output / method["name"]
        destination.mkdir()
        winners = json.loads(
            (source / method["name"] / "winners_selection.json").read_text()
        )
        structures = np.load(source / method["name"] / "structures.npz")
        flattened = []
        chosen = []
        original_population = json.loads(
            (source / "protocol.json").read_text()
        )["search"]["population"]
        for winner in winners:
            generation = winner["generation"]
            index = winner["task"] * original_population + winner["candidate"]
            structure = {
                "masks": [
                    structures[f"mask_{layer}"][generation, index]
                    for layer in range(4)
                ],
                "scales": structures["scales"][generation, index],
            }
            chosen.append(structure)
            flattened.extend(structure for _ in range(extension["restarts"]))
        fitted, selection, steps, curves, rates = train_population(
            plan, flattened, method["inner"], 0, data, "cuda"
        )
        save_json(destination / "curves.json", curves)
        save_json(destination / "source_winners.json", winners)
        np.savez_compressed(
            destination / "fitted_models.npz",
            **{
                f"parameter_{index}": value.cpu().numpy()
                for index, value in enumerate(fitted)
            },
        )
        np.savez_compressed(
            destination / "structures.npz",
            **{
                f"mask_{layer}": np.stack([s["masks"][layer] for s in chosen])
                for layer in range(4)
            },
            scales=np.stack([s["scales"] for s in chosen]),
        )
        records = []
        for index in range(len(flattened)):
            records.append(
                {
                    "task": index // extension["restarts"],
                    "restart": index % extension["restarts"],
                    "model_index": index,
                    "learning_rate": rates[index],
                    "selected_step": steps[index],
                    "selection": {
                        key: float(value[index])
                        for key, value in selection.items()
                    },
                }
            )
        results.append((method, fitted, records))
        save_json(destination / "records_selection.json", records)
        print(
            json.dumps(
                {
                    "method": method["name"],
                    "stage": "long_refit_complete",
                    "best_selection_accuracy_per_task": selection["accuracy"]
                    .reshape(plan["tasks"], extension["restarts"])
                    .max(1)
                    .values.tolist(),
                }
            ),
            flush=True,
        )
    generator.manual_seed(plan["seed"] + extension["evaluation_seed_offset"])
    evaluation_bits = torch.randint(
        0,
        2,
        (plan["tasks"], plan["search"]["evaluation_examples"], 64),
        generator=generator,
        device="cuda",
    ).float()
    evaluation_y = torch_labels(evaluation_bits, teachers)
    summaries = []
    for method, fitted, records in results:
        scores = metrics(
            fitted,
            2 * evaluation_bits - 1,
            evaluation_y,
            extension["restarts"],
        )
        for index, record in enumerate(records):
            record["evaluation"] = {
                key: float(value[index]) for key, value in scores.items()
            }
            record["success"] = successful(
                record["evaluation"], plan["search"]
            )
        selected = [
            min(
                [row for row in records if row["task"] == task],
                key=lambda row: row["selection"]["loss"],
            )
            for task in range(plan["tasks"])
        ]
        save_json(output / method["name"] / "records_evaluation.json", records)
        save_json(
            output / method["name"] / "winners_evaluation.json", selected
        )
        summaries.append(
            {
                "method": method["name"],
                "successes": sum(row["success"] for row in selected),
                "tasks": plan["tasks"],
                "mean_accuracy": float(
                    np.mean(
                        [row["evaluation"]["accuracy"] for row in selected]
                    )
                ),
                "mean_balanced_accuracy": float(
                    np.mean(
                        [
                            (
                                row["evaluation"]["recall_positive"]
                                + row["evaluation"]["recall_negative"]
                            )
                            / 2
                            for row in selected
                        ]
                    )
                ),
                "all_successful_fits": sum(row["success"] for row in records),
            }
        )
    save_json(
        output / "summary.json",
        {
            "stage": "conditional_refit",
            "methods": summaries,
            "new_target_tasks_evaluated": False,
            "prior_trained": False,
        },
    )
    print("SUMMARY " + json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
