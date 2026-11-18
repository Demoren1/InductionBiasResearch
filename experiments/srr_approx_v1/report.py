"""Freeze all real predictions, score once, and save paired comparisons."""

import csv
import json

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .real import DEST, ROOT, combine_dense


def half_interval(values):
    return float(
        t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))
    )


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def structure_statistics(dataset, name):
    path = ROOT / dataset / f"models/{name}.npz"
    if not path.exists():
        return {
            "codes": 12544,
            "task_parameters": 12645 if dataset == "omniglot" else 12594,
        }
    with np.load(path) as data:
        if "basis" in data:
            return {"codes": data["basis"].shape[-1]}
        index = data["index"]
    codes = int(np.abs(index).max())
    counts = np.bincount(np.abs(index).ravel(), minlength=codes + 1)[1:]
    incidence = np.zeros((codes + 1, index.shape[1]), bool)
    for neuron in range(index.shape[1]):
        incidence[np.unique(np.abs(index[:, neuron])), neuron] = True
    return {
        "codes": codes,
        "density": float(np.mean(index != 0)),
        "group_size_min": int(counts.min()),
        "group_size_median": float(np.median(counts)),
        "group_size_max": int(counts.max()),
        "singleton_positions": int(np.sum(counts == 1)),
        "cross_neuron_groups": int(np.sum(incidence[1:].sum(1) > 1)),
        "task_parameters": codes + (101 if dataset == "omniglot" else 50),
    }


def score():
    if (DEST / "results.json").exists():
        raise RuntimeError(
            "New query answers were already scored; render saved results instead"
        )
    choices = json.loads((DEST / "choices_frozen.json").read_text())
    extras = json.loads((DEST / "additional_controls_models.json").read_text())
    mappings, frozen = {}, {}
    for dataset in choices["choices"]:
        folder = ROOT / dataset
        combine_dense(dataset, "test")
        registration = json.loads(
            (DEST / f"{dataset}_test_registered.json").read_text()
        )
        if registration["choices_sha256"] != digest(DEST / "choices_frozen.json"):
            raise RuntimeError("Changed choices after freezing")
        if registration["inputs_sha256"] != digest(folder / "panels/test_inputs.npz"):
            raise RuntimeError("Changed test inputs")
        mapping = {**registration["models"], **extras[dataset]}
        mappings[dataset] = mapping
        for family, name in mapping.items():
            path = folder / f"fits/test/{name}.npz"
            complete = json.loads(path.with_suffix(".complete.json").read_text())
            if digest(path) != complete["prediction_sha256"]:
                raise RuntimeError("Changed completed prediction")
            frozen[f"{dataset}/{family}"] = {"path": str(path), "sha256": digest(path)}
        for family, selected in choices["choices"][dataset].items():
            if selected["model_sha256"] is not None:
                path = folder / f"models/{selected['name']}.npz"
                if digest(path) != selected["model_sha256"]:
                    raise RuntimeError(f"Changed {family} source index")
    save_json(
        DEST / "predictions_frozen.json",
        {
            "models": frozen,
            "choices_sha256": digest(DEST / "choices_frozen.json"),
            "additional_controls_sha256": digest(
                DEST / "additional_controls_registered.json"
            ),
            "query_answers_opened": False,
        },
    )
    rows, pairs = [], []
    comparisons = (
        ("new", "dense"),
        ("new", "g34"),
        ("hybrid", "dense"),
        ("hybrid", "g34"),
        ("new", "randomized"),
        ("new", "randomized_signs"),
        ("new", "randomized_both"),
        ("new", "positive_signs"),
        ("dense24", "dense6"),
        ("new", "legacy"),
    )
    for dataset, mapping in mappings.items():
        folder = ROOT / dataset
        registration = json.loads(
            (DEST / f"{dataset}_test_registered.json").read_text()
        )
        answer_path = folder / "private/query_answers.npz"
        if digest(answer_path) != registration["answers_sha256"]:
            raise RuntimeError("Changed private evaluation answers")
        truth = np.load(answer_path)["y"]
        outcomes = {}
        for family, name in mapping.items():
            prediction = np.load(folder / f"fits/test/{name}.npz")["prediction"]
            accuracy = (prediction.argmax(-1) == truth.argmax(-1)).mean(1)
            ce = (logsumexp(prediction, -1) - (prediction * truth).sum(-1)).mean(1)
            outcomes[family] = accuracy, ce
            rows.append(
                {
                    "dataset": dataset,
                    "method": family,
                    "name": name,
                    "accuracy": float(accuracy.mean()),
                    "accuracy_half_ci": half_interval(accuracy),
                    "ce": float(ce.mean()),
                    "ce_half_ci": half_interval(ce),
                    "episodes": len(accuracy),
                    **structure_statistics(dataset, name),
                }
            )
        for first, second in comparisons:
            if first not in outcomes or second not in outcomes:
                continue
            accuracy = outcomes[first][0] - outcomes[second][0]
            ce = outcomes[first][1] - outcomes[second][1]
            pairs.append(
                {
                    "dataset": dataset,
                    "first": first,
                    "second": second,
                    "accuracy_difference": float(accuracy.mean()),
                    "accuracy_half_ci": half_interval(accuracy),
                    "ce_difference": float(ce.mean()),
                    "ce_half_ci": half_interval(ce),
                }
            )
        np.savez_compressed(
            DEST / f"{dataset}_episode_scores.npz",
            **{
                f"{method}_{kind}": value[column]
                for method, value in outcomes.items()
                for column, kind in enumerate(("accuracy", "ce"))
            },
        )
    result = {
        "comparison": rows,
        "paired": pairs,
        "choices": choices,
        "query_answers_scored_after_all_predictions_frozen": True,
    }
    save_json(DEST / "results.json", result)
    write_csv(DEST / "comparison.csv", rows)
    write_csv(DEST / "paired_differences.csv", pairs)
    write_csv(DEST / "development.csv", choices["development"])
    print(
        json.dumps(
            {
                "comparison": [
                    row
                    for row in rows
                    if row["method"]
                    in ("dense", "g34", "new", "hybrid", "legacy", "randomized")
                ],
                "paired": [row for row in pairs if row["first"] == "new"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    score()
