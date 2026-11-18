"""Register sign/grouping ablations while all new query answers are closed."""

import json

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .core import canonical
from .real import DEST, ROOT, add_model


def prepare():
    if (DEST / "predictions_frozen.json").exists():
        raise RuntimeError("Additional controls must precede the first query scoring")
    choices = json.loads((DEST / "choices_frozen.json").read_text())["choices"]
    save_json(
        DEST / "additional_controls_registered.json",
        {
            "reason": "Separate learned grouping from learned signed templates and generic regularization",
            "timing": "After main fits completed, before any query answers are scored",
            "choices_sha256": digest(DEST / "choices_frozen.json"),
            "query_answers_opened": False,
            "seed": 3511158840,
        },
    )
    jobs, mappings = [], {}
    for dataset, selected in choices.items():
        folder = ROOT / dataset
        with np.load(folder / f"models/{selected['new']['name']}.npz") as data:
            index = data["index"]
        with np.load(folder / "models/randomized_partition.npz") as data:
            randomized = data["index"]
        rng = np.random.default_rng(3511158840)
        signs = rng.choice((-1, 1), index.shape)
        variants = {
            "randomized_signs": np.abs(index) * signs,
            "randomized_both": np.abs(randomized) * signs,
            "positive_signs": np.abs(index),
        }
        mapping = {}
        for name, variant in variants.items():
            variant = canonical(variant)
            if np.array_equal(variant, index):
                mapping[name] = selected["new"]["name"]
                continue
            add_model(
                dataset,
                name,
                "control",
                index=variant,
                seed_name=selected["new"]["name"],
            )
            mapping[name] = name
            jobs.append({"dataset": dataset, "name": name, "cohort": "test"})
        mappings[dataset] = mapping
    save_json(DEST / "additional_controls_models.json", mappings)
    save_json(DEST / "controls_jobs.json", jobs)
    print(json.dumps({"jobs": len(jobs), "models": mappings}), flush=True)


if __name__ == "__main__":
    prepare()
