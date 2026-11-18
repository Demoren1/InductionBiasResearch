"""Construct WSR at step zero and matched single-weight code controls."""

import time

import numpy as np

from experiments.paper_asrr_v1.extract import source_profiles
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import digest
from experiments.wsr_omniglot_v1.source import proximal_groups

from .clustering import CLUSTER_SEED
from .source import (
    BANKS,
    CODES,
    HIDDEN,
    INPUTS,
    OUTPUT,
    ROOT,
    WEIGHTS,
    read_json,
    save_groups,
)

WSR_ZERO = "wsr_104_step_zero"
PROFILE = "asrr_global_code_profile_exceptions"
RANDOM = "asrr_global_code_random_exceptions"
METHODS = (WSR_ZERO, PROFILE, RANDOM)


def background_index(exceptions):
    index = np.ones((INPUTS, HIDDEN), dtype=np.int16)
    index.ravel()[np.flatnonzero(exceptions.ravel())] = np.arange(2, CODES + 1)
    return index


def sample_exceptions(counts, seed):
    generator = np.random.default_rng(seed)
    mask = np.zeros((INPUTS, HIDDEN), dtype=bool)
    for neuron, count in enumerate(counts):
        inputs = generator.choice(INPUTS, size=int(count), replace=False)
        mask[inputs, neuron] = True
    if not np.array_equal(mask.sum(axis=0), counts):
        raise ValueError("The random control has different per-neuron counts")
    return mask


def construct():
    for method in METHODS:
        if (BANKS / f"{method}.npz").exists() or (
            OUTPUT / f"{method}_source.json"
        ).exists():
            raise FileExistsError(f"Already constructed: {method}")

    boundary = StudyBoundary(ROOT)
    boundary.install()
    archive = ROOT / "source/dense/models.npz"
    source_hash = digest(archive)
    if source_hash != read_json(archive.with_name("metrics.json"))["models_sha256"]:
        raise ValueError("Changed retained Dense fits")

    width_config = read_json(OUTPUT / "registration.json")
    baseline = read_json(OUTPUT / "choices_initial.json")["wsr"]
    row = next(
        row
        for row in read_json(OUTPUT / "wsr_source.json")["candidates"]
        if row["name"] == baseline["name"]
    )
    if (
        row["source_task"] != 104
        or row["step"] != 75
        or row["codes"] != CODES
        or row["model_sha256"] != baseline["model_sha256"]
    ):
        raise ValueError("Unexpected selected WSR reference")
    task = width_config["source_tasks"].index(row["source_task"])
    options = read_json(OUTPUT / "wsr_source_options.json")

    started = time.perf_counter()
    with np.load(archive) as data:
        weight = data["parameters"][task, :WEIGHTS]
    _, labels, strength, _, _ = proximal_groups(
        weight, CODES, 0.001, options["rewinding"], options["break_ties"]
    )
    index = labels.reshape(INPUTS, HIDDEN)
    save_groups(
        WSR_ZERO,
        [
            (
                WSR_ZERO,
                index,
                {
                    "step": 0,
                    "source_task": row["source_task"],
                    "reference_sha256": row["model_sha256"],
                    "dense_archive_sha256": source_hash,
                    "prox_strength": float(strength),
                },
            )
        ],
        started,
        None,
    )

    started = time.perf_counter()
    _, profiles, _, _ = source_profiles(archive, INPUTS, HIDDEN)
    common = np.median(profiles.reshape(len(profiles), -1), axis=1)
    score = np.square(profiles - common[:, None, None]).mean(axis=0)
    chosen = np.argsort(-score.ravel(), kind="stable")[: CODES - 1]
    exceptions = np.zeros((INPUTS, HIDDEN), dtype=bool)
    exceptions.ravel()[chosen] = True
    counts = exceptions.sum(axis=0)
    details = {
        "global_code_connections": WEIGHTS - (CODES - 1),
        "singleton_codes": CODES - 1,
        "exceptions_per_neuron": counts.tolist(),
        "dense_archive_sha256": source_hash,
    }
    save_groups(
        PROFILE,
        [
            (
                PROFILE,
                background_index(exceptions),
                {
                    **details,
                    "selection": "Largest mean squared deviation from the taskwise median profile",
                },
            )
        ],
        started,
        None,
    )

    started = time.perf_counter()
    random = sample_exceptions(counts, CLUSTER_SEED)
    save_groups(
        RANDOM,
        [
            (
                RANDOM,
                background_index(random),
                {
                    **details,
                    "profile_reference": PROFILE,
                    "seed": CLUSTER_SEED,
                    "overlap_with_profile_exceptions": int(
                        np.count_nonzero(exceptions & random)
                    ),
                },
            )
        ],
        started,
        None,
    )
    boundary.save(OUTPUT / "allocation_ablation_source_access.json", "complete")


if __name__ == "__main__":
    construct()
