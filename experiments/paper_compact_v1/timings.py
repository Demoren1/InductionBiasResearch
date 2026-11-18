"""Record candidate-construction times and their measurement boundaries."""

import json
import time
from statistics import median

import numpy as np
import torch

from experiments.omniglot_prior_v1 import budget, run as omniglot_prior
from experiments.paper_asrr_v1.extract import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import save_json
from experiments.srr_approx_v1.core import canonical, signed_forest

from .source import (
    HAR,
    OMNIGLOT,
    OUTPUT,
    SYNTHETIC_ORIGINAL,
    TUNING,
    read_json,
)


EXTENSION = OUTPUT.parent / "paper_recovery_extension_2026-11-14"
ASRR = OUTPUT.parent / "paper_asrr_2026-11-17"
WSR = OUTPUT.parent / "wsr_omniglot_confirmation_2026-11-18"


def replay():
    torch.set_num_threads(2)
    start = time.perf_counter()
    _, consensus, _, _ = source_profiles(
        HAR / "source/dense/models.npz", 561, 16
    )
    bank, _ = signed_forest(consensus, (128,))
    har_seconds = time.perf_counter() - start
    with np.load(TUNING / "banks/har/models/asrr_global_m128.npz") as frozen:
        if not np.array_equal(bank[128], frozen["index"]):
            raise ValueError("HAR aSRR replay changed the grouping")
    print(f"HAR aSRR {har_seconds:.3f}s", flush=True)

    common = read_json(omniglot_prior.OUTPUT / "choices_frozen.json")[
        "choices"
    ]["common"]
    start = time.perf_counter()
    seed = omniglot_prior.source_bank()[common["name"]]
    budget.BUDGETS = tuple(m for m in budget.BUDGETS if m <= 1024)
    index = canonical(budget.refinements(seed)[1024])
    omni_seconds = time.perf_counter() - start
    with np.load(
        TUNING / "banks/omniglot/models/g34_refined_m1024.npz"
    ) as frozen:
        if not np.array_equal(index, frozen["index"]):
            raise ValueError("Omniglot G34 replay changed the grouping")
    print(f"Omniglot G34 {omni_seconds:.3f}s", flush=True)
    save_json(
        OUTPUT / "timing_replay.json",
        {
            "har3_asrr": {
                "seconds": har_seconds,
                "device": "CPU",
                "scope": "Fixed global Ward grouping from saved Dense fits",
            },
            "omniglot10_g34": {
                "seconds": omni_seconds,
                "device": "CPU",
                "scope": "Original source seed proposals and refinement of the previously selected seed to 1024 codes",
            },
        },
    )


def collect():
    replayed = read_json(OUTPUT / "timing_replay.json")
    synth_g34 = read_json(SYNTHETIC_ORIGINAL / "metrics/case_04_g34.json")
    projection_log = EXTENSION / "jobs/har_g34_proposals.log"
    last = {}
    for line in projection_log.read_text().splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            last[row["projection_codes"]] = row["seconds"]
    wsr_omni = read_json(WSR / "timing.json")
    details = {
        "synthetic": {
            "g34": {
                "seconds": sum(
                    r["seconds"]
                    for r in synth_g34["stages"]
                    if r["stage"] in ("initialize", "propose")
                ),
                "scope": "Archived initialization and proposals, candidate generation",
            },
            "asrr": {
                "seconds": read_json(ASRR / "synthetic_4/extraction.json")[
                    "seconds"
                ],
                "scope": "Archived global grouping and projected candidate extraction",
            },
        },
        "omniglot10": {
            "g34": replayed["omniglot10_g34"],
            "asrr": {
                "seconds": median(
                    wsr_omni["asrr_fixed_1024_source_wall_seconds"]
                ),
                "device": "CPU",
                "scope": "One fixed input grouping, median of three runs",
            },
            "wsr": {
                "seconds": wsr_omni["wsr_16_candidate_source_wall_seconds"],
                "device": "V100 and CPU",
                "scope": "16 fixed-budget source candidates",
            },
        },
        "har3": {
            "g34": {
                "seconds": last[128] - last[64],
                "scope": "Archived duration of the fixed 128-code projection stage",
            },
            "asrr": replayed["har3_asrr"],
        },
    }
    native = {
        "synthetic": EXTENSION / "synthetic_4/source_provenance",
        "omniglot10": OMNIGLOT / "source",
        "har3": EXTENSION / "har/source_provenance",
    }
    for setting, codes in (
        ("synthetic", 4),
        ("omniglot10", 1024),
        ("har3", 128),
    ):
        rows = details[setting]
        if "wsr" not in rows:
            source = read_json(OUTPUT / setting / "wsr_source.json")
            rows["wsr"] = {
                "seconds": source["seconds"],
                "device": "V100 and CPU",
                "scope": f"{len(source['candidates'])} fixed-budget source fits",
            }
        for method, folder in (
            ("lutq", "lutq"),
            ("yeh_map", "yeh"),
            ("yeh_soft", "yeh"),
            ("msr", "msr"),
        ):
            path = native[setting] / f"{folder}_k{codes}/metrics.json"
            data = read_json(path)
            if data["status"] != "complete":
                raise ValueError(f"Incomplete source timing: {path}")
            rows[method] = {
                "seconds": data["wall_seconds"],
                "scope": "Native source candidate training, one run shared by the two Yeh rows",
                "receipt": str(path),
            }
    save_json(OUTPUT / "source_times.json", details)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("replay", "collect"))
    {"replay": replay, "collect": collect}[parser.parse_args().action]()
