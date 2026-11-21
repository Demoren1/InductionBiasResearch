"""Evaluate frozen aSRR and its controls on fresh matched episodes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_transfer_refinement_v1.common import DATASETS, OUT, SEED
from experiments.asrr_transfer_refinement_v1.confirm import PANEL_SEED, prepare
from experiments.asrr_transfer_refinement_v1.study import evaluate
from experiments.paper_transfer_v2.common import CONFIG, METHODS, freeze
from experiments.paper_transfer_v2.common import OUT as PAPER
from experiments.paper_transfer_v2.report import confidence, grouped
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .adaptive import REVISED_STAGE, STAGE as ADAPTIVE_STAGE, combine
from .fixed_blocks import STAGE as FIXED_STAGE
from .penalized import STAGE as SPECTRAL_STAGE, shuffled_bases
from .run import STAGE
from .structure import spectral_basis


def frozen_models(dataset, folder, output):
    decisions = json.loads((folder / "choices.json").read_text())
    catalog = json.loads((folder / "bank.json").read_text())
    chosen = decisions["family"][decisions["selected"]]
    guide = decisions.get("guide", decisions["family"]["group_weight_0"])
    adaptive = folder.name in (
        ADAPTIVE_STAGE,
        REVISED_STAGE,
        SPECTRAL_STAGE,
        FIXED_STAGE,
    )
    policies = {
        method: json.loads(
            (PAPER / dataset / f"{method}_choice.json").read_text()
        )
        for method in (
            ("asrr", *METHODS) if adaptive else ("asrr", "yeh_soft", "msr")
        )
    }
    for name, row in (("universal", chosen), ("guide", guide)):
        policies[name] = {**row, "model": catalog[row["candidate"]]["path"]}

    knobs = catalog[chosen["candidate"]]
    share = knobs["group_weight"]
    if share > 0:
        spectral = folder.name == SPECTRAL_STAGE
        if adaptive and not spectral:
            groups = read_data(catalog[knobs.get("groups", "mix_1")]["path"])[
                "factor"
            ]
            guides = read_data(catalog[guide["candidate"]]["path"])["factor"]
            mix = spectral_basis if folder.name == FIXED_STAGE else combine
        if not adaptive:
            count = knobs["blocks_per_site"]
            beta = knobs["gradient_weight"]
            moment = knobs["moment_weight"]
            groups = read_data(catalog[f"b{count}_g{beta:g}_s1_m0"]["path"])[
                "factor"
            ]
            guides = read_data(catalog[f"b1_g0_s0_m{moment:g}"]["path"])[
                "factor"
            ]
            mix = spectral_basis
        if not spectral:
            guides = np.repeat(guides, len(groups) // len(guides), axis=0)
        for i in range(CONFIG["target"]["shuffle_replicates"]):
            path = output / f"shuffle_{i}.npz"
            if not path.exists():
                rng = np.random.default_rng(SEED + 621173 + i)
                if spectral:
                    matrices = shuffled_bases(
                        folder / "operators.npz",
                        knobs["moment_weight"],
                        share,
                        rng,
                    )
                else:
                    matrices = [
                        mix(g[rng.permutation(len(g))], u, share)
                        for g, u in zip(groups, guides, strict=True)
                    ]
                atomic_npz(path, factor=np.asarray(matrices, dtype=np.float32))
            policies[f"shuffle_{i}"] = {**chosen, "model": str(path)}
    for row in policies.values():
        if row["model"] == "None":
            row["model"] = None
        row["sha256"] = digest(row["model"]) if row["model"] else None
    return policies


def run(dataset, device, replicate, stage):
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    folder = OUT / dataset / stage
    output = folder / f"confirmation_{replicate}"
    output.mkdir(parents=True, exist_ok=True)
    seed = PANEL_SEED + 25783 * replicate
    policies = frozen_models(dataset, folder, output)
    freeze(output / "choices.json", {"panel_seed": seed, "policies": policies})
    prepare(dataset, output, seed)
    panel = output / "test"
    units = read_data(panel / "answers.npz")["units"]
    scores = {}
    for name, policy in policies.items():
        path = Path(policy["model"]) if policy["model"] else None
        if path is not None and digest(path) != policy["sha256"]:
            raise ValueError(f"Changed frozen model {path}")
        batches = evaluate(
            dataset,
            "test",
            {name: path},
            device,
            rates=(policy["rate"],),
            steps=(policy["steps"],),
            replicates=CONFIG["target"]["seed_replicates"],
            folder=panel,
        )
        scores[name] = {
            k: v[:, 0] for k, v in next(iter(batches.values())).items()
        }

    shuffles = [
        row for key, row in scores.items() if key.startswith("shuffle_")
    ]
    if shuffles:
        scores["shuffled"] = {
            view: np.mean([row[view] for row in shuffles], axis=0)
            for view in scores["universal"]
        }
    metrics, differences = {}, {}
    for name, rows in scores.items():
        if name.startswith("shuffle_"):
            continue
        metrics[name] = {
            q: confidence(grouped(a, units)) for q, a in rows.items()
        }
        if name != "universal":
            differences[name] = {
                q: confidence(grouped(scores["universal"][q] - a, units))
                for q, a in rows.items()
            }
    atomic_npz(
        output / "per_episode.npz",
        units=units,
        **{
            f"{name}_{q}": a
            for name, row in scores.items()
            for q, a in row.items()
        },
    )
    save_json(
        output / "results.json",
        {
            "metrics": metrics,
            "universal_minus_method": differences,
            "episodes": len(units),
            "units": len(np.unique(units)),
        },
    )
    print(
        dataset,
        {
            k: {q: round(100 * a["mean"], 2) for q, a in v.items()}
            for k, v in metrics.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--device", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            STAGE,
            ADAPTIVE_STAGE,
            REVISED_STAGE,
            SPECTRAL_STAGE,
            FIXED_STAGE,
        ),
        default=STAGE,
    )
    args = parser.parse_args()
    run(args.dataset, args.device, args.replicate, args.stage)
