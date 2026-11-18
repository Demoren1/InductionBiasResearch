"""Retain all source-only families for direct development selection."""

import argparse
import json
import shutil

import numpy as np
from scipy.cluster.hierarchy import cut_tree

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.srr_approx_v1.core import (
    canonical,
    input_forest,
    local_forest,
    orient,
    synchronize_profiles,
)
from experiments.srr_real_v1.predictive import refinements

from .replicate import OLD, OLD_OUTPUT, OUTPUT, ROOT

BUDGETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def prepare(dataset):
    root, out = ROOT / "banks" / dataset, OUTPUT / "banks" / dataset
    (root / "models").mkdir(parents=True)
    out.mkdir(parents=True)
    records, seen = [], {}

    def add(name, family, index):
        codes = int(np.abs(index).max())
        if codes not in BUDGETS:
            return
        key = index.tobytes()
        if key in seen:
            old = records[seen[key]]
            old.setdefault("family_aliases", []).append(family)
            old.setdefault("name_aliases", []).append(name)
            return
        path = root / f"models/{name}.npz"
        np.savez_compressed(path, index=index.astype(np.int16))
        seen[key] = len(records)
        records.append(
            {
                "name": name,
                "family": family,
                "codes": codes,
                "model_sha256": digest(path),
            }
        )

    if dataset in ("mnist", "omniglot"):
        choices = json.loads((OLD_OUTPUT / "choices_frozen.json").read_text())
        for row in choices["development"]:
            if row["dataset"] != dataset or not (
                row["family"].startswith("asrr") or row["family"] == "g34"
            ):
                continue
            path = OLD / dataset / f"models/{row['name']}.npz"
            with np.load(path) as data:
                add(row["name"], row["family"], data["index"])
    else:
        parent = ROOT.parents[0] / "paper_asrr_2026-11-17" / dataset
        with np.load(parent / "source/asrr/profiles.npz") as data:
            selected, mean, error = (
                data[k] for k in ("selected", "consensus", "uncertainty")
            )
        with np.load(parent / "source/asrr/trees.npz") as data:
            trees = dict(data)
        profiles = mean.reshape(len(mean), -1).T
        auxiliary = orient(
            profiles, cut_tree(trees["global_tree"], n_clusters=[256])[:, 0]
        ).reshape(mean.shape[1:])
        synchronized = synchronize_profiles(auxiliary, mean, error)
        for family, tree, weights in (
            ("asrr_global", trees["global_tree"], mean),
            ("asrr_sync", trees["synchronized_tree"], synchronized),
        ):
            cuts = cut_tree(tree, n_clusters=list(BUDGETS))
            for column, codes in enumerate(BUDGETS):
                index = orient(
                    weights.reshape(len(weights), -1).T, cuts[:, column]
                ).reshape(weights.shape[1:])
                add(f"{family}_m{codes}", family, canonical(index))
        for family, bank in (
            ("asrr_local", local_forest(mean, BUDGETS)),
            ("asrr_input", input_forest(mean, BUDGETS)),
            ("asrr_selected_input", input_forest(selected, BUDGETS)),
        ):
            for codes, index in bank.items():
                add(f"{family}_m{codes}", family, index)
        for codes in (16, 32, 64, 128, 512):
            path = (
                OUTPUT.parents[0]
                / "paper_recovery_extension_2026-11-14"
                / dataset
                / f"development/g34/budgets/k{codes}/model.npz"
            )
            with np.load(path) as data:
                add(f"g34_m{codes}", "g34", data["index"])
        with np.load(
            OUTPUT.parents[0]
            / "paper_recovery_extension_2026-11-14"
            / dataset
            / "development/g34/budgets/k512/model.npz"
        ) as data:
            refined = refinements(data["index"], selected, error)
        for name, index in refined.items():
            add(name, "g34", index)
    records.append(
        {
            "name": "dense",
            "family": "dense",
            "codes": 561 * 16 if dataset == "har" else 12544,
        }
    )
    save_json(root / "models.json", records)
    save_json(out / "models.json", records)
    shutil.copytree(root / "models", out / "models")
    print(json.dumps({"dataset": dataset, "models": len(records)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    prepare(args.dataset)
