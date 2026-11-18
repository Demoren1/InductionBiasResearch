"""Source-only positive-scale refinement of development-screened seeds."""

import argparse
import json

import numpy as np
import torch

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.srr_approx_v1.projection import project_chunked

from .data import SETTINGS
from .replicate import OLD, OUTPUT, ROOT


def prepare(settings):
    jobs = []
    for setting in settings:
        dataset = SETTINGS[setting][0]
        models = {
            r["name"]: r
            for r in json.loads(
                (ROOT / "banks" / dataset / "models.json").read_text()
            )
        }
        options = []
        for p in (ROOT / "study" / setting / "fits/d1").glob("*.json"):
            fit = json.loads(p.read_text())
            model = models[fit["job"]["name"]]
            best = min(
                fit["metrics"].values(),
                key=lambda r: (-r["accuracy"], r["ce"]),
            )
            options.append({**model, **best})
        families = (
            "asrr_global",
            "asrr_sync",
            "asrr_local",
            "asrr_input",
            "asrr_selected_input",
            "g34",
        )
        seen = set()
        for family in families:
            candidates = [
                r
                for r in options
                if family in [r["family"], *r.get("family_aliases", [])]
            ]
            if not candidates:
                continue
            seed = min(
                candidates, key=lambda r: (-r["accuracy"], r["ce"], r["codes"])
            )
            if seed["name"] in seen:
                continue
            seen.add(seed["name"])
            name = f"{family}_projected_{setting}_m{seed['codes']}"
            jobs.append(
                {
                    "setting": setting,
                    "dataset": dataset,
                    "name": name,
                    "seed_name": seed["name"],
                    "family": "g34" if family == "g34" else "asrr_refined",
                    "codes": seed["codes"],
                    "steps": 12 if seed["codes"] <= 1024 else 4,
                }
            )
    save_json(OUTPUT / "refine_jobs.json", jobs)
    print(json.dumps({"refinement_jobs": len(jobs)}), flush=True)


def worker(job, device):
    torch.set_num_threads(2)
    dataset = job["dataset"]
    bank = ROOT / "banks" / dataset
    profiles = (
        OLD / dataset / "source_profiles.npz"
        if dataset in ("mnist", "omniglot")
        else ROOT.parent
        / "paper_asrr_2026-11-17"
        / dataset
        / "source/asrr/profiles.npz"
    )
    with np.load(profiles) as data:
        mean, error = data["consensus"], data["uncertainty"]
    seed = bank / f"models/{job['seed_name']}.npz"
    index, trace = project_chunked(
        np.load(seed)["index"], mean, error, steps=job["steps"], device=device
    )
    path = bank / f"models/{job['name']}.npz"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    np.savez_compressed(path, index=index)
    save_json(
        path.with_suffix(".json"),
        {
            **job,
            "seed_sha256": digest(seed),
            "model_sha256": digest(path),
            "source_profiles_sha256": digest(profiles),
            "trace": trace,
        },
    )


def collect():
    jobs = json.loads((OUTPUT / "refine_jobs.json").read_text())
    for dataset in {j["dataset"] for j in jobs}:
        root = ROOT / "banks" / dataset
        path = root / "models.json"
        models = json.loads(path.read_text())
        previous_sha = digest(path)
        existing = {m["name"] for m in models}
        for job in jobs:
            if job["dataset"] != dataset:
                continue
            record = json.loads(
                (root / f"models/{job['name']}.json").read_text()
            )
            if job["name"] in existing:
                raise RuntimeError("Refinement already collected")
            models.append(
                {
                    k: record[k]
                    for k in (
                        "name",
                        "family",
                        "codes",
                        "setting",
                        "model_sha256",
                    )
                }
            )
        save_json(path, models)
        save_json(
            OUTPUT / "banks" / dataset / "models_after_refinement.json",
            {"previous_registry_sha256": previous_sha, "models": models},
        )
    save_json(OUTPUT / "refinements_ready.json", {"jobs": jobs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("prepare", "worker", "collect"),
        nargs="?",
        default="worker",
    )
    parser.add_argument("--settings", nargs="+")
    parser.add_argument("--queue")
    parser.add_argument("--job", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.settings)
    elif args.phase == "collect":
        collect()
    else:
        worker(
            json.loads((OUTPUT / "refine_jobs.json").read_text())[args.job],
            args.device,
        )
