"""Apply the frozen projection-consensus algorithm to the original sine task."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .consensus_search import search
from .data import BASE, digest, save_json
from .run import refit_search
from .train import fit, save_fit


def configurations():
    rows = []
    for radius, rho, charge in itertools.product(
        (0.5, 2.0, 6.0, 8.0), (1.0, 100.0), (0.0, 0.003, 0.03, 0.3)
    ):
        rows.append(
            {
                "configuration": len(rows),
                "family": "sine_consensus",
                "radius": radius,
                "rho_final": rho,
                "nonzero_penalty": charge,
                "dual_step": 0.25,
                "numeric_lr": 0.01,
                "latent_lr": 0.01,
                "mode": "joint",
            }
        )
    return rows


def prepare(root, parent):
    root.mkdir(parents=True, exist_ok=False)
    frozen = BASE / "frozen_methods/deepsets_consensus_exact_v1"
    method = json.loads((frozen / "method.json").read_text())
    for name, expected in method["copied_sha256"].items():
        if digest(frozen / name) != expected:
            raise RuntimeError(f"Frozen method changed: {name}")
    save_json(
        root / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "method": str(frozen.resolve()),
            "method_sha256": digest(frozen / "method.json"),
            "frequencies": [4, 6, 8],
            "candidates": configurations(),
            "pilot_seeds": [401, 402],
            "confirmation_seeds": [403, 404, 405],
            "source_heldout_seed_offset": 17100000,
            "transfer_seed_offset": 18100000,
            "transfer_trained_functions": 4,
            "transfer_capacity_directions": 32,
            "rules": [
                "Keep the existing 24-input Gaussian two-sine teacher and one-sine student with free amplitude and no phase or offset. The population global minimum is nonzero; new seeds are not new source functions.",
                "Execute the existing consensus_search.search and exact signed DP projection without changing them. All24 positions remain available; no teacher direction, group membership, group sizes or nonzero count enter learning.",
                "Three free codes, fixed zero and tied negatives. Same2000dense warmup+10000Z updates, projection every50, dual.25, rho.01 to1/100, normalized task loss plus L2, Adam.01. Z is retained; no task-C fit or Z=C[I] copy inside the trajectory.",
                "Register32 radius/rho/charge configurations before the pilot. Keep the original frozen setting radius.5,rho100,lambda.3 as a separate confirmed reference even if pilot chooses another setting.",
                "Pilot401/402 chooses one setting by mean selection after hard-C refitting. Confirmation403-405 uses the fixed pilot winner and original setting. Each I snapshot gets1inherited+3freshGaussian C starts,3000steps; query checkpoints, mean4 selection ranks structures.",
                "Original support/query/selection observations are reused. Existing dense random, dense moment, dense long and oracle numerical calibration states can be reused only after verifying their frozen hashes and identical source/fit specifications; their origin/code provenance is recorded. No new training cost is charged for reused fits.",
                "Fresh numerical comparison calibrates rates/radii by mean selection of8starts and evaluates32new starts per method. Dense moment uses support-only mean(X*y), with2percent jitter, and is explicitly separated from random initialization. Dense long is an archived equal-total-update control with2starts*12000 versus8*3000.",
                "Teacher audits, exact Gaussian population risks, capacities and global-minimum references are computed only after source/numerical freeze. Test is generated after that freeze; neither teacher metrics nor test select a learner.",
                "Before generating new transfer directions freeze all source I. Four genuinely new Gaussian codebooks define normalized directions with the original hidden grouping. Fit learned/shuffled/oracle I and dense random/moment on new observations with all fresh numbers,3000steps,8starts,12rate/radius configurations. Report selected and all-start results.32direction analytic capacities are diagnostics, not trained outcomes.",
                "No software tests, no overwrite of existing results or frozen method, no commit/push.",
            ],
        },
    )
    snapshot = root / "code_snapshot"
    snapshot.mkdir()
    for name in (
        "sine_consensus.py",
        "consensus_search.py",
        "codebook_projection.py",
        "model.py",
        "run.py",
        "train.py",
        "evaluate.py",
        "sine_capacity.py",
        "data.py",
    ):
        shutil.copy2(BASE / name, snapshot / name)
    for frequency in (4, 6, 8):
        origin = parent / f"sine_{frequency}"
        output = root / f"sine_{frequency}"
        output.mkdir()
        plan = json.loads((origin / "protocol.json").read_text())
        spec = json.loads((origin / "specification.json").read_text())
        for name in (
            "hard_refit",
            "hard_refit_radii",
            "hard_refit_restarts_per_radius",
        ):
            spec.pop(name, None)
        spec["name"] = f"sine_{frequency}_consensus"
        plan["search"].update(
            pilot_seeds=[401, 402],
            confirmation_seeds=[403, 404, 405],
            pilot_steps=12000,
            confirmation_steps=12000,
            numeric_warmup=2000,
            restarts=1,
            top_configurations=1,
            snapshots=[0, 1 / 3, 2 / 3, 1],
        )
        plan["consensus"] = {
            "block_steps": 50,
            "rho_initial": 0.01,
            "projection": "dynamic",
        }
        for name in ("source.npz", "teacher_audit_only.npz"):
            shutil.copy2(origin / name, output / name)
        save_json(output / "protocol.json", plan)
        save_json(output / "specification.json", spec)
        cache = output / "cached_controls"
        cache.mkdir()
        old_frozen = json.loads((origin / "frozen.json").read_text())
        copied = {}
        for variant in ("dense_shared", "dense_moment", "dense_long", "oracle"):
            key = f"controls/{variant}.npz"
            if digest(origin / key) != old_frozen["sha256"][key]:
                raise RuntimeError(f"Referenced control changed: {origin / key}")
            for suffix in ("npz", "json"):
                path = origin / "controls" / f"{variant}.{suffix}"
                shutil.copy2(path, cache / path.name)
                copied[path.name] = digest(path)
        save_json(
            cache / "provenance.json",
            {
                "origin": str(origin.resolve()),
                "source_sha256": digest(origin / "source.npz"),
                "fit_protocol": plan["fit"],
                "sha256": copied,
                "original_code_sha256": {
                    name: digest(origin / "code_snapshot" / name)
                    for name in ("model.py", "train.py", "run.py")
                },
                "reuse": "Archived fits on the identical observations and original fit protocol; numerical choices are recomputed by the registered mean-selection rule. New32-start evaluations are actually trained.",
            },
        )
    save_json(
        root / "prepared.json",
        {
            "sha256": {
                str(path.relative_to(root)): digest(path)
                for path in root.rglob("*")
                if path.is_file()
            }
        },
    )


def run_phase(output, candidates, phase, device):
    plan = json.loads((output / "protocol.json").read_text())
    spec = json.loads((output / "specification.json").read_text())
    fixed = next(
        row["configuration"]
        for row in candidates
        if (
            row["radius"] == 0.5
            and row["rho_final"] == 100.0
            and row["nonzero_penalty"] == 0.3
        )
    )
    if phase == "confirmation":
        pilot = json.loads((output / "pilot_selection.json").read_text())
        selected = set(pilot["top_configurations"] + [fixed])
        candidates = [row for row in candidates if row["configuration"] in selected]
    rows = [
        {**row, "seed": seed}
        for row in candidates
        for seed in plan["search"][phase + "_seeds"]
    ]
    search(output, plan, spec, rows, phase, device)
    refit_search(output, plan, spec, phase, device)
    if phase == "confirmation":
        names = (
            "source.npz",
            "protocol.json",
            "specification.json",
            "architecture_selection.json",
            "confirmation/rows.json",
            "confirmation/hard_fits.npz",
            "confirmation/snapshots.npz",
        )
        save_json(
            output / "source_frozen.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "fixed_deepsets_configuration": fixed,
                "sha256": {name: digest(output / name) for name in names},
            },
        )


def numeric_rows(seed):
    return [
        {
            "radius": radius,
            "numeric_lr": rate,
            "seed": seed,
            "aligned_initial_codes": False,
        }
        for radius, rate in itertools.product((0.5, 2.0, 6.0, 8.0), (0.003, 0.01, 0.03))
    ]


def numerics(output, device):
    plan = json.loads((output / "protocol.json").read_text())
    spec = json.loads((output / "specification.json").read_text())
    choice = json.loads((output / "architecture_selection.json").read_text())
    with np.load(output / "confirmation/hard_fits.npz") as archive:
        index = archive["indices"][choice["primary"]["row"]].copy()
    shuffled = (
        np.random.default_rng(plan["seed"] + 17000003)
        .permutation(index.ravel())
        .reshape(index.shape)
    )
    # Source I and its negative control are fixed before isolated oracle access.
    np.savez_compressed(
        output / "numerical_structures.npz", learned=index, shuffled=shuffled
    )
    with np.load(output / "teacher_audit_only.npz") as archive:
        oracle = archive["indices"].copy()
    with np.load(output / "source.npz") as archive:
        moment = (archive["support_x"] * archive["support_y"][:, None]).mean(0)
    moment /= max(np.linalg.norm(moment), 1e-12)
    directory = output / "numerics"
    directory.mkdir(exist_ok=False)
    selection = {}
    methods = {
        "learned": index,
        "shuffled": shuffled,
        "oracle": oracle,
        "dense_random": None,
        "dense_moment": None,
    }
    cache_names = {
        "oracle": "oracle",
        "dense_random": "dense_shared",
        "dense_moment": "dense_moment",
    }
    for method, assignment in methods.items():
        rows = numeric_rows(501)
        if method in cache_names:
            cache_name = cache_names[method]
            provenance = json.loads(
                (output / "cached_controls/provenance.json").read_text()
            )
            cache_path = output / "cached_controls" / f"{cache_name}.npz"
            if (
                digest(cache_path) != provenance["sha256"][cache_path.name]
                or digest(output / "source.npz") != provenance["source_sha256"]
                or plan["fit"] != provenance["fit_protocol"]
            ):
                raise RuntimeError("Cached numerical control/protocol changed")
            previous_rows = json.loads(cache_path.with_suffix(".json").read_text())[
                "rows"
            ]
            ids = [i for i, row in enumerate(previous_rows) if row["method"] == "adam"]
            rows = [previous_rows[i] for i in ids]
            with np.load(cache_path) as archive:
                result = {
                    key: archive[key][:, ids]
                    if key in ("history_train", "history_query")
                    else archive[key][ids]
                    if key
                    in (
                        "numeric",
                        "weights",
                        "query_mse",
                        "selection_mse",
                        "train_mse",
                        "step",
                    )
                    else archive[key]
                    for key in archive.files
                }
            result["seconds"] = np.array(0.0)
        else:
            result = fit(
                output,
                plan,
                spec,
                rows,
                device,
                indices=np.repeat(assignment[None], len(rows), axis=0),
                restarts=8,
            )
        save_fit(directory / f"{method}_calibration", result, rows)
        number = int(result["selection_mse"].mean(-1).argmin())
        selected_rows = [{**rows[number], "seed": 502}]
        fitted = fit(
            output,
            plan,
            spec,
            selected_rows,
            device,
            indices=None if assignment is None else assignment[None],
            restarts=32,
            initial_direction=moment if method == "dense_moment" else None,
        )
        save_fit(directory / f"{method}_fresh", fitted, selected_rows)
        selection[method] = {
            "calibration_row": number,
            "settings": selected_rows[0],
            "criterion": "mean selection MSE over8starts",
            "query_selected_restart": int(fitted["query_mse"][0].argmin()),
            "calibration_reused": method in cache_names,
        }
        save_json(directory / "selection.json", selection)
        print(spec["name"], method, selection[method], flush=True)
    names = [
        "numerical_structures.npz",
        "numerics/selection.json",
        *[f"numerics/{method}_fresh.npz" for method in methods],
    ]
    save_json(
        output / "numerical_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "source_sha256": digest(output / "source_frozen.json"),
            "sha256": {name: digest(output / name) for name in names},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("prepare", "pilot", "confirmation", "numerics"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare(args.root, args.parent)
        return
    registration = json.loads((args.root / "registration.json").read_text())
    device = torch.device(args.device)
    for frequency in registration["frequencies"]:
        output = args.root / f"sine_{frequency}"
        if args.stage == "numerics":
            numerics(output, device)
        else:
            run_phase(output, registration["candidates"], args.stage, device)


if __name__ == "__main__":
    main()
