"""Observation-only signed row clustering of learned continuous sharing bases."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .linear import adaptation_errors, index_basis, load_observations

SOURCES = ("msr", "mtsr", "yeh_shared_soft")
GAMMAS = (0.0, 0.001, 0.01, 0.1, 0.3, 1.0)


def load_basis(root, profile, family, phase, method, seed):
    shared = method.startswith("yeh_shared")
    hidden = profile == "tiny"
    choice_profile = "hidden" if hidden else profile
    name = (
        f"{choice_profile}_shared_hpo_frozen.json"
        if shared
        else "native_hpo_frozen.json"
        if profile == "native"
        else "hpo_frozen.json"
    )
    choice = json.loads((root / name).read_text())["choices"][method]
    directory = f"shared_{phase}" if shared else phase
    path = root / profile / f"family_{family}" / directory / f"{method}_seed_{seed}.npz"
    metadata = json.loads(path.with_suffix(".json").read_text())
    rows = metadata["inputs"]["rows"]
    row = next(
        i
        for i, settings in enumerate(rows)
        if settings["configuration"] == choice["settings"]["configuration"]
    )
    with np.load(path) as archive:
        basis = archive["basis"][row].copy()
    return basis, path


@torch.no_grad()
def proposals(basis, codes, gammas, seed, device):
    # An arbitrary invertible latent-coordinate change must not alter distances.
    # The polar factor gives invariant row distances up to orthogonal rotation.
    matrix = basis.reshape(-1, basis.shape[-1]).astype(np.float64)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    keep = singular > max(singular.max(initial=0) * 1e-7, 1e-12)
    matrix = (left * keep[None]) @ right
    values = torch.tensor(matrix, device=device)
    restarts = 8
    count = len(gammas) * restarts
    centers = values.new_zeros((count, codes, values.shape[-1]))
    generator = torch.Generator(device=device).manual_seed(seed)
    distance = values.square().sum(-1).expand(count, -1).clone()
    for code in range(codes):
        position = torch.multinomial(distance + 1e-20, 1, generator=generator)[:, 0]
        centers[:, code] = values[position]
        positive = (values[None] - centers[:, code, None]).square().sum(-1)
        negative = (values[None] + centers[:, code, None]).square().sum(-1)
        distance = torch.minimum(distance, torch.minimum(positive, negative))
    penalty = values.new_tensor(gammas).repeat_interleave(restarts)
    penalty *= values.square().sum(-1).mean()
    labels = None
    for _ in range(60):
        dictionary = torch.cat((centers[:, :1] * 0, centers, -centers), 1)
        costs = (values[None, :, None] - dictionary[:, None]).square().sum(-1)
        costs[..., 1:] += penalty[:, None, None]
        proposed = costs.argmin(-1)
        if labels is not None and torch.equal(proposed, labels):
            break
        labels = proposed
        signed = torch.where(labels > codes, codes - labels, labels)
        for code in range(1, codes + 1):
            active = signed.abs() == code
            counts = active.sum(-1)
            average = torch.einsum(
                "rn,nd->rd", (active * signed.sign()).to(values.dtype), values
            ) / counts[:, None].clamp_min(1)
            centers[:, code - 1] = torch.where(
                (counts > 0)[:, None], average, centers[:, code - 1]
            )
    signed = torch.where(labels > codes, codes - labels, labels)
    return signed.reshape(count, *basis.shape[:2]), restarts


def fit(root, profile, family, phase, method, seed, gammas, device):
    folder = root / profile / f"family_{family}"
    spec = json.loads((folder / "specification.json").read_text())
    destination = folder / f"decoded_{phase}" / f"{method}_cluster_seed_{seed}.npz"
    basis, source = load_basis(root, profile, family, phase, method, seed)
    inputs = {
        "source_sha256": digest(source),
        "source": str(source.relative_to(root)),
        "gammas": list(gammas),
        "seed": seed,
        "restarts": 8,
        "iterations": 60,
    }
    if destination.exists():
        state = json.loads(destination.with_suffix(".json").read_text())
        if state["inputs"] != inputs or digest(destination) != state["result_sha256"]:
            raise RuntimeError(f"Different existing decoder inputs: {destination}")
        return
    started = time.perf_counter()
    data = load_observations(folder, device)
    indices, restarts = proposals(basis, spec["codes"], gammas, seed + 918371, device)
    source_mse = adaptation_errors(
        index_basis(indices, spec["codes"]),
        data["train_support_x"],
        data["train_support_y"],
        data["train_query_x"],
        data["train_query_y"],
    ).mean(-1)
    winners = source_mse.reshape(len(gammas), restarts).argmin(-1)
    selected = indices[torch.arange(len(gammas), device=device) * restarts + winners]
    selected_basis = index_basis(selected, spec["codes"])
    selection = adaptation_errors(
        selected_basis,
        data["selection_support_x"],
        data["selection_support_y"],
        data["selection_query_x"],
        data["selection_query_y"],
    ).mean(-1)
    destination.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        destination,
        indices=selected.cpu().numpy(),
        basis=selected_basis.cpu().numpy(),
        selection_mse=selection.cpu().numpy(),
    )
    save_json(
        destination.with_suffix(".json"),
        {
            "inputs": inputs,
            "seconds": time.perf_counter() - started,
            "result_sha256": digest(destination),
            "completed_utc": datetime.now(UTC).isoformat(),
        },
    )
    print(profile, family, phase, method, seed, selection.tolist(), flush=True)


def confirmation(root, profile, families, seeds, device):
    choices = json.loads((root / "decoder_hpo_frozen.json").read_text())["choices"]
    for family in families:
        for seed in seeds:
            for method in SOURCES:
                fit(
                    root,
                    profile,
                    family,
                    "confirmation",
                    method,
                    seed,
                    [
                        choices["hidden" if profile == "tiny" else profile][method][
                            "gamma"
                        ]
                    ],
                    device,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stage", choices=("pilot", "confirmation"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = json.loads((args.root / "protocol.json").read_text())
    device = torch.device(args.device)
    registration = args.root / "decoder_registration.json"
    if not registration.exists():
        save_json(
            registration,
            {
                "registered_utc": datetime.now(UTC).isoformat(),
                "gammas": GAMMAS,
                "source_methods": SOURCES,
                "restarts": 8,
                "iterations": 60,
                "method": "Polar-whitened row-vector signed Lloyd clustering with fixed zero, K centers and validation-chosen nonzero charge. Explicit postprocessing baseline, not claimed as an original author algorithm.",
                "selection": "Restarts selected by source-query. Gamma selected by independent coefficient-selection tasks on pilot families. No teacher/test access.",
                "code_sha256": digest(Path(__file__)),
            },
        )
    if args.stage == "pilot":
        choices = {}
        for profile, families, phase, seeds in (
            ("hidden", plan["pilot_families"], "pilot", plan["pilot_seeds"]),
            ("native", [301], "calibration", [1]),
        ):
            choices[profile] = {}
            for method in SOURCES:
                scores = []
                for family in families:
                    for seed in seeds:
                        fit(
                            args.root,
                            profile,
                            family,
                            phase,
                            method,
                            seed,
                            GAMMAS,
                            device,
                        )
                        path = (
                            args.root
                            / profile
                            / f"family_{family}"
                            / f"decoded_{phase}"
                            / f"{method}_cluster_seed_{seed}.npz"
                        )
                        with np.load(path) as archive:
                            scores.append(archive["selection_mse"])
                mean = np.mean(scores, axis=0)
                choices[profile][method] = {
                    "gamma": GAMMAS[int(mean.argmin())],
                    "all_mean_selection_mse": mean.tolist(),
                }
        save_json(
            args.root / "decoder_hpo_frozen.json",
            {"frozen_utc": datetime.now(UTC).isoformat(), "choices": choices},
        )
    else:
        for profile, families in (
            ("hidden", plan["confirmation_families"]),
            ("native", [301]),
        ):
            confirmation(
                args.root, profile, families, plan["confirmation_seeds"], device
            )
        paths = [
            path
            for profile in ("hidden", "native")
            for path in args.root.glob(f"{profile}/family_*/decoded_confirmation/*.npz")
        ]
        expected = (
            (len(plan["confirmation_families"]) + 1)
            * len(plan["confirmation_seeds"])
            * len(SOURCES)
        )
        if len(paths) != expected:
            raise RuntimeError(f"Expected {expected} decoded models, got {len(paths)}")
        save_json(
            args.root / "decoded_models_frozen.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "sha256": {
                    **{
                        str(path.relative_to(args.root)): digest(path) for path in paths
                    },
                    "decoder_hpo_frozen.json": digest(
                        args.root / "decoder_hpo_frozen.json"
                    ),
                    "decoder_registration.json": digest(registration),
                },
            },
        )


if __name__ == "__main__":
    main()
