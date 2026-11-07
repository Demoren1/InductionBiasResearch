"""Transfer source-frozen sine indexators to four new numerical functions."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .data import digest, save_json, teacher_predict
from .sine_consensus import numeric_rows
from .train import fit, save_fit


def register(root):
    path = root / "transfer_registration.json"
    if path.exists():
        raise FileExistsError(path)
    save_json(
        path,
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "frequency": 6.0,
            "source_frequencies": [4, 6, 8],
            "trained_functions": 4,
            "capacity_directions": 32,
            "seed_offset": 18100000,
            "rules": [
                "Use one common f=6 target cohort for source indexators learned at f4/6/8, so source structures are compared on identical new numerical functions without repeating dense baselines per source frequency.",
                "Freeze every source I before generating new directions or observations. Draw32new iid standard-normal three-code vectors; expand through isolated teacher assignments and normalize the resulting24-vector. Train only the first4functions;32analytic capacities remain a representational diagnostic.",
                "Per source compare learned I with a frozen permutation of its signed assignments; isolated true I is a numeric oracle. All7fixed structures get12rate/radius configurations and8freshGaussian numeric starts,3000steps.",
                "Dense random and support-only moment-initialized Adam get the same12settings,8starts and3000steps. Each target uses16384support/4096query/8192selection; new16384test is generated only after numerical choices freeze.",
                "Query selects each numerical checkpoint and one restart; mean selection over8starts selects settings separately for each method. Report mean all-start as well as selected results. No source I or structure-search setting is revised using targets.",
            ],
        },
    )


def prepare(root):
    registration = json.loads((root / "transfer_registration.json").read_text())
    folder = root / "transfer"
    folder.mkdir(exist_ok=False)
    structures, records = [], []
    for frequency in registration["source_frequencies"]:
        source = root / f"sine_{frequency}"
        frozen = json.loads((source / "source_frozen.json").read_text())
        for name, expected in frozen["sha256"].items():
            if digest(source / name) != expected:
                raise RuntimeError(f"Source changed before transfer: {source / name}")
        choice = json.loads((source / "architecture_selection.json").read_text())[
            "primary"
        ]
        with np.load(source / "confirmation/hard_fits.npz") as archive:
            index = archive["indices"][choice["row"]].copy()
        rng = np.random.default_rng(18100401 + frequency)
        shuffled = rng.permutation(index.ravel()).reshape(index.shape)
        for variant, assignment in (("learned", index), ("shuffled", shuffled)):
            structures.append(assignment)
            records.append(
                {
                    "method": f"source_{frequency}/{variant}",
                    "source_frequency": frequency,
                    "variant": variant,
                    "source_choice": choice,
                }
            )
    np.savez_compressed(folder / "learned_structures.npz", indices=np.array(structures))
    save_json(
        folder / "source_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "records": records,
            "structures_sha256": digest(folder / "learned_structures.npz"),
            "registration_sha256": digest(root / "transfer_registration.json"),
        },
    )
    # Source assignments and their controls are already fixed above.
    source = root / "sine_6"
    plan = json.loads((source / "protocol.json").read_text())
    spec = json.loads((source / "specification.json").read_text())
    with np.load(source / "teacher_audit_only.npz") as archive:
        true_indices = archive["indices"].copy()
    rng = np.random.default_rng(plan["seed"] + registration["seed_offset"])
    values = rng.normal(size=(registration["capacity_directions"], spec["codes"]))
    directions = values[:, true_indices[:, 0] - 1]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    np.savez_compressed(
        folder / "teacher_audit_only.npz", indices=true_indices, directions=directions
    )
    for task in range(registration["trained_functions"]):
        output = folder / f"task_{task}"
        output.mkdir()
        local_spec = {
            **spec,
            "name": f"sine_transfer_{task}",
            "frequency": registration["frequency"],
        }
        local_rng = np.random.default_rng(
            plan["seed"] + registration["seed_offset"] + 100003 * task + 101
        )
        data = {}
        truth = {"direction": directions[task]}
        for split in ("support", "query", "selection"):
            x = local_rng.normal(size=(spec[split], spec["inputs"])).astype(np.float32)
            data[split + "_x"] = x
            data[split + "_y"] = teacher_predict(x, truth, local_spec)
        local_spec["target_variance"] = float(data["support_y"].var())
        local_spec["target_mean"] = float(data["support_y"].mean())
        np.savez_compressed(output / "source.npz", **data)
        save_json(output / "specification.json", local_spec)
        save_json(output / "protocol.json", plan)
        save_json(
            output / "prepared.json", {"source_sha256": digest(output / "source.npz")}
        )


def run(root, device):
    registration = json.loads((root / "transfer_registration.json").read_text())
    folder = root / "transfer"
    frozen = json.loads((folder / "source_frozen.json").read_text())
    if (
        digest(folder / "learned_structures.npz") != frozen["structures_sha256"]
        or digest(root / "transfer_registration.json") != frozen["registration_sha256"]
    ):
        raise RuntimeError("Transferred structures or registration changed")
    with np.load(folder / "learned_structures.npz") as archive:
        structures = list(archive["indices"])
    with np.load(folder / "teacher_audit_only.npz") as archive:
        structures.append(archive["indices"])
    methods = [record["method"] for record in frozen["records"]] + ["oracle"]
    (folder / "training_code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for task in range(registration["trained_functions"]):
        output = folder / f"task_{task}"
        plan = json.loads((output / "protocol.json").read_text())
        spec = json.loads((output / "specification.json").read_text())
        rows, indices = [], []
        for method, index in zip(methods, structures):
            for row in numeric_rows(601):
                rows.append({**row, "method": method})
                indices.append(index)
        fitted = fit(
            output, plan, spec, rows, device, indices=np.array(indices), restarts=8
        )
        fitted["indices"] = np.array(indices)
        save_fit(output / "fixed_fits", fitted, rows)
        choices = {}
        for method in methods:
            selected = min(
                (i for i, row in enumerate(rows) if row["method"] == method),
                key=lambda i: fitted["selection_mse"][i].mean(),
            )
            choices[method] = {
                "fit": "fixed_fits",
                "row": selected,
                "settings": rows[selected],
                "restart": int(fitted["query_mse"][selected].argmin()),
            }
        with np.load(output / "source.npz") as archive:
            moment = (archive["support_x"] * archive["support_y"][:, None]).mean(0)
        moment /= max(np.linalg.norm(moment), 1e-12)
        for method in ("dense_random", "dense_moment"):
            dense_rows = [{**row, "method": method} for row in numeric_rows(601)]
            dense = fit(
                output,
                plan,
                spec,
                dense_rows,
                device,
                restarts=8,
                initial_direction=moment if method == "dense_moment" else None,
            )
            save_fit(output / method, dense, dense_rows)
            selected = int(dense["selection_mse"].mean(-1).argmin())
            choices[method] = {
                "fit": method,
                "row": selected,
                "settings": dense_rows[selected],
                "restart": int(dense["query_mse"][selected].argmin()),
            }
        save_json(
            output / "choices_frozen.json",
            {
                "frozen_utc": datetime.now(UTC).isoformat(),
                "choices": choices,
                "sha256": {
                    name: digest(output / name)
                    for name in (
                        "source.npz",
                        "fixed_fits.npz",
                        "dense_random.npz",
                        "dense_moment.npz",
                    )
                },
            },
        )
        print(
            "TRANSFER FROZEN",
            task,
            {name: row["settings"] for name, row in choices.items()},
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--stage", choices=("register", "prepare", "run"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "register":
        register(args.root)
    elif args.stage == "prepare":
        prepare(args.root)
    else:
        run(args.root, torch.device(args.device))


if __name__ == "__main__":
    main()
