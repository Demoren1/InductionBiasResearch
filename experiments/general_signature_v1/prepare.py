"""Freeze source-only structures and generate new sealed numerical tasks."""

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.protocol.prepare import outputs_for
from experiments.paper_structure_benchmark_v1.protocol.prepare_general import (
    general_index,
    general_numbers,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    save_json,
)

TARGET_SEED = 26111286017
SHUFFLE_SEED = 26111299117
METHODS = ("current", "signature", "signature_random", "dense")
PROJECT = Path(__file__).resolve().parents[2]


def export_structures(root, output, source, sizes):
    proposal = json.loads((source / "proposal_freeze.json").read_text())
    selection = json.loads((source / "selection_freeze.json").read_text())
    records, frozen = [], {}
    compact = output / "structures"
    compact.mkdir()
    for number in range(6):
        case = f"case_{number:02d}"
        shape = sizes[f"general/{case}"]
        for condition in (0, 1):
            key = f"{case}_c{condition}"
            original = source / "general" / key
            hashes = {}
            for name, commitment in (
                ("candidates.npz", proposal),
                ("candidates.json", proposal),
                ("selection.npz", selection),
            ):
                path = original / name
                relative = str(path.relative_to(source))
                if digest(path) != commitment["files"][relative]:
                    raise RuntimeError(f"Frozen source input changed: {relative}")
                hashes[name] = digest(path)
            with np.load(original / "selection.npz") as stored:
                current = canonical_index(stored["baseline_full_index"], 4)
            with np.load(original / "candidates.npz") as stored:
                count = len(stored["indices"])
                signature = canonical_index(stored["indices"][0], 4) if count else None
                geometric_score = float(stored["score"][0]) if count else None
            arrays = {"current": current}
            if signature is not None:
                rng = np.random.default_rng(SHUFFLE_SEED + 100003 * number)
                shuffled = signature.ravel()[rng.permutation(signature.size)]
                arrays.update(
                    signature=signature,
                    signature_random=canonical_index(shuffled.reshape(16, 16), 4),
                )
            folder = root / "targets" / key
            folder.mkdir(parents=True)
            save_json(folder / "sizes.json", shape)
            np.savez_compressed(folder / "structures.npz", **arrays)
            np.savez_compressed(compact / f"{key}.npz", **arrays)
            metadata = json.loads((original / "candidates.json").read_text())
            record = {
                "case": case,
                "condition": condition,
                "geometry": ("local", "quartet_mix", "random")[number // 2],
                "stratum": number % 2,
                "outputs": shape["outputs"],
                "signature_available": signature is not None,
                "signature_score": geometric_score,
                "signature_origin": metadata["origins"][0] if count else None,
                "source_files": hashes,
                "active_edges": {
                    name: int(np.count_nonzero(index)) for name, index in arrays.items()
                },
            }
            records.append(record)
            for name in ("sizes.json", "structures.npz"):
                path = folder / name
                frozen[str(path.relative_to(root))] = digest(path)
    save_json(
        output / "structure_freeze.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "files": frozen,
            "cases": records,
        },
    )
    return records


def generate_targets(root):
    private = root / "private"
    private.mkdir()
    for number in range(6):
        case = f"case_{number:02d}"
        geometry = ("local", "quartet_mix", "random")[number // 2]
        stratum, outputs = number % 2, 1 + number % 2
        # Only the data generator sees the reference structure and its seed.
        source_rng = np.random.default_rng(27111064019 + 100003 * number)
        index = general_index(source_rng, geometry)
        rng = np.random.default_rng(TARGET_SEED + 100003 * number)
        code, bias, head, offset = general_numbers(rng, geometry, stratum, 8, outputs)
        panels = {}
        for name, count in (("support", 1024), ("selection", 256), ("query", 2048)):
            x = rng.normal(size=(8, count, 16))
            y = np.stack(
                [
                    outputs_for(x[t], index, code[t], bias[t], head[t], offset[t])
                    for t in range(8)
                ]
            )
            panels[name] = {"x": x, "y": y}
        np.savez_compressed(
            private / f"{case}.npz",
            index=index,
            codes=code,
            bias=bias,
            readout=head,
            offset=offset,
            query_y=panels["query"]["y"],
        )
        for condition in (0, 1):
            folder = root / "targets" / f"{case}_c{condition}"
            for name in ("support", "selection"):
                np.savez_compressed(folder / f"{name}.npz", **panels[name])
            np.savez_compressed(folder / "query_x.npz", x=panels["query"]["x"])


def prepare(root, output, source):
    root.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    paths = [
        *Path(__file__).parent.glob("*.py"),
        *Path(__file__).parent.glob("*.md"),
        PROJECT / "experiments/paper_structure_benchmark_v1/fitting/target_fit.py",
        PROJECT / "experiments/paper_structure_benchmark_v1/fitting/adaptation.py",
        PROJECT / "experiments/paper_structure_benchmark_v1/fitting/numerical.py",
        PROJECT
        / "experiments/paper_structure_benchmark_v1/protocol/prepare_general.py",
    ]
    code_hashes = {}
    for path in paths:
        relative = path.relative_to(PROJECT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        code_hashes[str(relative)] = digest(path)
    save_json(
        output / "registration.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "code_sha256": code_hashes,
            "source": str(source),
            "target_seed": TARGET_SEED,
            "shuffle_seed": SHUFFLE_SEED,
            "source_outcomes_previously_known": True,
            "new_independent_source_cohort": False,
            "new_target_numerical_tasks": True,
            "target_answers_opened": False,
            "primary": "Query MSE at N=1024 by geometry and source noise",
        },
    )
    sizes = json.loads(
        (PROJECT / "artifacts/anchor_start_2026-11-10/case_sizes.json").read_text()
    )
    records = export_structures(root, output, source, sizes)
    generate_targets(root)
    jobs, identities, assignments, input_hashes = [], {}, [], {}
    for record in records:
        case, condition = record["case"], record["condition"]
        folder = root / "targets" / f"{case}_c{condition}"
        with np.load(folder / "structures.npz") as stored:
            structures = dict(stored)
        for path in folder.iterdir():
            input_hashes[str(path.relative_to(root))] = digest(path)
        for method in METHODS:
            if method != "dense" and method not in structures:
                continue
            identity = (
                case,
                "dense"
                if method == "dense"
                else hashlib.sha256(structures[method].tobytes()).hexdigest(),
            )
            if identity not in identities:
                job_id = f"{case}_c{condition}_{method}"
                identities[identity] = job_id
                jobs.append(
                    {
                        "id": job_id,
                        "case": case,
                        "condition": condition,
                        "method": method,
                    }
                )
            assignments.append(
                {**record, "method": method, "job": identities[identity]}
            )
    save_json(
        output / "inputs_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "public_sha256": input_hashes,
            "private_sha256": {p.name: digest(p) for p in (root / "private").iterdir()},
            "structure_freeze_sha256": digest(output / "structure_freeze.json"),
        },
    )
    save_json(
        output / "plan.json",
        {"root": str(root), "jobs": jobs, "assignments": assignments},
    )
    print(
        json.dumps(
            {
                "unique_jobs": len(jobs),
                "method_conditions": len(assignments),
                "signature_available": sum(r["signature_available"] for r in records),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/theorem_pair_recovery_weighted_2026-11-12"),
    )
    args = parser.parse_args()
    prepare(args.root.resolve(), args.output.resolve(), args.source.resolve())


if __name__ == "__main__":
    main()
