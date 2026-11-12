"""Extract a frozen signature bank from archived dense source models."""

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_joint_bank import align_source

from .recover import (
    THRESHOLDS,
    canonical_partition,
    score_magnitudes,
    signature_partitions,
    signed_proposals,
    task_precision_weights,
)


def load_sources(bundle):
    """Read only source solutions and the initial source-fitted bank."""
    arrays, hashes = {}, {}
    with tarfile.open(bundle / "numerical_state.tar.gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            permitted = (
                len(parts) == 4
                and parts[0] in ("standard", "general")
                and (
                    (parts[1] == "source" and parts[3] == "learned.npz")
                    or (parts[1] == "fits" and parts[3] == "fitted.npz")
                )
            )
            if not member.isfile() or not permitted:
                continue
            payload = archive.extractfile(member).read()
            hashes[member.name] = hashlib.sha256(payload).hexdigest()
            with np.load(io.BytesIO(payload)) as stored:
                arrays[member.name] = dict(stored)
    return arrays, hashes


def extract_case(sources, cohort, case, condition, sizes, space_weighting):
    started = time.monotonic()
    key = f"{case}_c{condition}"
    base = sources[f"{cohort}/fits/{key}/fitted.npz"]
    tasks, codes = base["parameters"].shape[1], sizes["codes"]
    observed = [
        sources[f"{cohort}/source/{key}_t{task}/learned.npz"] for task in range(tasks)
    ]
    anchors = base["validation_mse"].mean(1).argsort()[:4]
    unsigned, diagnostics = {}, []
    for anchor_number in anchors:
        anchor = base["indices"][anchor_number]
        mappings = [
            align_source(
                anchor,
                base["parameters"][anchor_number, task, :codes],
                source["normal"],
                source["uncertainty"],
            )
            for task, source in enumerate(observed)
        ]
        normal = np.stack(
            [
                source["normal"][:, mapping]
                for source, mapping in zip(observed, mappings)
            ]
        )
        uncertainty = np.stack(
            [
                source["uncertainty"][mapping]
                for source, mapping in zip(observed, mappings)
            ]
        )
        task_weights = None
        if space_weighting == "uncertainty":
            task_weights = task_precision_weights(uncertainty)
        evidence = np.abs(normal) > np.maximum(3 * uncertainty[:, None], 0.002)
        masks = {"anchor": anchor != 0}
        for votes in (1, 2, 3):
            mask = evidence.sum(0) >= votes
            if not any(np.array_equal(mask, old) for old in masks.values()):
                masks[f"votes_{votes}"] = mask
        for mask_name, mask in masks.items():
            proposals, diagnostic = signature_partitions(
                normal, mask, codes, task_weights
            )
            diagnostic["task_weights"] = (
                None if task_weights is None else task_weights.tolist()
            )
            diagnostic.update(anchor=int(anchor_number), mask=mask_name)
            diagnostics.append(diagnostic)
            for proposal in proposals:
                index = proposal.pop("index")
                canonical = canonical_partition(index, codes)
                partition_key = canonical.tobytes()
                score, guesses = score_magnitudes(index, normal, uncertainty, codes)
                if (
                    partition_key in unsigned
                    and unsigned[partition_key]["log_score"] <= score
                ):
                    continue
                unsigned[partition_key] = {
                    "index": index,
                    "canonical": canonical,
                    "normal": normal,
                    "uncertainty": uncertainty,
                    "log_score": score,
                    "code_guesses": guesses,
                    "origin": {
                        "anchor": int(anchor_number),
                        "mask": mask_name,
                        **proposal,
                    },
                }
    ranked = sorted(unsigned.values(), key=lambda item: item["log_score"])
    bank = {}
    for partition in ranked[:32]:
        for index, sign_error, origin in signed_proposals(
            partition["index"], partition["normal"], partition["uncertainty"], codes
        ):
            canonical = canonical_index(index, codes)
            bank_key = canonical.tobytes()
            score = partition["log_score"] + 4 * sign_error
            if bank_key in bank and bank[bank_key]["score"] <= score:
                continue
            bank[bank_key] = {
                "index": canonical,
                "score": score,
                "log_score": partition["log_score"],
                "sign_error": sign_error,
                "code_guesses": partition["code_guesses"],
                "origin": {**partition["origin"], **origin},
            }
    signed = sorted(bank.values(), key=lambda item: item["score"])
    shape = (0, sizes["inputs"], sizes["hidden"])
    result = {
        "indices": np.stack([item["index"] for item in signed])
        if signed
        else np.empty(shape, dtype=np.int16),
        "unsigned_indices": np.stack([item["canonical"] for item in ranked])
        if ranked
        else np.empty(shape, dtype=np.int16),
        "score": np.array([item["score"] for item in signed]),
        "unsigned_score": np.array([item["log_score"] for item in ranked]),
        "code_guesses": np.stack([item["code_guesses"] for item in signed])
        if signed
        else np.empty((0, tasks, codes)),
        "initial_indices": base["indices"],
    }
    metadata = {
        "cohort": cohort,
        "case": case,
        "condition": condition,
        "tasks": tasks,
        "anchors": anchors.tolist(),
        "unsigned_candidates": len(ranked),
        "signed_candidates": len(signed),
        "seconds": time.monotonic() - started,
        "origins": [item["origin"] for item in signed],
        "log_score": [item["log_score"] for item in signed],
        "sign_error": [item["sign_error"] for item in signed],
        "diagnostics": diagnostics,
        "teacher_access": False,
        "space_weighting": space_weighting,
    }
    return result, metadata


def extract(bundle, output, space_weighting):
    output.mkdir(parents=True, exist_ok=False)
    sources, hashes = load_sources(bundle)
    sizes = json.loads((bundle / "case_sizes.json").read_text())
    code_paths = sorted(Path(__file__).parent.glob("*.py"))
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in [*code_paths, *Path(__file__).parent.glob("*.md")]:
        shutil.copy2(path, snapshot / path.name)
    save_json(
        output / "registration.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "source_bundle": str(bundle.resolve()),
            "source_files": hashes,
            "code_sha256": {str(path): digest(path) for path in code_paths},
            "thresholds": THRESHOLDS.tolist(),
            "ranks": "all integers 0,...,m-1; candidate rank must match",
            "unsigned_limit": 32,
            "signed_numeric_limit": 32,
            "teacher_access": False,
            "retrospective_experiment": True,
            "space_weighting": space_weighting,
            "weight_rule": (
                "inverse RMS direction uncertainty; floor 0.0003; maximum normalized to one"
                if space_weighting == "uncertainty"
                else "uniform"
            ),
        },
    )
    frozen = {}
    for case_key, case_sizes in sorted(sizes.items()):
        cohort, case = case_key.split("/")
        for condition in (0, 1):
            arrays, metadata = extract_case(
                sources, cohort, case, condition, case_sizes, space_weighting
            )
            folder = output / cohort / f"{case}_c{condition}"
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "candidates.npz", **arrays)
            save_json(folder / "candidates.json", metadata)
            for path in folder.iterdir():
                frozen[str(path.relative_to(output))] = digest(path)
            print(
                json.dumps(
                    {
                        k: metadata[k]
                        for k in (
                            "cohort",
                            "case",
                            "condition",
                            "unsigned_candidates",
                            "signed_candidates",
                            "seconds",
                        )
                    }
                ),
                flush=True,
            )
    save_json(
        output / "proposal_freeze.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "files": frozen,
            "teacher_access": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=Path("artifacts/anchor_start_2026-11-10")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--space-weighting", choices=("uniform", "uncertainty"), default="uniform"
    )
    args = parser.parse_args()
    extract(args.bundle, args.output, args.space_weighting)


if __name__ == "__main__":
    main()
