"""Rank complete signature partitions using the existing geometric score."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.theorem_pair_recovery_v1.recover import score_magnitudes


def canonical_groups(index):
    result = np.zeros_like(index)
    rename = {}
    for label in index.flat:
        if label and abs(label) not in rename:
            rename[abs(label)] = (len(rename) + 1) * (1 if label > 0 else -1)
    for label, value in rename.items():
        result[index == label] = value
        result[index == -label] = -value
    return result


def best_signs(index, normal, uncertainty):
    """Vectorize the original sign starts/votes; keep its lowest-error result."""
    rows, columns = np.where(index)
    labels = index[rows, columns] - 1
    codes = int(labels.max()) + 1
    observed = np.where(normal[:, rows, columns] >= 0, 1, -1)
    weight = np.clip(
        np.abs(normal[:, rows, columns]) / np.maximum(uncertainty[:, columns], 0.0003),
        0.01,
        10.0,
    )
    matrix = weight * observed
    order = np.argsort(labels, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(labels, minlength=codes))[:-1]]
    sorted_matrix = matrix[:, order]
    total = float(weight.sum())
    best = None
    for reference in range(len(normal)):
        signs = observed[reference].copy()
        for refinement in range(2):
            if refinement:
                for _ in range(8):
                    vote = np.add.reduceat(sorted_matrix * signs[order], starts, axis=1)
                    profile = np.where(vote >= 0, 1, -1)
                    update = np.where((matrix * profile[:, labels]).sum(0) >= 0, 1, -1)
                    if np.array_equal(update, signs):
                        break
                    signs = update
            vote = np.add.reduceat(sorted_matrix * signs[order], starts, axis=1)
            error = max(0.0, (total - float(np.abs(vote).sum())) / (2 * total))
            signed = index.copy()
            signed[rows, columns] *= signs
            signed = canonical_groups(signed)
            key = error, signed.tobytes()
            if best is None or key < best[0]:
                best = (
                    key,
                    signed.copy(),
                    {"sign_reference": reference, "sign_refined": bool(refinement)},
                )
    return best[1], best[0][0], best[2]


def run(root, output):
    if not (output / "partitions_frozen.json").exists():
        raise RuntimeError("All rank/threshold scans must finish before ranking")
    boundary = StudyBoundary(root)
    boundary.install()
    spaces = json.loads((output / "spaces.json").read_text())["spaces"]
    spaces = {row["key"]: row for row in spaces}
    source_spaces = root / "source" / "signature_spaces"
    observations = {}
    for anchor in {row["anchor"] for row in spaces.values()}:
        observations[anchor] = dict(np.load(source_spaces / f"anchor_{anchor}.npz"))
    unsigned = {}
    started = time.perf_counter()
    for metadata in sorted((output / "ranks").rglob("r*.json")):
        record = json.loads(metadata.read_text())
        arrays = np.load(metadata.with_suffix(".npz"))
        anchor = spaces[record["key"]]["anchor"]
        normal, uncertainty = [
            observations[anchor][key] for key in ("normal", "uncertainty")
        ]
        for number, origin in enumerate(record["candidates"]):
            index = arrays[f"candidate_{number}"]
            codes = origin["codes"]
            score, _ = score_magnitudes(index, normal, uncertainty, codes)
            key = codes, canonical_groups(index).tobytes()
            previous = unsigned.get(key)
            if previous is not None and previous["log_score"] <= score:
                continue
            unsigned[key] = {
                "index": index.copy(),
                "log_score": score,
                "anchor": anchor,
                "origin": {"space": record["key"], **origin},
                "codes": codes,
            }
            print(
                json.dumps(
                    {
                        "scored": record["key"],
                        "rank": origin["rank"],
                        "codes": codes,
                        "log_score": score,
                    }
                ),
                flush=True,
            )
    candidates, all_ranked = {}, {}
    for codes in (8, 16, 32, 64):
        ordered = sorted(
            (row for row in unsigned.values() if row["codes"] == codes),
            key=lambda row: (row["log_score"], row["index"].tobytes()),
        )[:32]
        signed_rows = []
        for row in ordered:
            observation = observations[row["anchor"]]
            index, error, sign_origin = best_signs(
                row["index"], observation["normal"], observation["uncertainty"]
            )
            signed_rows.append(
                {
                    "index": index,
                    "score": row["log_score"] + 4 * error,
                    "log_score": row["log_score"],
                    "sign_error": error,
                    "origin": {**row["origin"], **sign_origin},
                    "codes": codes,
                }
            )
        signed_rows.sort(key=lambda row: (row["score"], row["index"].tobytes()))
        all_ranked[str(codes)] = [
            {k: v for k, v in row.items() if k != "index"} for row in signed_rows
        ]
        if not signed_rows:
            continue
        first = signed_rows[0]
        candidates[f"k{codes}"] = first["index"]
        folder = root / "fits" / f"signature_k{codes}"
        folder.mkdir()
        # Same basis normalization as G34's common source fitter; no source
        # numerical values, losses or fitted state determine this choice.
        index = first["index"]
        basis = (
            (np.abs(index)[..., None] == np.arange(1, codes + 1))
            * np.sign(index)[..., None]
        ).astype(np.float32)
        basis /= max(float(np.sqrt((basis**2).sum(-1).mean())), 1e-12)
        np.savez_compressed(folder / "selected.npz", basis=basis, index=index)
        save_json(
            folder / "selected.json",
            {
                "method": "signature",
                "codes": codes,
                "source_M_loss": None,
                "selection": "First geometric candidate; source M not read or used",
                "geometric_score": first["score"],
                "origin": first["origin"],
                "selected_sha256": digest(folder / "selected.npz"),
            },
        )
        print(
            json.dumps(
                {
                    "first_geometric_budget": codes,
                    "score": first["score"],
                    "active": int(np.count_nonzero(index)),
                }
            ),
            flush=True,
        )
    np.savez_compressed(output / "first_candidates.npz", **candidates)
    save_json(
        output / "geometric_ranking.json",
        {
            "by_budget": all_ranked,
            "unsigned_partitions": len(unsigned),
            "criterion": "existing robust standardized log score + 4 sign disagreement",
            "first_candidate_only": True,
            "seconds": time.perf_counter() - started,
        },
    )
    save_json(
        output / "geometric_choices_frozen.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "structures_sha256": digest(output / "first_candidates.npz"),
            "ranking_sha256": digest(output / "geometric_ranking.json"),
            "available_budgets": [int(key[1:]) for key in candidates],
        },
    )
    boundary.save(output / "ranking_access_receipt.json", "complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
