"""G23 independent shared-I/source-C benchmark: registered source phases."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_data import prepare
from .shared_structure_extract import run

BASE = Path(__file__).resolve().parent
METHODS = ("prefix", "lc", "dense_dp", "lutq", "yeh")


def register(root, parent):
    if not (parent / "report/summary.json").exists():
        raise RuntimeError("G22 diagnosis must be complete")
    root.mkdir(parents=True, exist_ok=False)
    families = [
        {"family": family, "variant": variant}
        for variant, ids in (
            ("local", (9901, 9902)),
            ("quartet_mix", (9911, 9912)),
            ("random", (9921, 9922)),
        )
        for family in ids
    ]
    weights = ("physical", "uniform")
    settings = {
        "lc": [
            {"mu_initial": value, "loss_weights": weight}
            for value, weight in itertools.product((1e-6, 1e-4), weights)
        ],
        "dense_dp": [
            {"ridge": value, "loss_weights": weight}
            for value, weight in itertools.product((1e-10, 1e-4), weights)
        ],
        "lutq": [
            {"lr": value, "loss_weights": weight}
            for value, weight in itertools.product((0.003, 0.01), weights)
        ],
        "yeh": [
            {"lr": value, "loss_weights": weight, "nuclear": nuclear}
            for value, weight, nuclear in itertools.product(
                (0.01, 0.03), weights, (0.0, 0.001)
            )
        ],
    }
    plan = {
        "stage": "G23 independent common-I pilot, matched observed panels and shared candidate/meta-selection protocol; not final novelty/robustness confirmation",
        "registered_utc": datetime.now(UTC).isoformat(),
        "seed": 26110523017,
        "diagnostic_parent": str(parent.resolve()),
        "parent_summary_sha256": digest(parent / "report/summary.json"),
        "families": families,
        "inputs": 16,
        "hidden": 16,
        "codes": 4,
        "sources": 4,
        "targets": 8,
        "bias": 0.15,
        "noise": [0.2, 2.0],
        "axis_points": 1025,
        "axis_min": 0.001,
        "axis_max": 1024.0,
        "axis_repeats": 4194304,
        "slice_repeats": 4096,
        "axis_code_min": 0.0001,
        "axis_code_max": 16.0,
        "axis_starts": 7,
        "axis_solver_evaluations": 400,
        "dp_chunk": 16,
        "slice_points": 2048,
        "prefix_validation": 512,
        "slice_scales": [0.05, 0.2, 0.8, 3.2, 12.8, 51.2],
        "fit_validation": 2048,
        "meta_points": 2048,
        "rank_relative": 1e-12,
        "qp_gap_absolute": 1e-8,
        "qp_primal_tolerance": 1e-7,
        "active_iterations": 1000,
        "active_zero_tolerance": 1e-12,
        "active_direction_tolerance": 1e-12,
        "integer_moves": 4096,
        "integer_decrease": 1e-12,
        "exchange_iterations": 512,
        "composition_chunk": 16384,
        "code_refit_steps": 12,
        "code_ridge_relative": 1e-10,
        "code_line_search": 16,
        "prefix_settings": ["unit", "unit_refit", "parent_pair", "parent_pair_refit"],
        "methods": list(METHODS),
        "native_settings": settings,
        "native_restarts": 3,
        "native_stages": 24,
        "native_l_steps": 6,
        "native_warmup_stages": 4,
        "gradient_warmup_solves": 24,
        "gradient_steps": 5000,
        "gradient_checkpoint": 250,
        "gradient_batch": 1024,
        "minimum_temperature": 0.05,
        "categorical_entropy": 0.001,
        "numeric_chunk": 4096,
        "numeric_code_steps": 24,
        "source_polish_steps": 12,
        "numeric_ridge": 1e-10,
        "numeric_line_search": 16,
        "bank_per_source": 4,
        "meta_alpha": 0.05,
        "target_support": 4096,
        "target_validation": 1024,
        "target_test": 8192,
        "target_repeats": 64,
        "known": "Only D/H/K, commonb.15,readout1/sqrt(H),offset0 and observed panels. No teacherI/C/roles/activecount/groupmatching to extractors or selection. I common to4independentGaussianC. Fresh target C signed, no numeric transfer.",
        "observations": "Frozen observedmeans only; source and metaY in separate NPZ. IdenticalX across4tasks, independenttasknoise; pairedunitnoises across2sigma. Allsource/query panels cost rphysical labels. No teacherautograd queries. Allmethods receive samepanels; extra competitorHPO compute disclosed.",
        "selection": "Native/checkpoint numericalC-polish selects on fit-validation; top4uniquecanonicalI/source/method. Pool4source banks; reset everyC and fit32freshsignedstarts/source/I using train and fit-validation only. Freeze allcandidatepredictors before openingmetaY; selectonecommonI/method on newmeta. Wholecohortfreeze beforeteacherI. Common outerprotocol for allmethods; source1ablation separate.",
        "adaptation": "LC properdual and exactscalarDP, 24of144data-onlywarmup; denseGN explicitextraDP readout; LUTQ nativeLloyd/nearest+STE; Yeh-style categoricalMAP entropy/nuclear. Objective-level signed/known-bias adaptations, not unchangedauthorscripts. SamefixedI numericalpolish never changesnativeI.",
        "transfer": "AfterselectedIfreeze,8newtargetC/I, fixedI and32freshsignedstarts, support4096/r64 and independentnumericvalidation1024/r64. Test8192cleanfunctionMSE auditaftertargetfitfreeze. TrueI adaptation oracle separate. No previousC orW; exactcanonicalI reusedfits onlywithmatchingdatahash.",
        "limits": "Sixnewgeometries pilot,2perclass; not sufficientfinalconfirmation. Knowncommonbias/readout, Gaussiannoise, costlyactivequeries. Hypothesis selection/blockoptimization notnew. Independentconfirmation/freebias/readout/misspecifiednoise/priority and competitor superiority still required.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "shared_structure_benchmark.py",
                "shared_structure_data.py",
                "shared_structure_numeric.py",
                "shared_structure_native.py",
                "shared_structure_extract.py",
                "functional_observability.py",
                "codebook_counts.py",
                "prefix_census.py",
                "count_projection.py",
                "count_qp_active.py",
                "count_exchanges.py",
            )
        },
        "projection_sha256": digest(
            BASE.parent / "darts_structured_tasks_v1/codebook_projection.py"
        ),
    }
    jobs = [
        {"family_id": family_id, "task": task, "noise": sigma, "method": method}
        for family_id, task, sigma, method in itertools.product(
            range(len(families)), range(plan["sources"]), plan["noise"], METHODS
        )
    ]
    order = np.random.default_rng(plan["seed"] + 17).permutation(len(jobs))
    plan["extract_jobs"] = [jobs[int(index)] for index in order]
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in plan["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    shutil.copy2(
        BASE.parent / "darts_structured_tasks_v1/codebook_projection.py",
        snapshot / "codebook_projection.py",
    )
    for name in ("data", "extract", "logs"):
        (root / name).mkdir()


def freeze(root, plan, stage):
    destination = root / (stage + "_frozen.json")
    if destination.exists():
        raise RuntimeError("Stage already frozen")
    paths = []
    if stage == "data":
        for family_id in range(len(plan["families"])):
            base = root / "data" / f"family_{family_id:03d}"
            paths.extend(
                (
                    base.with_suffix(".json"),
                    base.with_name(base.name + "_source.npz"),
                    base.with_name(base.name + "_meta.npz"),
                )
            )
    else:
        for job_id in range(len(plan["extract_jobs"])):
            base = root / "extract" / f"job_{job_id:04d}"
            paths.extend((base.with_suffix(".json"), base.with_suffix(".npz")))
    save_json(
        destination,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": {str(path.relative_to(root)): digest(path) for path in paths},
            "protocol_sha256": digest(root / "protocol.json"),
            "teacher_audit_opened": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("register", "data", "freeze_data", "extract", "freeze_extract"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.parent)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action.startswith("freeze_"):
        freeze(args.root, plan, args.action.removeprefix("freeze_"))
        return
    length = (
        len(plan["families"]) if args.action == "data" else len(plan["extract_jobs"])
    )
    for job_id in range(args.shard, length, args.shards):
        if args.action == "data":
            prepare(args.root, plan, job_id, args.device)
        else:
            run(args.root, plan, job_id, args.device)


if __name__ == "__main__":
    main()
