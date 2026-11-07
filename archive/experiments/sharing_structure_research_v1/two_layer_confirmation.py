"""G32: an independent frozen cohort for the synchronized two-layer method."""

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.darts_indexator_v1.data import digest, save_json

from .pattern_prefix import timestamp
from .synchronized_head_study import audit as synchronized_audit
from .synchronized_head_study import fit_job
from .two_layer_study import audit, make_case, make_geometry

BASE = Path(__file__).resolve().parent
PARENT = BASE / "results/20261106/guarded_two_layer_g31/confirmation"


def prepare(root):
    root.mkdir(exist_ok=False)
    folder = root / "confirmation"
    folder.mkdir()
    for name in (
        "panels",
        "data",
        "private",
        "source",
        "synchronized_head_ablation",
        "code_snapshot",
    ):
        (folder / name).mkdir()
    plan = json.loads((PARENT / "protocol.json").read_text())
    plan.update(
        stage="G32 frozen independent confirmation of synchronized I1/J2 search",
        registered_utc=timestamp(),
        seed=26110632037,
        primary="keep2_indexed_with_task_symmetry_alignment",
        parent_protocol_sha256=digest(PARENT / "protocol.json"),
        parent_sync_registration_sha256=digest(
            PARENT / "synchronized_head_ablation/registration.json"
        ),
        method="Unchanged keep2 prefix pool; canonical I1 bank; data-axis code gauge enumeration/H0 and16step numeric polish. Both axis and polished numeric states kept. Enumerate only structural signed-code/hidden-column automorphisms of each I1; reference task fixes global gauge. Native functional J2 merge paths and task gauges selected jointly by paired H1 uncertainty, including all-independent W2. Meta selects common I1/J2. At most4096 gauge combinations, otherwise record budget limit and retain identity/free fallback. No true structure, K2, roles or code values supplied.",
        confirmation_policy="Fresh six synthetic geometries and source codes/readouts/query panels/noise, balanced scalar/vector outputs, free/tied2/tied4 heads; new numerical Pattern0110 and new hidden DeepSets geometry. No source-code rejection/separation. All baselines and synchronized predictors frozen before teacher structural audit; no HPO on this cohort. Exact I1 and joint-equivalence-aware J2 are the primary outcomes; no target codebook runs.",
    )
    for case in plan["cases"]:
        if case["kind"] == "synthetic":
            case["family"] += 1000
    files = (
        "two_layer_confirmation.py",
        "two_layer_study.py",
        "guarded_prefix.py",
        "indexed_readout.py",
        "readout_symmetry.py",
        "synchronized_head_study.py",
    )
    plan["code_sha256"] = {name: digest(BASE / name) for name in files}
    for name in files:
        shutil.copy2(BASE / name, folder / "code_snapshot" / name)
    save_json(folder / "protocol.json", plan)
    geometries = {}
    for number, case in enumerate(plan["cases"]):
        key = case["geometry"]
        if key not in geometries:
            geometries[key] = make_geometry(folder, plan, case, len(geometries))
        make_case(folder, plan, case, number, geometries[key])
    save_json(
        folder / "data_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {
                str(p.relative_to(folder)): digest(p)
                for name in ("panels", "data", "private")
                for p in (folder / name).glob("*.npz")
            },
        },
    )
    policy = json.loads((BASE / "GPU_POLICY.json").read_text())
    policy["active_continuation"] = str((folder / "protocol.json").relative_to(BASE))
    policy["current_execution"] = (
        "G32 independent full two-layer structure confirmation; CPU prefix/symmetry search, packed numerical gauges GPU0-3 only. No GPU6/7."
    )
    save_json(BASE / "GPU_POLICY.json", policy)
    print(
        {"prepared_cases": len(plan["cases"]), "new_geometries": len(geometries)},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "synchronize", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
        return
    folder = root / "confirmation"
    plan = json.loads((folder / "protocol.json").read_text())
    if args.action == "synchronize":
        jobs = [
            (folder, case, condition, plan)
            for case in plan["cases"]
            for condition in range(len(plan["noise"]))
        ]
        if not all(
            (folder / "source" / f"{case['name']}_{condition}" / "fitted.json").exists()
            for _, case, condition, _ in jobs
        ):
            raise ValueError(
                "Need all base numerical candidate fits before synchronized stage"
            )
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(fit_job, jobs))
        return
    paths = list((folder / "source").glob("*/fitted.json")) + list(
        (folder / "synchronized_head_ablation").glob("*/fitted.json")
    )
    expected = 2 * len(plan["cases"]) * len(plan["noise"])
    if len(paths) != expected:
        raise ValueError(f"Need all {expected} frozen source groups")
    save_json(
        folder / "combined_source_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {str(p.relative_to(folder)): digest(p) for p in paths},
        },
    )
    audit(folder, plan)
    synchronized_audit(folder, plan)
    print({"audited_source_groups": len(paths)}, flush=True)


if __name__ == "__main__":
    main()
