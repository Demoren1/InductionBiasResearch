"""G22 prospective pair/parent exchange ablation on frozen G21 observations."""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .count_exchanges import search_counts
from .count_metric import audit, freeze, source_data
from .count_projection import census
from .count_qp_active import function_qp
from .prefix_census import forward, make_teacher, numeric_prefix

BASE = Path(__file__).resolve().parent
METHODS = ("pair", "parent", "parent_pair", "parent_pair_refit")


def register(root, parent):
    original = json.loads((parent / "protocol.json").read_text())
    if not (parent / "report/failure_diagnosis.json").exists():
        raise RuntimeError("Frozen G21 local-barrier diagnosis is required")
    final = parent / "numeric_continuation/face_stationarity"
    if not (final / "audit.json").exists():
        raise RuntimeError("Final G21 audited source states must exist")
    root.mkdir(parents=True, exist_ok=False)
    names = (
        "count_exchanges.py",
        "count_exchange_search.py",
        "count_projection.py",
        "count_qp_active.py",
        "count_metric.py",
        "prefix_census.py",
    )
    plan = {
        **original,
        "stage": "G22 opened G21 pair/parent exchange ablation, no new observations or independent native benchmark",
        "registered_utc": datetime.now(UTC).isoformat(),
        "g21_parent": str(parent.resolve()),
        "g21_final_audit_sha256": digest(final / "audit.json"),
        "g21_diagnosis_sha256": digest(parent / "report/failure_diagnosis.json"),
        "methods": list(METHODS),
        "active_iterations": 1000,
        "active_zero_tolerance": 1e-12,
        "active_direction_tolerance": 1e-12,
        "exchange_iterations": 512,
        "composition_chunk": 16384,
        "methods_detail": "Same final certified QP, then all feasible one/two-unit transfers; or exact conditional integer parent simplex; or best of both; final ablation adds unchanged validation-selected C-refit. Strict original halfMSE decrease1e-12. Finite neighbourhood optimum only, not global integer MLE. All source states retained.",
        "source": "Identical G20 Cinit and G19 observed slices plus G21 independent prefix-validation, same seed. No new labels. Frozen G20/G21 baselines read without reruns. All available inherited labels counted.",
        "audit": "Wholefreeze360new primary states before new teacher-I audit. Afterwards true-C diagnosis2noise*3decoder*18jobs, without C-refit or axisprecision duplicates. Never revise primary using oracle outcomes.",
        "limits": "Known block/local discrete optimization, no solver novelty claim. Opened one-function/I, knownb/readout, huge inherited axisbudget. Independent common-I multi-source/native competition/transfer still required.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {name: digest(BASE / name) for name in names},
    }
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in names:
        shutil.copy2(BASE / name, snapshot / name)
    (root / "models").mkdir()


@torch.no_grad()
def run_case(root, plan, job_id, device, *, oracle=False):
    started = time.monotonic()
    g21 = Path(plan["g21_parent"])
    if (
        digest(g21 / "numeric_continuation/face_stationarity/audit.json")
        != plan["g21_final_audit_sha256"]
    ):
        raise RuntimeError("G21 final audit changed")
    if oracle and not (root / "audit.json").exists():
        raise RuntimeError("Whole G22 source freeze and audit must precede oracle")
    old, targets, vx, vy, parent = source_data(plan, job_id, device)
    archived_path = g21 / "models" / f"job_{job_id:03d}.npz"
    frozen = json.loads((g21 / "models_frozen.json").read_text())
    if digest(archived_path) != frozen["sha256"][str(archived_path.relative_to(g21))]:
        raise RuntimeError("G21 validation observations changed")
    with np.load(archived_path) as archived:
        for sigma, values in vy.items():
            if not np.array_equal(
                archived[f"validation_noise{sigma}"], values.cpu().numpy()
            ):
                raise RuntimeError("Prefix validation regenerated differently")
    truth = known_C = None
    if oracle:
        _, truth, code = make_teacher(plan["jobs"][job_id], plan["g19_plan"], device)
        known_C = np.sort(np.abs(code))
    panels = (
        [{"noise": sigma, "budget": None} for sigma in (0.2, 2.0)]
        if oracle
        else plan["panels"]
    )
    methods = METHODS[:-1] if oracle else METHODS
    records, arrays = [], {"order": old["order"]}
    if not oracle:
        arrays.update(
            {
                f"validation_noise{sigma}": values.cpu().numpy()
                for sigma, values in vy.items()
            }
        )
    for panel_id, panel in enumerate(panels):
        sigma = panel["noise"]
        reference = None
        if not oracle:
            reference = next(
                row
                for row in parent["records"]
                if row["method"] == plan["initial_C_method"]
                and row["panel"]
                == {
                    "noise": sigma,
                    "grid": plan["initial_C_grid"],
                    "coarse_equivalent_repeats": panel["budget"],
                }
            )
        centers = known_C if oracle else reference["codebook"]
        for method in methods:
            local_started = time.monotonic()
            base_method = (
                "qp_integer_refit" if method.endswith("_refit") else "qp_integer"
            )
            optimizer = partial(
                search_counts, kind=method.removesuffix("_refit"), device=device
            )
            output = census(
                old["slices"],
                targets[sigma],
                vx,
                vy[sigma],
                old["order"],
                centers,
                base_method,
                plan,
                qp_solver=function_qp,
                integer_solver=optimizer,
            )
            key = f"panel_{panel_id}__{method}"
            record = {
                "key": key,
                "panel": panel,
                "noise": sigma,
                "method": method,
                "initial_C": np.asarray(centers).tolist(),
                "final_C": output["centers"].tolist(),
                "complete": output["complete"],
                "trace": output["trace"],
                "seconds": time.monotonic() - local_started,
            }
            if oracle:
                record["exact_I"] = False
                if output["complete"]:
                    metric, _, _ = align_indices(
                        output["indices"][-1], truth, plan["codes"]
                    )
                    record.update(metric)
                    record["exact_I"] = metric["categorical_errors"] == 0
            else:
                validation_labels = (
                    plan["inputs"]
                    * plan["prefix_validation"]
                    * (1 if sigma == 0 else 4096)
                )
                record.update(
                    {
                        "parent_reference_key": reference["key"],
                        "full_available_source_labels": reference[
                            "full_available_source_labels"
                        ]
                        + validation_labels,
                        "new_prefix_validation_labels": 0,
                        "inherited_prefix_validation_labels": validation_labels,
                    }
                )
                if output["complete"]:
                    prediction = forward(
                        old["validation_x"],
                        numeric_prefix(
                            output["indices"][-1], output["centers"], device
                        ),
                        plan,
                    )
                    observed = old["validation_clean"] + sigma * old["validation_noise"]
                    record["iid_validation_mse"] = float(
                        (prediction - observed).square().mean()
                    )
            records.append(record)
            arrays[key + "__indices"] = output["indices"]
            arrays[key + "__codebooks"] = output["codebooks"]
            print(
                json.dumps(
                    {
                        "job": job_id,
                        "oracle": oracle,
                        "panel": panel,
                        "method": method,
                        "complete": output["complete"],
                        "prefixes": len(output["indices"]),
                        "reason": output["trace"][-1]["reason"],
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    path = root / "models" / (f"job_{job_id:03d}" + ("_oracle" if oracle else ""))
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    save_json(
        path.with_suffix(".json"),
        {
            "job": plan["jobs"][job_id],
            "records": records,
            "seconds": time.monotonic() - started,
            "protocol_sha256": digest(root / "protocol.json"),
            "models_frozen_sha256": digest(root / "models_frozen.json")
            if oracle
            else None,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("register", "run", "freeze", "audit", "oracle")
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
    if args.action in ("freeze", "audit"):
        path = args.root / (
            "models_frozen.json" if args.action == "freeze" else "audit.json"
        )
        if path.exists():
            raise RuntimeError("G22 phase already complete")
        (freeze if args.action == "freeze" else audit)(args.root, plan)
        return
    for job_id in range(args.shard, len(plan["jobs"]), args.shards):
        suffix = "_oracle.json" if args.action == "oracle" else ".json"
        if (args.root / "models" / (f"job_{job_id:03d}" + suffix)).exists():
            raise RuntimeError("Refusing to repeat completed G22 fit")
        run_case(args.root, plan, job_id, args.device, oracle=args.action == "oracle")


if __name__ == "__main__":
    main()
