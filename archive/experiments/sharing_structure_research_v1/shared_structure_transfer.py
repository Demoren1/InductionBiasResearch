"""G23 fixed-I transfer: fresh target C and a separate true-I numeric control."""

import argparse
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_data import predict, teacher, weights_from_index
from .shared_structure_numeric import (
    canonical_index,
    code_prediction,
    features,
    fit_codes,
    fresh_codes,
)

BASE = Path(__file__).resolve().parent


def register(root):
    common = root / "common"
    if not (common / "selection_frozen.json").exists():
        raise RuntimeError("All source-I choices must be frozen before target work")
    destination = root / "transfer"
    destination.mkdir(exist_ok=False)
    names = (
        "shared_structure_transfer.py",
        "shared_structure_numeric.py",
        "shared_structure_data.py",
    )
    save_json(
        destination / "protocol.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "stage": "G23 fixed-I/fresh-signed-C transfer and true-I numerical adaptation oracle",
            "source_protocol_sha256": digest(root / "protocol.json"),
            "selected_source_I_freeze_sha256": digest(common / "selection_frozen.json"),
            "policy": "SelectedIimmutable. AlltargetC fresh32signedstarts and same support/numeric-validation. No previousC/W/teacherC. TrueIcontrol onlyafterselectedfreeze, separateorigin. ExactsameI targetfits may be reusedwithsameobservations/initials; no numericaltransfer across targettasks. Clean testlabels unavailable tofit, alltargetmodelsfrozenbeforetestaudit.",
            "code_sha256": {name: digest(BASE / name) for name in names},
            "theory_sha256": digest(BASE / "THEORY.md"),
        },
    )
    snapshot = destination / "executed_code_snapshot"
    snapshot.mkdir()
    for name in names:
        shutil.copy2(BASE / name, snapshot / name)
    shutil.copy2(BASE / "THEORY.md", destination / "theory_at_registration.md")
    for name in ("data", "banks", "fits", "logs"):
        (destination / name).mkdir()


@torch.no_grad()
def prepare(root, plan, family_id, device):
    transfer = root / "transfer"
    path = transfer / "data" / f"family_{family_id:03d}"
    if path.with_suffix(".json").exists():
        raise RuntimeError("Target data already prepared")
    family = plan["families"][family_id]
    index, all_codes = teacher(plan, family, tasks=plan["sources"] + plan["targets"])
    weights = weights_from_index(index, all_codes[plan["sources"] :], device)
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + family["family"] * 137 + 2000003
    )
    observed = {}
    private = {}
    for name, count in (
        ("support", plan["target_support"]),
        ("validation", plan["target_validation"]),
        ("test", plan["target_test"]),
    ):
        x = torch.randn(
            (count, plan["inputs"]),
            device=device,
            dtype=torch.float64,
            generator=generator,
        )
        clean = predict(x, weights, plan)
        if name == "test":
            private.update(
                {"test_x": x.cpu().numpy(), "test_clean": clean.cpu().numpy()}
            )
        else:
            noise = torch.randn(
                clean.shape, device=device, dtype=torch.float64, generator=generator
            )
            observed[name + "_x"] = x.cpu().numpy()
            for sigma in plan["noise"]:
                observed[f"{name}_y_{sigma}"] = (
                    (clean + sigma / np.sqrt(plan["target_repeats"]) * noise)
                    .cpu()
                    .numpy()
                )
    np.savez_compressed(path.with_suffix(".npz"), **observed)
    np.savez_compressed(path.with_name(path.name + "_audit_only.npz"), **private)
    save_json(
        path.with_suffix(".json"),
        {
            "family": family,
            "target_tasks": plan["targets"],
            "labels_per_family_condition": plan["targets"]
            * (plan["target_support"] + plan["target_validation"])
            * plan["target_repeats"],
            "test_true_function_evaluations": plan["targets"] * plan["target_test"],
            "numeric_C_transferred": False,
        },
    )


def add_index(value, method, scope, values, descriptions, lookup):
    signature = hashlib.sha256(value.tobytes()).hexdigest()
    if signature not in lookup:
        lookup[signature] = len(values)
        values.append(value)
        descriptions.append({"signature": signature, "origins": []})
    position = lookup[signature]
    descriptions[position]["origins"].append({"method": method, "scope": scope})
    return position


def banks(root, plan):
    transfer = root / "transfer"
    source = root / "common"
    frozen = json.loads((source / "selection_frozen.json").read_text())
    for relative, expected in frozen["sha256"].items():
        if digest(source / relative) != expected:
            raise RuntimeError("Frozen source-I selection changed")
    for family_id, family in enumerate(plan["families"]):
        index, _ = teacher(plan, family)
        reference = canonical_index(index, plan["codes"])
        for noise_id, sigma in enumerate(plan["noise"]):
            job_id = family_id * len(plan["noise"]) + noise_id
            destination = transfer / "banks" / f"job_{job_id:03d}"
            if destination.with_suffix(".json").exists():
                raise RuntimeError("Target bank already assembled")
            values, descriptions, lookup = [], [], {}

            reference_id = add_index(
                reference, "oracle_true_I", "oracle", values, descriptions, lookup
            )
            path = source / "selected" / f"job_{job_id:03d}"
            meta = json.loads(path.with_suffix(".json").read_text())
            records = []
            with np.load(path.with_suffix(".npz")) as arrays:
                for row in meta["records"]:
                    record = {
                        "method": row["method"],
                        "scope": row["scope"],
                        "complete": row["complete"],
                    }
                    if row["complete"]:
                        record["candidate"] = add_index(
                            arrays[f"{row['method']}__{row['scope']}__index"],
                            row["method"],
                            row["scope"],
                            values,
                            descriptions,
                            lookup,
                        )
                    records.append(record)
            np.savez_compressed(
                destination.with_suffix(".npz"), indices=np.stack(values)
            )
            save_json(
                destination.with_suffix(".json"),
                {
                    "job_id": job_id,
                    "family_id": family_id,
                    "noise": sigma,
                    "oracle_reference": reference_id,
                    "candidates": descriptions,
                    "records": records,
                    "source_I_freeze_sha256": digest(source / "selection_frozen.json"),
                    "source_I_changed": False,
                },
            )


@torch.no_grad()
def fit_job(root, plan, job_id, device):
    transfer = root / "transfer"
    path = transfer / "banks" / f"job_{job_id:03d}"
    bank = json.loads(path.with_suffix(".json").read_text())
    destination = transfer / "fits" / f"job_{job_id:03d}"
    if destination.with_suffix(".json").exists():
        raise RuntimeError("Target numeric fits already complete")
    destination.mkdir(exist_ok=True)
    family_id, sigma = bank["family_id"], bank["noise"]
    family = plan["families"][family_id]["family"]
    observed_path = transfer / "data" / f"family_{family_id:03d}.npz"
    frozen = json.loads((transfer / "data_frozen.json").read_text())
    if (
        digest(observed_path)
        != frozen["sha256"][str(observed_path.relative_to(transfer))]
    ):
        raise RuntimeError("Target observations changed")
    with np.load(observed_path) as arrays:
        x, vx = [
            torch.tensor(arrays[name], device=device)
            for name in ("support_x", "validation_x")
        ]
        y, vy = [
            torch.tensor(arrays[f"{name}_y_{sigma}"], device=device)
            for name in ("support", "validation")
        ]
    with np.load(path.with_suffix(".npz")) as arrays:
        indices = arrays["indices"]
    phase_hash, data_hash = digest(transfer / "protocol.json"), digest(observed_path)
    artefacts = []
    for number, index in enumerate(indices):
        signature = bank["candidates"][number]["signature"]
        output = destination / signature
        if output.with_suffix(".json").exists():
            old = json.loads(output.with_suffix(".json").read_text())
            if (
                old["phase_protocol_sha256"] != phase_hash
                or old["data_sha256"] != data_hash
                or old["arrays_sha256"] != digest(output.with_suffix(".npz"))
            ):
                raise RuntimeError(
                    "Existing target fit changed or belongs to another phase"
                )
            artefacts.append(str(output.relative_to(transfer)))
            continue
        started = time.monotonic()
        codes, histories = [], []
        for task in range(plan["targets"]):
            code, history = fit_codes(
                index,
                x,
                y[task],
                vx,
                vy[task],
                fresh_codes(plan, family, plan["sources"] + task, device),
                plan,
            )
            codes.append(code.cpu().numpy())
            histories.append(history)
        np.savez_compressed(
            output.with_suffix(".npz"), index=index, codes=np.stack(codes)
        )
        save_json(
            output.with_suffix(".json"),
            {
                "candidate": number,
                "signature": signature,
                "numeric_fits": histories,
                "numeric_C_transferred": False,
                "test_labels_opened": False,
                "phase_protocol_sha256": phase_hash,
                "data_sha256": data_hash,
                "arrays_sha256": digest(output.with_suffix(".npz")),
                "seconds": time.monotonic() - started,
            },
        )
        artefacts.append(str(output.relative_to(transfer)))
        print(
            json.dumps(
                {
                    "target_job": job_id,
                    "candidate": number,
                    "of": len(indices),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    save_json(
        destination.with_suffix(".json"),
        {
            "job_id": job_id,
            "fit_artefacts": artefacts,
            "bank_sha256": digest(path.with_suffix(".json")),
            "source_I_changed": False,
            "test_labels_opened": False,
        },
    )


def freeze(root, plan, stage):
    transfer = root / "transfer"
    destination = transfer / (stage + "_frozen.json")
    if destination.exists():
        raise RuntimeError("Target stage already frozen")
    paths = []
    if stage == "data":
        for family_id in range(len(plan["families"])):
            base = transfer / "data" / f"family_{family_id:03d}"
            paths.extend(
                (
                    base.with_suffix(".json"),
                    base.with_suffix(".npz"),
                    base.with_name(base.name + "_audit_only.npz"),
                )
            )
    else:
        for job_id in range(len(plan["families"]) * len(plan["noise"])):
            bank = transfer / "banks" / f"job_{job_id:03d}"
            fitted = transfer / "fits" / f"job_{job_id:03d}.json"
            paths.extend((bank.with_suffix(".json"), bank.with_suffix(".npz"), fitted))
            for relative in json.loads(fitted.read_text())["fit_artefacts"]:
                paths.extend(
                    (
                        (transfer / relative).with_suffix(".json"),
                        (transfer / relative).with_suffix(".npz"),
                    )
                )
    save_json(
        destination,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": {str(path.relative_to(transfer)): digest(path) for path in paths},
            "phase_protocol_sha256": digest(transfer / "protocol.json"),
            "test_labels_opened": False,
        },
    )


@torch.no_grad()
def audit(root, plan, device):
    transfer = root / "transfer"
    frozen = json.loads((transfer / "fit_frozen.json").read_text())
    for relative, expected in frozen["sha256"].items():
        if digest(transfer / relative) != expected:
            raise RuntimeError("Frozen target model changed")
    records, arrays = [], {}
    for job_id in range(len(plan["families"]) * len(plan["noise"])):
        path = transfer / "banks" / f"job_{job_id:03d}"
        bank = json.loads(path.with_suffix(".json").read_text())
        private_path = (
            transfer / "data" / f"family_{bank['family_id']:03d}_audit_only.npz"
        )
        observed_freeze = json.loads((transfer / "data_frozen.json").read_text())
        if (
            digest(private_path)
            != observed_freeze["sha256"][str(private_path.relative_to(transfer))]
        ):
            raise RuntimeError("Held-out target test data changed")
        with np.load(private_path) as private:
            x, y = [
                torch.tensor(private[name], device=device)
                for name in ("test_x", "test_clean")
            ]
        risks = []
        for number, candidate in enumerate(bank["candidates"]):
            fitted = (
                transfer
                / "fits"
                / f"job_{job_id:03d}"
                / (candidate["signature"] + ".npz")
            )
            with np.load(fitted) as data:
                index = data["index"]
                codes = torch.tensor(data["codes"], device=device)
            prediction = code_prediction(features(x, index, plan["codes"]), codes, plan)
            risk = (prediction - y).square().mean(-1).cpu().numpy()
            risks.append(risk)
            arrays[f"job_{job_id}__candidate_{number}__test_mse"] = risk
        reference = risks[bank["oracle_reference"]]
        for chosen in bank["records"] + [
            {
                "method": "oracle_true_I",
                "scope": "oracle",
                "complete": True,
                "candidate": bank["oracle_reference"],
            }
        ]:
            record = {
                "job_id": job_id,
                "family_id": bank["family_id"],
                "family": plan["families"][bank["family_id"]],
                "noise": bank["noise"],
                **chosen,
            }
            if chosen["complete"]:
                risk = risks[chosen["candidate"]]
                record.update(
                    {
                        "target_clean_mse": risk.tolist(),
                        "mean_target_clean_mse": float(risk.mean()),
                        "oracle_target_clean_mse": reference.tolist(),
                        "mean_excess_mse_over_true_I": float((risk - reference).mean()),
                        "ratio_to_true_I_mean_mse": float(
                            risk.mean() / max(float(reference.mean()), 1e-30)
                        ),
                    }
                )
            records.append(record)
    np.savez_compressed(transfer / "audit_only.npz", **arrays)
    save_json(
        transfer / "audit.json",
        {
            "records": records,
            "target_fit_freeze_sha256": digest(transfer / "fit_frozen.json"),
            "source_selection_freeze_sha256": digest(
                root / "common/selection_frozen.json"
            ),
            "scope": "Eight fresh target C per I, numbers reset and signed; sourceI fixed, trueI numeric oracle separate",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "register",
            "data",
            "freeze_data",
            "banks",
            "fit",
            "freeze_fit",
            "audit",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action == "register":
        register(args.root)
    elif args.action == "banks":
        banks(args.root, plan)
    elif args.action in ("freeze_data", "freeze_fit"):
        freeze(args.root, plan, args.action.removeprefix("freeze_"))
    elif args.action == "audit":
        if (args.root / "transfer/audit.json").exists():
            raise RuntimeError("Target audit already completed")
        audit(args.root, plan, args.device)
    else:
        length = (
            len(plan["families"])
            if args.action == "data"
            else len(plan["families"]) * len(plan["noise"])
        )
        for job_id in range(args.shard, length, args.shards):
            (prepare if args.action == "data" else fit_job)(
                args.root, plan, job_id, args.device
            )


if __name__ == "__main__":
    main()
