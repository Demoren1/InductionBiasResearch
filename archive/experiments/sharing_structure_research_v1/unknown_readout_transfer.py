"""Fresh C/W2 transfer after an unknown-readout structure cohort is frozen."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.darts_indexator_v1.data import digest, load_archive, save_json

from .joint_readout_control import PATTERN, feature, head_groups
from .pattern_prefix import timestamp
from .shared_structure_numeric import canonical_index
from .unknown_readout import teacher_output
from .weighted_prefix_search import fit_numeric, values


def add_candidate(index, codes, bank, seen):
    canonical = canonical_index(index, codes)
    key = canonical.tobytes()
    if key not in seen:
        seen[key] = len(bank)
        bank.append(canonical)
    return seen[key]


def prepare(root):
    source_frozen = json.loads((root / "source_frozen.json").read_text())
    for name, expected in source_frozen["sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Source choices changed")
    plan = json.loads((root / "protocol.json").read_text())
    output = root / "target"
    output.mkdir(exist_ok=False)
    for number, case in enumerate(plan["cases"]):
        rng = np.random.default_rng(plan["seed"] + 9000001 + number * 100003)
        teacher = load_archive(root / "private" / f"{case['name']}.npz")
        true_index = teacher["index"]
        bank, seen, origins = [], {}, {}

        for condition in range(len(plan["noise"])):
            selected = root / "source" / f"{case['name']}_{condition}" / "selected.npz"
            if selected.exists():
                origins[f"selected_{condition}"] = add_candidate(
                    load_archive(selected)["index"], case["codes"], bank, seen
                )
            origins[f"oracle_{condition}"] = add_candidate(
                true_index, case["codes"], bank, seen
            )
        random_index = rng.permutation(true_index.ravel()).reshape(true_index.shape)
        random_id = add_candidate(random_index, case["codes"], bank, seen)
        for condition in range(len(plan["noise"])):
            origins[f"random_{condition}"] = random_id
        tasks, hidden, outputs = (
            plan["target_functions"],
            case["hidden"],
            case["outputs"],
        )
        code = (
            rng.uniform(0.8, 1.3, (tasks, case["codes"]))
            if case["kind"] == "pattern"
            else rng.normal(size=(tasks, case["codes"]))
        )
        if case.get("variant") == "quartet_mix":
            code /= 2
        if case["kind"] == "sets":
            pooled = rng.uniform(0.4, 1.5, (tasks, 2, outputs)) * rng.choice(
                (-1, 1), (tasks, 2, outputs)
            )
            readout = pooled[:, head_groups(true_index)]
        else:
            readout = rng.uniform(0.5, 1.5, (tasks, hidden, outputs)) * rng.choice(
                (-1, 1), (tasks, hidden, outputs)
            )
        offset = rng.normal(0, 0.3, (tasks, outputs))
        private = {"codes": code, "readout": readout, "offset": offset}
        data = {"indices": np.stack(bank)}
        zero = teacher_output(
            np.zeros((1, case["inputs"])),
            true_index,
            code,
            readout,
            offset,
            case["bias"],
        )[:, 0]
        zero_noise = rng.normal(size=zero.shape)
        for condition, sigma in enumerate(plan["noise"]):
            data[f"zero_{condition}"] = zero + sigma * zero_noise / math.sqrt(
                plan["fit_points"] * plan["fit_repeats"]
            )
        for split, count in [
            ("support", plan["fit_points"]),
            ("validation", plan["fit_validation"]),
            ("test", plan["target_test"]),
        ]:
            x = rng.normal(size=(count, case["inputs"]))
            y = teacher_output(x, true_index, code, readout, offset, case["bias"])
            if split == "test":
                private.update(test_x=x, test_y=y)
            else:
                data[split + "_x"] = x
                noise = rng.normal(size=y.shape)
                for condition, sigma in enumerate(plan["noise"]):
                    data[f"{split}_y_{condition}"] = y + sigma * noise / math.sqrt(
                        plan["fit_repeats"]
                    )
        if case["kind"] == "sets":
            points = rng.normal(size=(64, 8, 2))
            permuted = np.stack(
                [[row[rng.permutation(8)] for _ in range(16)] for row in points]
            ).reshape(64, 16, 16)
            observed = np.empty_like(permuted)
            observed[..., teacher["permutation"]] = permuted
            private["permutation_x"] = observed.reshape(-1, 16)
        np.savez_compressed(output / f"data_{case['name']}.npz", **data)
        np.savez_compressed(root / "private" / f"target_{case['name']}.npz", **private)
        save_json(output / f"origins_{case['name']}.json", origins)
    save_json(
        output / "data_frozen.json",
        {
            "frozen_utc": timestamp(),
            "source_freeze_sha256": digest(root / "source_frozen.json"),
            "sha256": {p.name: digest(p) for p in output.glob("data_*.npz")},
            "generator_sha256": digest(Path(__file__)),
        },
    )
    if any(case["kind"] == "pattern" for case in plan["cases"]):
        prepare_pattern(root, plan)
    print(json.dumps({"targets_prepared": len(plan["cases"])}), flush=True)


def prepare_pattern(root, plan):
    folder = root / "target_pattern"
    folder.mkdir(exist_ok=False)
    old = json.loads(
        (PATTERN.parent / "../20261106/trainable_readout_g28/protocol.json")
        .resolve()
        .read_text()
    )
    config = {
        "seed": plan["seed"] + 880001,
        "pattern": old["pattern"],
        "source_freeze_sha256": digest(root / "source_frozen.json"),
        "scope": "Fresh binary C/b/W2/d fit after unknown-readout I search. All selected Pattern I must be identical for deduplication.",
    }
    indices = [
        load_archive(root / "source" / f"{c['name']}_{j}" / "selected.npz")["index"]
        for c in plan["cases"]
        if c["kind"] == "pattern"
        for j in range(2)
    ]
    keys = {canonical_index(index, 2).tobytes() for index in indices}
    if len(keys) != 1:
        raise ValueError("Distinct selected Pattern I require separate binary cohorts")
    index = canonical_index(indices[0], 2)
    rng = np.random.default_rng(config["seed"])
    random_index = rng.permutation(index.ravel()).reshape(index.shape)
    data = load_archive(PATTERN / "data/binary_train.npz")
    np.savez_compressed(
        folder / "pattern_data.npz", index=index, random_index=random_index, **data
    )
    save_json(folder / "protocol.json", config)


def fit(root, case_number, device):
    plan = json.loads((root / "protocol.json").read_text())
    case = plan["cases"][case_number]
    output = root / "target"
    data = load_archive(output / f"data_{case['name']}.npz")
    origins = json.loads((output / f"origins_{case['name']}.json").read_text())
    fitted, records = {}, []
    for condition in range(len(plan["noise"])):
        used = sorted(
            {
                candidate
                for key, candidate in origins.items()
                if key.endswith("_" + str(condition))
            }
        )
        anchor = data[f"zero_{condition}"]
        for candidate in used:
            parameters, trace, history = fit_numeric(
                data["indices"][candidate],
                data["support_x"],
                data[f"support_y_{condition}"] - anchor[:, None],
                data["validation_x"],
                data[f"validation_y_{condition}"] - anchor[:, None],
                {**plan, **case},
                plan["seed"] + case_number * 73 + 101,
                device,
            )
            key = f"fit_{condition}_{candidate}"
            fitted[key + "_parameters"] = parameters
            fitted.update({key + "_" + k: v for k, v in history.items()})
            records.append({"condition": condition, "candidate": candidate, **trace})
            print(
                json.dumps(
                    {
                        "target": case["name"],
                        "condition": condition,
                        "candidate": candidate,
                        "fit": trace,
                    }
                ),
                flush=True,
            )
    np.savez_compressed(output / f"fits_{case['name']}.npz", **fitted)
    save_json(
        output / f"frozen_{case['name']}.json",
        {
            "frozen_utc": timestamp(),
            "fits": records,
            "sha256": digest(output / f"fits_{case['name']}.npz"),
        },
    )


@torch.no_grad()
def audit(root, device):
    plan = json.loads((root / "protocol.json").read_text())
    output = root / "target"
    freezes = [output / f"frozen_{case['name']}.json" for case in plan["cases"]]
    for path in freezes:
        row = json.loads(path.read_text())
        if (
            digest(
                output / path.name.replace("frozen_", "fits_").replace(".json", ".npz")
            )
            != row["sha256"]
        ):
            raise ValueError("Target predictors changed")
    pattern = root / "target_pattern"
    if pattern.exists():
        freezes.append(pattern / "pattern_frozen.json")
        json.loads(freezes[-1].read_text())
    save_json(
        output / "cohort_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {str(p.relative_to(root)): digest(p) for p in freezes},
        },
    )
    rows = []
    for case in plan["cases"]:
        data = load_archive(output / f"data_{case['name']}.npz")
        private = load_archive(root / "private" / f"target_{case['name']}.npz")
        fitted = load_archive(output / f"fits_{case['name']}.npz")
        origins = json.loads((output / f"origins_{case['name']}.json").read_text())
        for condition, noise in enumerate(plan["noise"]):
            for method in ("selected", "oracle", "random"):
                origin = f"{method}_{condition}"
                if origin not in origins:
                    continue
                candidate = origins[origin]
                index = data["indices"][candidate]
                p = torch.tensor(
                    fitted[f"fit_{condition}_{candidate}_parameters"], device=device
                )
                prediction = (
                    values(
                        feature(
                            private["test_x"],
                            index,
                            case["codes"],
                            device,
                            torch.float64,
                        ),
                        p,
                        {**plan, **case},
                        case["outputs"],
                    )
                    .cpu()
                    .numpy()
                )
                prediction += data[f"zero_{condition}"][:, None]
                mse = ((prediction - private["test_y"]) ** 2).mean((1, 2))
                variance = private["test_y"].var(axis=1).mean(1)
                nmse = mse / variance
                perm = np.full(len(p), np.nan)
                if "permutation_x" in private:
                    pred = (
                        values(
                            feature(
                                private["permutation_x"],
                                index,
                                case["codes"],
                                device,
                                torch.float64,
                            ),
                            p,
                            {**plan, **case},
                            case["outputs"],
                        )
                        .cpu()
                        .numpy()
                        .reshape(len(p), 64, 16, -1)
                    )
                    perm = np.sqrt(
                        ((pred - pred.mean(2, keepdims=True)) ** 2).mean((1, 2, 3))
                    ) / np.sqrt(variance)
                for task in range(len(p)):
                    rows.append(
                        {
                            "case": case["name"],
                            "kind": case["kind"],
                            "noise": noise,
                            "method": method,
                            "target": task,
                            "mse": float(mse[task]),
                            "nmse": float(nmse[task]),
                            "permutation_normalized_rms": float(perm[task])
                            if np.isfinite(perm[task])
                            else None,
                        }
                    )
    result = {"evaluated_utc": timestamp(), "rows": rows}
    if pattern.exists():
        from .joint_readout_report import evaluate_pattern

        summary, detailed = evaluate_pattern(pattern, device)
        result["pattern_binary"] = summary
        save_json(pattern / "evaluation.json", {"summary": summary, "rows": detailed})
    save_json(output / "evaluation.json", result)
    for kind in sorted({r["kind"] for r in rows}):
        for noise in plan["noise"]:
            selected = [
                r
                for r in rows
                if r["kind"] == kind
                and r["noise"] == noise
                and r["method"] == "selected"
            ]
            print(
                json.dumps(
                    {
                        "kind": kind,
                        "noise": noise,
                        "tasks": len(selected),
                        "mean_nmse": np.mean([r["nmse"] for r in selected]),
                        "worst_nmse": max(r["nmse"] for r in selected),
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "fit", "audit", "pattern"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "prepare":
        prepare(args.root.resolve())
    elif args.action == "fit":
        fit(args.root.resolve(), args.case, args.device)
    elif args.action == "audit":
        audit(args.root.resolve(), args.device)
    else:
        from .joint_readout_control import pattern_fit

        pattern_fit(args.root.resolve() / "target_pattern", args.device)


if __name__ == "__main__":
    main()
