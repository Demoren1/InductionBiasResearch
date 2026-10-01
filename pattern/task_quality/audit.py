"""Independent recomputation of saved scientific states and test metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .core import build_experiment_data, build_test_pool
from .evaluate import EXPECTED_OUTER_SEEDS, score_children_batched, validate_eval_record
from .meta import child_logits_batch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plateau_numpy(values: np.ndarray) -> bool:
    """Recompute the empirical criterion independently of the fitter."""
    if len(values) < 16 or not np.isfinite(values[-16:]).all():
        return False
    before, after = np.asarray(values[-16:], dtype=np.float64).reshape(2, 8)
    denominator = max(abs(before.mean()), abs(after.mean()), 0.01)
    change = abs(after.mean() - before.mean()) / denominator
    slope = np.polyfit(np.arange(8), after, 1)[0]
    return bool(change <= 0.01 and abs(slope) * 8 / denominator <= 0.01)


def audit_meta(root: Path) -> dict:
    summaries = []
    for seed in EXPECTED_OUTER_SEEDS:
        for method in ("transformer_mask", "free_mask"):
            directory = root / f"seed_{seed}/{method}/meta"
            last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
            assert last["converged"] and last["stop_reason"] == "empirical_train_val_outer_query_plateau"
            assert last["step"] >= last["config"]["min_steps"]
            for field in ("train_monitor_query_bce", "val_query_bce"):
                curve = torch.as_tensor(last["curves"][field]).numpy()
                assert all(plateau_numpy(curve[:len(curve) - offset]) for offset in range(3))
            best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
            assert float(last["curves"]["val_query_bce"].min()) == last["best_val"]
            assert best["step"] == last["best_step"]
            if method == "transformer_mask":
                assert last["bank_sha256"] == sha256(root / f"seed_{seed}/bank/bank.pt")
            summaries.append({"seed": seed, "method": method,
                              "steps": last["step"], "best_step": best["step"]})
    return {"status": "PASS", "fits": summaries,
            "checks": ["independent NumPy plateau for three consecutive train and val checks",
                       "best validation checkpoint and frozen source bank hash"]}


def audit_banks(root: Path) -> dict:
    summaries = []
    for seed in EXPECTED_OUTER_SEEDS:
        path = root / f"seed_{seed}" / "bank" / "bank.pt"
        bank = torch.load(path, map_location="cpu", weights_only=False)
        assert bank["feature"].shape == (960, 8, 187)
        assert torch.isfinite(bank["feature"]).all()
        assert torch.equal(bank["weff"], bank["w"] * bank["masks"])
        assert ((bank["masks"] == 0) | (bank["masks"] == 1)).all()
        assert torch.equal(bank["masks"].sum((1, 2)).to(torch.int64), bank["density"])
        data = build_experiment_data(probe_seed=seed)
        source_tasks = {task.pattern for task in data["splits"]["train"]}
        assert set(bank["source_pattern"]) == source_tasks
        assert torch.equal(bank["probe_ids"], data["probe"]["ids"])
        x = data["probe"]["x"]
        pre = torch.einsum("pi,mih->pmh", x, bank["weff"]) + bank["b"][None]
        psi = (torch.relu(pre) * bank["a"][None]).permute(1, 0, 2)
        assert torch.allclose(psi, bank["probe_psi"], rtol=1e-5, atol=2e-6)
        q = (x[:, None, :, None] * bank["weff"][None]
             * bank["a"][None, :, None, :] * (pre > 0)[:, :, None, :])
        assert torch.allclose(q.mean(0), bank["q_signed"], rtol=1e-5, atol=2e-6)
        assert torch.allclose(q.abs().mean(0), bank["q_abs"], rtol=1e-5, atol=2e-6)
        assert torch.allclose(q.var(0, unbiased=False), bank["q_variance"], rtol=1e-5, atol=2e-6)
        packed = torch.cat([
            psi.permute(0, 2, 1), q.mean(0).permute(0, 2, 1),
            q.abs().mean(0).permute(0, 2, 1), q.var(0, unbiased=False).permute(0, 2, 1),
            bank["masks"].permute(0, 2, 1), bank["weff"].permute(0, 2, 1),
            bank["b"][..., None], bank["a"][..., None],
            (bank["density"].float() / 88)[:, None, None].expand(-1, 8, 1),
            bank["quality"][:, None, None].expand(-1, 8, 1),
        ], dim=-1)
        assert torch.allclose(packed, bank["raw_feature"], rtol=1e-5, atol=2e-6)
        normalized = (bank["raw_feature"] - bank["feature_mean"]) / bank["feature_std"]
        assert torch.equal(normalized, bank["feature"])
        remaining = 0
        for pattern in sorted(source_tasks):
            teacher = torch.load(root / f"seed_{seed}/bank/teachers/pattern_{pattern}.pt",
                                 map_location="cpu", weights_only=False)
            indices = torch.tensor([i for i, name in enumerate(bank["source_pattern"])
                                    if name == pattern])
            selected = bank["selected_index"][indices]
            for key, teacher_key in (("w", "w1"), ("b", "b1"), ("a", "w2"), ("c", "b2"),
                                     ("masks", "masks")):
                assert torch.equal(bank[key][indices], teacher["best"][teacher_key][selected])
            assert torch.equal(bank["quality"][indices],
                               teacher["best_query_balanced_bce"][selected])
            remaining += int((~teacher["converged_per_model"][selected]).sum())
        summaries.append({"seed": seed, "bank_sha256": sha256(path), "maps": 960,
                          "selected_unconverged": remaining, "step_cap": bank["config"]["max_steps"]})
    return {"status": "PASS", "banks": summaries,
            "checks": ["exact selected full teacher states", "source-only task/probe membership",
                       "binary masks and exact densities", "recomputed ReLU functional features",
                       "complete packed 187-dimensional feature layout and normalization"]}


def audit_records(root: Path) -> dict:
    records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines() if line]
    assert len(records) == 768
    for row in records:
        validate_eval_record(row)
        child_path = Path(row["child_checkpoint"])
        assert sha256(child_path) == row["child_checkpoint_sha256"]
        checkpoint = torch.load(child_path, map_location="cpu", weights_only=False)
        for field, key in (("weight", "w"), ("bias", "b"), ("readout", "a")):
            assert torch.equal(torch.tensor(row[field], dtype=torch.float32), checkpoint["best_params"][key])
        assert torch.equal(torch.tensor(row["mask"], dtype=torch.uint8), checkpoint["mask"])
        assert float(checkpoint["best_params"]["c"]) == row["output_bias"]
        assert checkpoint["fit_status"] == row["fit_status"]
        assert row["best_step"] <= row["fit_status"]["steps"]
        if not row["fit_status"]["converged"]:
            assert row["fit_status"]["stop_reason"] == "step_cap"
            assert row["fit_status"]["steps"] == row["fit_status"]["max_steps"]
    for seed in EXPECTED_OUTER_SEEDS:
        for budget in (32, 128):
            directory = root / f"seed_{seed}/eval"
            frozen = torch.load(directory / f"test_budget_{budget}.frozen.pt",
                                map_location="cpu", weights_only=False)
            assert sha256(Path(frozen["batch_checkpoint"])) == frozen["batch_checkpoint_sha256"]
            assert frozen["frozen_before_test"] and not frozen["test_labels_materialized"]
            for index, status in enumerate(frozen["fit_status"]):
                history = [entry for entry in frozen["history"] if int(entry["step"]) <= status["steps"]]
                scores = np.asarray([float(entry["query_balanced_bce"][index]) for entry in history])
                assert abs(scores.min() - float(frozen["best_query_balanced_bce"][index])) < 1e-6
                if status["converged"]:
                    for field in ("support_balanced_bce", "query_balanced_bce"):
                        curve = np.asarray([float(entry[field][index]) for entry in history])
                        assert all(plateau_numpy(curve[:len(curve) - offset]) for offset in range(3))
    maximum_delta = 0.0
    # Recompute all scores from the raw saved weights, rather than from summaries.
    for start in range(0, len(records), 96):
        group = records[start:start + 96]
        pools = [build_test_pool(row["task_id"].split(":")[1]) for row in group]
        for row, pool in zip(group, pools):
            assert pool["ids"].tolist() == row["test_ids"]
        params = {key: torch.stack([torch.as_tensor(row[field], dtype=torch.float32) for row in group])
                  for key, field in (("w", "weight"), ("b", "bias"), ("a", "readout"), ("c", "output_bias"))}
        masks = torch.tensor(np.asarray([row["mask"] for row in group]), dtype=torch.float32)
        scores = score_children_batched(params, masks,
                                        {"x": torch.stack([p["x"] for p in pools]),
                                         "y": torch.stack([p["y"] for p in pools])},
                                        predictor=child_logits_batch, device="cpu")
        for row, score in zip(group, scores):
            for key in ("accuracy", "bce", "brier", "balanced_accuracy", "balanced_bce"):
                delta = abs(float(score[key]) - float(row["test"][key]))
                maximum_delta = max(maximum_delta, delta)
                assert delta < 3e-5, (row["method"], key, delta)
    return {"status": "PASS", "records_recomputed": len(records),
            "maximum_CPU_vs_saved_metric_delta": maximum_delta,
            "checks": ["checkpoint hashes and exact full states", "entire test partition",
                       "test metrics recomputed from saved W, M, biases and readout"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--banks-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"source_banks": audit_banks(args.root)}
    if not args.banks_only:
        result["meta_fits"] = audit_meta(args.root)
        result["final_records"] = audit_records(args.root)
    target = args.root / ("source_audit.json" if args.banks_only else "independent_audit.json")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
