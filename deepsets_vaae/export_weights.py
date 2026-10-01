"""Recreate target fits and export their restored numerical weights.

The original pilot deliberately stored masks and scalar records only.  This
command replays its deterministic target protocol against those frozen masks,
then writes checkpoint tensors for plotting without modifying the pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

from .core import evaluate_masks, load_data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def _record_key(record: dict) -> tuple[int, int, str, int]:
    return (int(record["task"]), int(record["support_size"]),
            str(record["method"]), int(record["init"]))


def _source_snapshot_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        return {}
    return {path.name: _sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}


def _save_export_snapshot(output: Path) -> dict[str, str]:
    """Preserve the two files whose code produces this export."""
    snapshot = output / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    sources = {"export_weights.py": Path(__file__), "core.py": Path(__file__).with_name("core.py")}
    for name, source in sources.items():
        shutil.copy2(source, snapshot / name)
    return {name: _sha256(source) for name, source in sources.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-out", type=Path, required=True,
                        help="completed original seed directory, e.g. .../seed_4100")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--all-conditions", action="store_true",
                        help="reproduce all eight target tasks and all four labelled budgets")
    args = parser.parse_args()

    original = args.original_out.resolve()
    output = args.out.resolve()
    protocol_path = original / "protocol.json"
    results_path = original / "results.json"
    masks_path = original / "masks.pt"
    for path in (protocol_path, results_path, masks_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing original pilot input: {path}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"export output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(protocol_path.read_text())
    original_results = json.loads(results_path.read_text())
    if int(protocol["seed"]) != args.seed or int(original_results["seed"]) != args.seed:
        raise ValueError("--seed must match the original protocol and results")
    if protocol.get("smoke"):
        raise ValueError("the weight exporter only supports the full pilot protocol")
    target_device = torch.device(args.device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA exporter requested but CUDA is unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    all_sizes = tuple(int(value) for value in protocol["support_sizes"])
    support_sizes = all_sizes if args.all_conditions else (max(all_sizes),)
    test_task_count = int(protocol["test_tasks"]) if args.all_conditions else 1
    masks = torch.load(masks_path, map_location=target_device, weights_only=True)
    data = load_data(args.data_dir, args.seed, target_device,
                     per_digit_train=1000, per_digit_validation=300, per_digit_test=300)
    task_costs = torch.tensor(protocol["task_vectors"]["test"][:test_task_count],
                              dtype=torch.float32, device=target_device)
    records = evaluate_masks(
        data, task_costs, masks, args.seed + 100000, target_device,
        support_sizes=support_sizes, steps=int(protocol["eval_steps"]), batch_size=32,
        set_size=int(protocol["set_size"]), validation_sets=int(protocol["validation_sets"]),
        test_sets=int(protocol["test_sets"]), artifact_dir=output)

    requested = {(task, budget) for task in range(test_task_count) for budget in support_sizes}
    copied_records = [record for record in original_results["records"]
                      if (int(record["task"]), int(record["support_size"])) in requested]
    expected = {(task, budget, method, init)
                for task, budget in requested
                for method in masks
                for init in range(len(masks[method]))}
    original_by_key = {_record_key(record): record for record in copied_records}
    reproduced_by_key = {_record_key(record): record for record in records}
    if set(original_by_key) != expected or set(reproduced_by_key) != expected:
        raise AssertionError("the selected record keys do not match the frozen mask protocol")
    deltas = {"/".join(map(str, key)): abs(float(reproduced_by_key[key]["mse"])
                                              - float(original_by_key[key]["mse"]))
              for key in sorted(expected)}
    maximum_delta = max(deltas.values(), default=0.0)

    _write_json(output / "records.json", {
        "schema": "deepsets_vaae.weight_export_records.v1",
        "original_records": copied_records,
        "reproduced_records": records,
        "mse_absolute_deltas": deltas,
        "max_absolute_mse_delta": maximum_delta,
    })
    export_hashes = _save_export_snapshot(output)
    provenance = {
        "schema": "deepsets_vaae.weight_export_provenance.v1",
        "original": {
            "directory": str(original),
            "protocol_sha256": _sha256(protocol_path),
            "results_sha256": _sha256(results_path),
            "masks_sha256": _sha256(masks_path),
            "protocol_source_sha256": protocol.get("source_sha256", {}),
            "source_snapshot_sha256": _source_snapshot_hashes(original.parent / "source_snapshot"),
        },
        "export": {
            "directory": str(output),
            "source_sha256": export_hashes,
            "device": str(target_device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "conditions": {"all_conditions": args.all_conditions, "tasks": list(range(test_task_count)),
                       "support_sizes": list(support_sizes), "records": len(records)},
        "max_absolute_mse_delta": maximum_delta,
        "mse_delta_threshold": 1e-4,
        "passed": maximum_delta <= 1e-4,
    }
    _write_json(output / "export_provenance.json", provenance)
    if maximum_delta > 1e-4:
        raise AssertionError(f"reproduced MSE differs from original by {maximum_delta:.9g} (> 1e-4)")
    print(json.dumps({"records": len(records), "max_absolute_mse_delta": maximum_delta,
                      "out": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
