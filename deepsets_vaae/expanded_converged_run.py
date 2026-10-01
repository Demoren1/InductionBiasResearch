"""Convergence-based VAE follow-up on the cached expanded source banks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from .core import load_data
from .followup_batched_eval import evaluate_masks_batched
from .followup_common import configure, run_cuda_queue
from .run import write_json

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "outputs/deepsets_vaae/20261001_expanded_functional_vae"
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_converged_functional_vae"
SEEDS = list(range(4100, 4108))
METHODS = ["functional_mean_small", "functional_mean_large", "functional_vae_small",
           "functional_vae_large", "raw_vae_large", "random", "dense"]
SOURCE_FILES = ["expanded_converged_run.py", "expanded_converged_vae.py",
                "expanded_functional_vae.py", "masks.py", "core.py",
                "followup_importance.py", "followup_batched_eval.py",
                "followup_common.py", "run.py"]
BUDGETS = [32, 64, 128, 256]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _input_manifest() -> dict[str, str]:
    paths = [INPUT_ROOT / f"seed_{seed}" / f"bank_{task}.pt"
             for seed in SEEDS for task in range(4)]
    paths += [INPUT_ROOT / f"seed_{seed}" / "functional/functional_vae_arrays.npz"
              for seed in SEEDS]
    paths += [INPUT_ROOT / f"seed_{seed}" / "masks.pt" for seed in SEEDS]
    paths += [INPUT_ROOT / f"seed_{seed}" / "functional/masks.pt" for seed in SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing cached expanded inputs: {missing[:5]}")
    manifest = {path.relative_to(INPUT_ROOT).as_posix(): _sha(path) for path in paths}
    for seed in SEEDS:
        root_masks = torch.load(INPUT_ROOT / f"seed_{seed}/masks.pt", map_location="cpu", weights_only=True)
        functional_masks = torch.load(INPUT_ROOT / f"seed_{seed}/functional/masks.pt", map_location="cpu", weights_only=True)
        if set(root_masks) != set(functional_masks) or any(
                not torch.equal(root_masks[name], functional_masks[name]) for name in root_masks):
            raise ValueError(f"cached mask copies differ for seed {seed}")
    return manifest


def protocol() -> dict:
    old_path = INPUT_ROOT / "protocol.json"
    old = _json(old_path)
    vectors = old["task_vectors"]
    if (old.get("candidates"), old.get("keep"), old.get("source_tasks")) != (1024, 256, 4):
        raise ValueError("cached input protocol is not the expected 4 × 1024/256 expanded bank")
    return {
        "experiment": "Convergence-based repeat of expanded functional-map VAE",
        "date_moscow": "2026-10-01", "seeds": SEEDS,
        "input_root": str(INPUT_ROOT),
        "original_protocol_sha256": _sha(old_path),
        "input_artifact_sha256": _input_manifest(),
        "task_vectors": vectors,
        "source_tasks": 4, "source_candidates": 1024, "source_keep": 256,
        "source_density": 0.2, "source_train_images_per_digit": 1000,
        "source_validation_images_per_digit": 300,
        "target_density": 0.3, "target_edges": 7526,
        "methods": METHODS, "support_sizes": BUDGETS,
        "vae_families": {
            "functional_vae_small": {"train_maps": 26, "input": "function_maps"},
            "functional_vae_large": {"train_maps": 205, "input": "function_maps"},
            "raw_vae_large": {"train_maps": 205, "input": "raw_maps"},
        },
        "vae_validation_maps": 51, "vae_latent": 16, "vae_width": 128,
        "vae_objective": "sum BCE-with-logits per map + 0.1 * KL, averaged over maps",
        "convergence": {"minimum_epochs": 1000, "maximum_epochs": 20000,
                        "window_epochs": 200, "plateau_tolerance": 0.001,
                        "patience_epochs": 400,
                        "validation_improvement_tolerance": 0.0001,
                        "requires_train_and_validation_plateau": True,
                        "max_epochs_is_not_convergence": True},
        "target_eval": {"steps": 800, "batch_size": 32, "set_size": 5,
                        "validation_sets": 128, "test_sets": 512,
                        "batch_conditions": 8, "kernel_mode": "reference",
                        "initialization_reference_models": 20,
                        "gradient_denominator": 20,
                        "old_seed_offset": 100000, "fresh_seed_offset": 300000},
        "comparison_status": {
            "original_predeclared_primary": old.get("primary_comparison"),
            "classification": "all comparisons in this convergence repeat are exploratory",
            "reason": "the same fresh tasks were already inspected in the original 160-update study",
            "confirmatory_claim": False,
        },
        "old_source_files_sha256": old.get("source_sha256", {}),
    }


def freeze_sources(out: Path, spec: dict) -> None:
    snapshot = out / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in SOURCE_FILES:
        source = ROOT / "deepsets_vaae" / name
        content = source.read_bytes()
        target = snapshot / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Frozen source differs: {name}")
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    spec["source_sha256"] = hashes
    path = out / "protocol.json"
    if path.exists() and _json(path) != spec:
        raise ValueError("Existing protocol differs; use a new output")
    write_json(path, spec)


def _verify_protocol(out: Path) -> dict:
    spec = _json(out.parent / "protocol.json")
    if _sha(INPUT_ROOT / "protocol.json") != spec["original_protocol_sha256"]:
        raise ValueError("cached input protocol changed after launch")
    for name, expected in spec["source_sha256"].items():
        if _sha(ROOT / "deepsets_vaae" / name) != expected:
            raise ValueError(f"production source changed: {name}")
        snap = out.parent / "source_snapshot" / name
        if _sha(snap) != expected:
            raise ValueError(f"frozen source snapshot changed: {name}")
    return spec


def _verify_input_seed(seed: int, spec: dict) -> tuple[Path, dict]:
    folder = INPUT_ROOT / f"seed_{seed}"
    for marker in ("BANK_COMPLETE", "COMPLETE"):
        if not (folder / marker).is_file():
            raise ValueError(f"cached source seed is incomplete: {folder / marker}")
    old = _json(INPUT_ROOT / "protocol.json")
    seed_spec = _json(folder / "protocol.json")
    if int(seed_spec.get("seed", -1)) != seed:
        raise ValueError(f"cached source seed protocol mismatch: {folder}")
    if {k: v for k, v in seed_spec.items() if k not in ("seed", "cuda_visible_devices")} != old:
        raise ValueError(f"cached seed protocol differs from original root protocol: {folder}")
    manifest = _json(folder / "bank_artifact_hashes.json")
    for task in range(4):
        name = f"bank_{task}.pt"
        expected = spec["input_artifact_sha256"][f"seed_{seed}/{name}"]
        if manifest.get(name) != expected or _sha(folder / name) != expected:
            raise ValueError(f"cached bank hash mismatch: {folder / name}")
    for relative in (f"seed_{seed}/functional/functional_vae_arrays.npz", f"seed_{seed}/masks.pt",
                     f"seed_{seed}/functional/masks.pt"):
        if _sha(INPUT_ROOT / relative) != spec["input_artifact_sha256"][relative]:
            raise ValueError(f"cached functional input hash mismatch: {relative}")
    return folder, _json(folder / "data_provenance.json")


def _prepare_seed_output(out: Path, seed: int, source: Path, old_provenance: dict,
                         current_data: dict, spec: dict) -> None:
    if old_provenance.get("row_ids_pairwise_disjoint") is not True:
        raise ValueError("cached source data provenance does not establish disjoint splits")
    if old_provenance.get("split_hashes") != current_data.get("split_hashes"):
        raise ValueError("current MNIST8m splits differ from the cached source data")
    out.mkdir(parents=True, exist_ok=True)
    seed_protocol = {**spec, "seed": seed,
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    p = out / "protocol.json"
    if p.exists() and _json(p) != seed_protocol:
        raise ValueError(f"existing seed protocol differs: {p}")
    write_json(p, seed_protocol)
    for task in range(4):
        src, dst = source / f"bank_{task}.pt", out / f"bank_{task}.pt"
        if dst.exists() or dst.is_symlink():
            if not dst.is_symlink() or dst.resolve() != src.resolve():
                raise ValueError(f"source bank link differs: {dst}")
        else:
            dst.symlink_to(src.resolve())
    data_path = out / "data_provenance.json"
    if data_path.exists() and _json(data_path) != old_provenance:
        raise ValueError(f"existing data provenance differs: {data_path}")
    shutil.copyfile(source / "data_provenance.json", data_path)


def _validate_masks(masks: dict) -> None:
    if list(masks) != METHODS:
        raise ValueError(f"unexpected mask methods/order: {list(masks)}")
    for name, value in masks.items():
        mask = torch.as_tensor(value)
        if tuple(mask.shape) != (4, 784, 32) or not bool(torch.all((mask == 0) | (mask == 1))):
            raise ValueError(f"invalid mask shape or values for {name}")
        expected = 784 * 32 if name == "dense" else 7526
        if not bool(torch.all(mask.sum((-1, -2)) == expected)):
            raise ValueError(f"invalid mask edge count for {name}")


def _validate_records(records: list[dict], block: str) -> None:
    expected = {(t, b, method, init) for t in range(8) for b in BUDGETS
                for method in METHODS for init in range(4)}
    seen = set()
    for row in records:
        key = (int(row["task"]), int(row["support_size"]), str(row["method"]), int(row["init"]))
        if key in seen or not torch.isfinite(torch.tensor(float(row["mse"]))):
            raise ValueError(f"duplicate or invalid {block} target record: {key}")
        seen.add(key)
    if seen != expected:
        raise ValueError(f"{block} target coverage mismatch: {len(seen)} records, expected {len(expected)}")


def worker(out: Path, seed: int) -> None:
    configure(seed)
    spec = _verify_protocol(out)
    started = time.monotonic()
    last_stage = "starting"
    def status(stage: str, **details) -> None:
        nonlocal last_stage
        last_stage = stage
        row = {"seed": seed, "stage": stage, "elapsed_seconds": time.monotonic() - started, **details}
        write_json(out / "status.json", row)
        print(json.dumps(row), flush=True)

    if (out / "COMPLETE").is_file():
        result = _json(out / "results.json")
        diag = _json(out / "functional/functional_vae_diagnostics.json")
        if diag.get("converged") is True and len(result.get("records", [])) == 896 and len(result.get("fresh_records", [])) == 896:
            return
        raise ValueError(f"invalid existing COMPLETE output: {out}")
    try:
        source, provenance = _verify_input_seed(seed, spec)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        status("loading_target_data")
        data = load_data(ROOT / "datasets/mnist8m", seed, device,
                         per_digit_train=1000, per_digit_validation=300, per_digit_test=300)
        _prepare_seed_output(out, seed, source, provenance, data, spec)

        from .expanded_converged_vae import extract_converged_functional_masks
        functional_out = out / "functional"
        diagnostics_path = functional_out / "functional_vae_diagnostics.json"
        cached_masks_path = functional_out / "masks.pt"
        if diagnostics_path.is_file() and cached_masks_path.is_file() \
                and _json(diagnostics_path).get("converged") is True:
            diagnostics = _json(diagnostics_path)
            masks = torch.load(cached_masks_path, map_location="cpu", weights_only=True)
        else:
            status("functional_vae_convergence", minimum_epochs=1000, maximum_epochs=20000)
            masks, diagnostics = extract_converged_functional_masks(
                source / "functional", functional_out, seed, device)
        if diagnostics.get("converged") is not True:
            status("convergence_not_reached", diagnostics_path="functional/functional_vae_diagnostics.json",
                   converged=False)
            raise RuntimeError("VAE fit budget ended before all 12 fits converged; no COMPLETE marker written")
        _validate_masks(masks)
        tmp = out / "masks.pt.tmp"
        torch.save({name: torch.as_tensor(value).detach().cpu() for name, value in masks.items()}, tmp)
        os.replace(tmp, out / "masks.pt")

        eval_kw = dict(support_sizes=tuple(BUDGETS), steps=800, batch_size=32, set_size=5,
                       validation_sets=128, test_sets=512, batch_conditions=8,
                       kernel_mode="reference", initialization_reference_models=20)
        old_tasks = spec["task_vectors"]["test"]
        fresh_tasks = spec["task_vectors"]["fresh_test"]
        status("target_old")
        records = evaluate_masks_batched(data, old_tasks, masks, seed + 100000, device,
                                         artifact_dir=out / "weights", **eval_kw)
        status("target_fresh")
        fresh_records = evaluate_masks_batched(data, fresh_tasks, masks, seed + 300000, device,
                                               artifact_dir=out / "fresh_weights", **eval_kw)
        _validate_records(records, "old")
        _validate_records(fresh_records, "fresh")
        write_json(out / "results.json", {
            "seed": seed, "records": records, "fresh_records": fresh_records,
            "diagnostics": diagnostics, "comparison_status": spec["comparison_status"],
            "elapsed_seconds": time.monotonic() - started,
        })
        status("complete", records=len(records), fresh_records=len(fresh_records), converged=True)
        (out / "COMPLETE").write_text("complete\n")
    except Exception as exc:
        if last_stage == "convergence_not_reached":
            status("convergence_not_reached", error=f"{type(exc).__name__}: {exc}",
                   diagnostics_path="functional/functional_vae_diagnostics.json", converged=False)
        else:
            status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        spec = protocol()
        freeze_sources(args.out, spec)
        gpu_path = INPUT_ROOT / "gpu_selection.json"
        prior = _json(gpu_path)
        selected = prior.get("selected", [])
        if len(selected) != 8 or len(set(selected)) != 8:
            raise ValueError("original expanded run does not record eight unique GPU UUIDs")
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
                                 "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
        rows = [line.split(",") for line in result.stdout.strip().splitlines()]
        current = {r[1].strip(): int(r[2].strip()) for r in rows}
        if any(uuid not in current or current[uuid] != 0 for uuid in selected):
            raise RuntimeError("an originally assigned GPU is missing or not utilization-idle")
        write_json(args.out / "gpu_selection.json", {
            "original_selection_sha256": _sha(gpu_path), "selected": selected,
            "original_observation": prior.get("observation"), "current_observation": result.stdout,
            "memory_considered": False,
        })
        run_cuda_queue("deepsets_vaae.expanded_converged_run", args.out, selected,
                       seeds=SEEDS, workers_per_gpu=1,
                       env_overrides={"DEEPSETS_EVAL_BATCH_CONDITIONS": "8"})
        (args.out / "COMPLETE").write_text("all8 complete\n")
    else:
        if args.seed is None:
            parser.error("--seed required for a worker")
        worker(args.out, args.seed)


if __name__ == "__main__":
    main()
