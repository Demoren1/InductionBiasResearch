"""Choose a global numerical learning rule on held-out development tasks."""

import argparse
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from .adaptation import fit_adam, signed_basis
from ..protocol.real_boundary import StudyBoundary


def choose_source(root, method, grid, device):
    if method in ("mask_only", "mask_random"):
        origin = root / "validation_fit" / "current"
        mask = np.load(origin / "model.npz")["index"] != 0
        if method == "mask_random":
            rng = np.random.default_rng(777001)
            mask = np.stack(
                [rng.permutation(mask[:, h]) for h in range(mask.shape[1])], -1
            )
        index = np.zeros(mask.shape, dtype=np.int16)
        index[mask] = np.arange(1, mask.sum() + 1)
        return (
            None,
            index,
            {
                "codes": int(mask.sum()),
                "mask_only": True,
                "random_mask": method == "mask_random",
                "matched_model_sha256": digest(origin / "model.npz"),
                "selection": "Free weights inside learned connectivity; random control preserves each hidden unit's observed fan-in; larger numerical budget disclosed",
            },
        )
    if method == "dense":
        return (
            None,
            None,
            {
                "codes": 0,
                "source_selection": "Fully connected, no shared structure to select",
            },
        )
    if method == "random":
        origin = root / "validation_fit" / "current"
        metadata = json.loads((origin / "selected.json").read_text())
        codes = metadata["source_selection"]["codes"]
        index = np.load(origin / "model.npz")["index"].copy()
        rng = np.random.default_rng(769001)
        index = rng.permutation(index.ravel()).reshape(index.shape)
        basis = signed_basis(index, codes, device, torch.float32)
        basis /= basis.square().sum(-1).mean().sqrt().clamp_min(1e-12)
        return (
            basis,
            index,
            {
                "codes": codes,
                "matched_model_sha256": digest(origin / "model.npz"),
                "random_control": True,
                "selection": "Shuffle the selected current indexator, preserving its chosen budget, density and group counts",
            },
        )
    owner = method
    files = [(root / "fits" / f"{owner}_k{k}" / "selected.json", k) for k in grid]
    ranking = [
        (json.loads(path.read_text())["source_M_loss"], path, k) for path, k in files
    ]
    _, selected, codes = min(ranking, key=lambda item: item[0])
    numeric = selected.with_suffix(".npz")
    archive = np.load(numeric)
    basis = torch.tensor(archive["basis"], device=device, dtype=torch.float32)
    index = archive["index"].copy()
    return (
        basis,
        index,
        {
            "codes": codes,
            "source_selected": str(selected),
            "source_selected_sha256": digest(selected),
            "numeric_source_file_sha256": digest(numeric),
            "selection": "Source M selects the candidate within the prescribed code budget; disjoint development tasks select the budget and global numerical rule",
            "random_control": method == "random",
        },
    )


def run(root, method, device, steps, codes_override=None):
    torch.set_num_threads(2)
    if not (root / "source_choices_frozen.json").exists():
        raise RuntimeError(
            "Source comparison and conditional diagnosis must be frozen first"
        )
    boundary = StudyBoundary(root, "validation")
    boundary.install()
    folder = root / "validation_fit" / method
    folder.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    h, o, kind = protocol["hidden"], protocol["outputs"], protocol["loss"]
    code_grid = (
        [codes_override]
        if codes_override is not None
        else protocol["independent_code_grid"]
    )
    save_json(
        folder / "registered.json",
        {
            "method": method,
            "code_sha256": digest(Path(__file__)),
            "adapter_sha256": digest(Path(__file__).with_name("adaptation.py")),
            "code_grid": code_grid
            if method not in ("dense", "random", "mask_only", "mask_random")
            else None,
            "max_numeric_steps": steps,
            "choice": "Code budget and one global learning rate/activity quantile/update count minimize development query loss averaged over tasks and two fresh restarts; no seed selection or per-test-task query selection",
        },
    )
    for name in ("real_validation.py", "adaptation.py"):
        shutil.copy2(Path(__file__).with_name(name), folder / name)
    data = dict(np.load(root / "validation" / "validation.npz"))
    support_count = 25 if kind == "cross_entropy" else 512
    x, y, vx, vy = [
        torch.tensor(array, device=device, dtype=torch.float32)
        for array in (
            data["support_x"][:, :support_count],
            data["support_y"][:, :support_count],
            data["query_x"],
            data["query_y"],
        )
    ]
    x, vx = x / 255, vx / 255
    center, scale = y.mean(1), y.std((1, 2), correction=0).clamp_min(1e-8)
    if kind == "cross_entropy":
        center, scale = torch.zeros_like(center), torch.ones_like(scale)
    y, vy = (
        (y - center[:, None]) / scale[:, None, None],
        (vy - center[:, None]) / scale[:, None, None],
    )
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    status = "failed"
    try:
        records, all_traces = [], {}
        best_loss, best_model, best_trace, chosen = float("inf"), None, None, None
        with counter:
            for budget in (
                [0]
                if method in ("dense", "random", "mask_only", "mask_random")
                else code_grid
            ):
                basis, index, selection = choose_source(root, method, [budget], device)
                mask = (
                    torch.tensor(index != 0, device=device, dtype=torch.float32)
                    if method in ("mask_only", "mask_random")
                    else None
                )
                _, trace = fit_adam(
                    x,
                    y,
                    vx,
                    vy,
                    basis,
                    h,
                    o,
                    loss=kind,
                    steps=steps,
                    seed=769003,
                    mask=mask,
                )
                error = (
                    trace["validation_history"] * scale.cpu().numpy()[None, None] ** 2
                )
                # Do not tune a lucky random seed: each setting is averaged over both restarts.
                mean = error.mean(-1).reshape(len(trace["steps"]), 9, 2).mean(-1)
                checkpoint, scenario = np.unravel_index(mean.argmin(), mean.shape)
                value = float(mean[checkpoint, scenario])
                record = {
                    "codes": selection["codes"],
                    "validation_mean_loss": value,
                    "numeric_updates": int(trace["steps"][checkpoint]),
                    "global_setting": trace["settings"][2 * scenario],
                    "source_selection": selection,
                }
                records.append(record)
                all_traces.update(
                    {
                        f"k{selection['codes']}_{key}": item
                        for key, item in trace.items()
                        if isinstance(item, np.ndarray)
                    }
                )
                if value < best_loss:
                    best_loss, chosen, best_trace = value, record, trace
                    best_model = {
                        "basis": np.empty((0,))
                        if basis is None
                        else basis.cpu().numpy(),
                        "index": np.empty((0,), dtype=np.int16)
                        if index is None
                        else index.copy(),
                    }
                    if mask is not None:
                        best_model["mask"] = mask.cpu().numpy()
                print(json.dumps({"validation_method": method, **record}), flush=True)
        np.savez_compressed(
            folder / "model.npz",
            **best_model,
        )
        np.savez_compressed(
            folder / "curves.npz",
            **{
                key: value
                for key, value in best_trace.items()
                if isinstance(value, np.ndarray)
            },
        )
        np.savez_compressed(folder / "all_curves.npz", **all_traces)
        save_json(
            folder / "selected.json",
            {
                "method": method,
                "global_setting": chosen["global_setting"],
                "numeric_updates": chosen["numeric_updates"],
                "support_count": support_count,
                "validation_mean_loss": best_loss,
                "model_sha256": digest(folder / "model.npz"),
                "source_selection": chosen["source_selection"],
                "all_budget_choices": records,
                "target_query_used": False,
            },
        )
        status = "complete"
    finally:
        torch.cuda.synchronize(device)
        save_json(
            folder / "metrics.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            },
        )
        boundary.save(folder / "access_receipt.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--codes", type=int)
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.method,
        torch.device(args.device),
        args.steps,
        args.codes,
    )


if __name__ == "__main__":
    main()
