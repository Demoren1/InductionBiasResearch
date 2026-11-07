"""Strong competitors using the same task statistic without a learned prior."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.adaptive_tanh_v1.source import scalar_kmeans
from experiments.basin_routing_v1.run import ROOT, store_selected
from experiments.basin_structure_transfer_v1.core import canonicalize, fit
from experiments.basin_structure_transfer_v1.run import save_fit


def run(frequency, cue, device):
    protocol = json.loads((ROOT / "cue_control_protocol.json").read_text())
    source = Path(protocol["input"]) / f"frequency_{frequency}_{cue}"
    parent = json.loads((source / "protocol.json").read_text())
    base_path = Path(parent["source_run"]) / "protocol.json"
    base = json.loads(base_path.read_text())
    plan = {**base, "batch": parent["batch"], "query": parent["query"]}
    chosen = json.loads((source / "numeric_settings.json").read_text())
    output = Path(protocol["output"]) / source.name
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        ROOT / "cue_controls.py",
        ROOT / "cue_control_protocol.json",
        ROOT / "run.py",
        ROOT.parent / "basin_structure_transfer_v1/core.py",
        ROOT.parent / "basin_structure_transfer_v1/run.py",
        ROOT.parent / "adaptive_tanh_v1/source.py",
        base_path,
    ]
    paths += [
        source / name
        for name in (
            "protocol.json",
            "numeric_settings.json",
            "teacher_diagnostic.npz",
            "feature_diagnostic.npz",
            "complete.json",
        )
    ]
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    (output / "frozen_before_fit.json").write_text(
        json.dumps(hashes, indent=2)
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in paths:
        if path.suffix == ".py":
            name = str(path.relative_to(ROOT.parents[1])).replace("/", "__")
            (snapshot / name).write_bytes(path.read_bytes())
    teacher = np.load(source / "teacher_diagnostic.npz")
    spec = {
        key: teacher[key]
        if key in ("indices", "families")
        else torch.tensor(teacher[key], device=device)
        for key in teacher.files
    }
    feature = np.load(source / "feature_diagnostic.npz")["empirical_moment"]
    torch.manual_seed(protocol["cluster_seed"])
    kmeans = scalar_kmeans(torch.tensor(feature, device=device).float(), 3)
    kmeans = canonicalize(kmeans.cpu().numpy())
    order = np.argsort(feature, axis=1)
    balanced = np.empty_like(order)
    np.put_along_axis(balanced, order, np.arange(24)[None] // 8, axis=1)
    balanced = canonicalize(balanced)
    count = len(feature)
    for method in protocol["methods"]:
        dense = method.startswith("moment_dense_")
        if dense:
            optimizer = method.rsplit("_", 1)[-1]
            configs = [
                {**chosen[f"dense_{optimizer}"], "radius": radius}
                for radius in protocol["radii"]
            ]
            starts = protocol["dense_starts_per_radius"]
            indices = None
        else:
            configs = [chosen["code"]]
            starts = parent["static_starts"]
            route = kmeans if method == "moment_kmeans" else balanced
            indices = np.repeat(route[:, None], starts, axis=1)
        result, history, timing = fit(
            plan,
            spec,
            configs,
            starts,
            parent["initial_fit_seed"],
            device,
            indices=indices,
            steps=protocol["numeric_steps"],
            query_seed=parent["final_query_seed"],
            initial_direction=feature if dense else None,
        )
        if dense:
            for key, value in result.items():
                result[key] = value.reshape(
                    count, 1, len(configs) * starts, *value.shape[3:]
                )
        else:
            result["indices"] = indices
        timing["model_updates_per_task"] = (
            len(configs) * starts * protocol["numeric_steps"]
        )
        save_fit(output / method, result, history, timing)
        extra = (
            {}
            if dense
            else {
                "route": route,
                "route_exact": (route == spec["indices"]).all(1),
            }
        )
        if dense:
            candidate = result["query"][:, 0].argmin(1)
            extra["selected_initial_radius"] = np.repeat(
                protocol["radii"], starts
            )[candidate]
        store_selected(output / method, result, spec, plan, device, extra)
    unchanged = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
        for path, value in hashes.items()
    )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "frozen_inputs_unchanged": unchanged,
                "methods": protocol["methods"],
                "frequency": frequency,
                "cue": cue,
            },
            indent=2,
        )
    )
    if not unchanged:
        raise RuntimeError("Control inputs changed during fitting")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=int, choices=(6, 8), required=True)
    parser.add_argument("--cue", choices=("present", "absent"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    torch.set_num_threads(2)
    run(args.frequency, args.cue, torch.device("cuda:0"))


if __name__ == "__main__":
    main()
