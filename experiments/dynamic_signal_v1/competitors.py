"""Controls with the same source information and target optimization budget."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.dynamic_signal_v1.fit import fit
from experiments.dynamic_signal_v1.run import choose, configurations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-id", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--stage", choices=["calibrate", "target"], required=True)
    parser.add_argument(
        "--protocol", default="experiments/dynamic_signal_v1/protocol.json"
    )
    args = parser.parse_args()
    plan = json.loads(Path(args.protocol).read_text())
    root = Path(plan["run"])
    role = "calibration" if args.stage == "calibrate" else "target"
    output = (
        root / "competitors" / f"noise_{args.noise_id}" / f"{args.stage}_n{args.size}"
    )
    output.mkdir(parents=True, exist_ok=False)
    data = dict(np.load(root / role / f"noise_{args.noise_id}.npz"))
    bank = root / "banks_relations" / f"noise_{args.noise_id}"
    consensus = np.load(bank / "extracted.npz")["consensus"]
    original_bank = (
        root
        / ("banks_large_tuned" if args.noise_id < 2 else "banks")
        / f"noise_{args.noise_id}"
    )
    if plan.get("pool_source_noise", False):
        source_parts = [
            np.load(
                root
                / ("banks_large_tuned" if noise < 2 else "banks")
                / f"noise_{noise}"
                / "round_4096.npz"
            )
            for noise in range(3)
        ]
        source = {
            key: np.concatenate([part[key] for part in source_parts])
            for key in source_parts[0].files
        }
        best = []
        for part_id in range(3):
            ids_offset = part_id * len(source_parts[0]["task_ids"])
            for task in range(8):
                ids = np.flatnonzero(source_parts[part_id]["task_ids"] == task)
                best.append(
                    ids_offset + ids[source_parts[part_id]["query_loss"][ids].argmin()]
                )
    else:
        source = np.load(original_bank / "round_4096.npz")
        best = []
        for task in range(8):
            ids = np.flatnonzero(source["task_ids"] == task)
            best.append(ids[source["query_loss"][ids].argmin()])
    torch.set_num_threads(2)
    device = torch.device("cuda")
    tasks = len(data["support_y"])
    restarts = (
        plan["calibration_restarts"]
        if args.stage == "calibrate"
        else plan["target_restarts"]
    )
    selected = {}
    for method in ["transfer", "mask_unshared", "random_topology"]:
        if args.stage == "calibrate":
            configs = configurations(
                plan, "adam" if method != "random_topology" else "oracle"
            )
            # Pretrained weights are fixed; initialization scale is not an extra parameter.
            if method == "transfer":
                configs = [config for config in configs if config["initial_scale"] == 1]
        else:
            calibration_size = max(
                [n for n in plan["calibration_sizes"] if n <= args.size] or [256]
            )
            settings = json.loads(
                (
                    output.parent / f"calibrate_n{calibration_size}" / "selection.json"
                ).read_text()
            )
            configs = [settings[method]["config"]]
        count = len(configs) * tasks * restarts
        initial = indices = mask = None
        if method == "transfer":
            # Each restart uses a different source task, without target matching.
            starts = np.array(
                [
                    best[(task * restarts + restart) % len(best)]
                    for _ in configs
                    for task in range(tasks)
                    for restart in range(restarts)
                ]
            )
            initial = [source[f"parameter_{i}"][starts] for i in range(4)]
        elif method == "mask_unshared":
            mask = torch.tensor(consensus != 0, device=device).expand(count, -1, -1)
        else:
            torch.manual_seed(plan["seed"] + 1200)
            # Preserve all column fan-ins and tying counts; shuffle only input identities.
            base = torch.tensor(consensus, device=device)
            proposals = torch.stack(
                [
                    base[torch.randperm(128, device=device)]
                    for _ in range(tasks * restarts)
                ]
            )
            indices = proposals.repeat(len(configs), 1, 1)
        arrays = fit(
            plan,
            data,
            method,
            configs,
            args.size,
            restarts,
            device,
            output / method,
            indices=indices,
            initial=initial,
            mask=mask,
        )
        best_id, grid = choose(arrays, configs, tasks, restarts)
        selected[method] = {"config": configs[best_id], "grid": grid}
        (output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    (output / "complete.json").write_text(
        json.dumps(
            {
                "stage": args.stage,
                "noise_id": args.noise_id,
                "size": args.size,
                "source": str(original_bank),
                "structure": str(bank),
                "target_restarts": restarts,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
