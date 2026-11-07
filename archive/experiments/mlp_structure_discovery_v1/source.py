"""Collect generic dense/pruned solvers without access to task geometry."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mlp_structure_discovery_v1.fit import (
    fit_models,
    read_observations,
    untied_indices,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="experiments/mlp_structure_discovery_v1/protocol.json"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    path = Path(args.protocol)
    plan = json.loads(path.read_text())
    root = Path(plan["output"])
    output = root / "source"
    output.mkdir(parents=True, exist_ok=False)
    (root / "protocol.json").write_text(path.read_text())
    (root / "started.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(
                    Path(plan["source_data"]).read_bytes()
                ).hexdigest(),
                "torch": torch.__version__,
                "device": args.device,
                "source_files": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.glob("*.py")
                },
            },
            indent=2,
        )
        + "\n"
    )
    torch.set_num_threads(2)
    device = torch.device(args.device)
    data = read_observations(plan["source_data"], plan["discovery_tasks"])
    dimensions = data["support_x"].shape[-1]
    shape = (dimensions, plan["hidden"])
    configs = list(itertools.product(plan["learning_rates"], plan["weight_decays"]))
    rows = [
        {
            "config": cid,
            "task": task,
            "restart": restart,
            "lr": rate,
            "weight_decay": decay,
        }
        for cid, (rate, decay) in enumerate(configs)
        for task in range(len(data["support_y"]))
        for restart in range(plan["source_restarts"])
    ]
    index = np.broadcast_to(untied_indices(np.ones(shape)), (len(rows), *shape)).copy()
    raw = fit_models(
        data,
        index,
        rows,
        plan,
        output / "dense_grid",
        device,
        steps=plan["source_steps"],
    )
    losses = raw["query_loss"].reshape(len(configs), -1).mean(-1)
    selected = int(losses.argmin())
    ids = np.array([i for i, row in enumerate(rows) if row["config"] == selected])
    raw = {key: value[ids] for key, value in raw.items()}
    rows = [rows[i] for i in ids]
    (output / "selection.json").write_text(
        json.dumps(
            {
                "selected_config": selected,
                "configs": configs,
                "query_losses": losses.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(output / "level_0.npz", **raw)
    (output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    for level, fraction in enumerate(plan["retained_fractions"][1:], 1):
        weights, bias, readout, offset = [raw[f"parameter_{i}"] for i in range(4)]
        # W*a is invariant to a function-preserving positive ReLU rescaling.
        scores = np.abs(weights * readout[:, None]).reshape(len(rows), -1)
        retained = max(1, round(np.prod(shape) * fraction))
        chosen = np.argsort(-scores, axis=1, kind="stable")[:, :retained]
        mask = np.zeros_like(scores, dtype=bool)
        np.put_along_axis(mask, chosen, True, axis=1)
        index = untied_indices(mask.reshape(len(rows), *shape))
        raw = fit_models(
            data,
            index,
            rows,
            plan,
            output / f"level_{level}",
            device,
            steps=plan["pruning_steps"],
            initial=[weights, bias, readout, offset],
        )
    (output / "complete.json").write_text(
        json.dumps(
            {"levels": len(plan["retained_fractions"]), "models_per_level": len(rows)},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
