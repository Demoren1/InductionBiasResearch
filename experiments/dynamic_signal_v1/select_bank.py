"""Select the source bank on held-out source tasks across noise levels."""

import json
from pathlib import Path

import numpy as np
import torch

from experiments.dynamic_signal_v1.fit import fit
from experiments.dynamic_signal_v1.run import configurations


def main():
    plan = json.loads(Path("experiments/dynamic_signal_v1/protocol.json").read_text())
    root = Path(plan["run"])
    output = root / "bank_selection"
    output.mkdir(exist_ok=False)
    parts = [
        dict(np.load(root / "source" / f"noise_{noise}.npz")) for noise in range(3)
    ]
    data = {
        key: np.concatenate([part[key][8:10] for part in parts]) for key in parts[0]
    }
    torch.set_num_threads(2)
    device = torch.device("cuda")
    configs = configurations(plan, "oracle")
    candidates = []
    for noise in range(3):
        bank = root / "banks_relations" / f"noise_{noise}"
        consensus = torch.tensor(
            np.load(bank / "extracted.npz")["consensus"], device=device
        )
        arrays = fit(
            plan,
            data,
            "bank_selection",
            configs,
            8192,
            2,
            device,
            output / f"candidate_{noise}",
            indices=consensus,
        )
        # Tune numeric configuration separately per noise; compare source banks
        # by the same six held-out source tasks, never by latent-group recovery.
        loss = arrays["query_loss"].reshape(len(configs), 3, 2, 2)
        scores = loss.min(-1).mean(-1).min(0)
        candidates.append(
            {
                "bank": str(bank),
                "mean_query_bce": float(scores.mean()),
                "noise_scores": scores.tolist(),
            }
        )
    chosen = min(range(3), key=lambda item: candidates[item]["mean_query_bce"])
    (output / "selection.json").write_text(
        json.dumps(
            {
                "candidates": candidates,
                "chosen_id": chosen,
                "chosen_bank": candidates[chosen]["bank"],
                "rule": "Held-out source tasks8/9 across all three noise levels. No target data or teacher parameters.",
            },
            indent=2,
        )
        + "\n"
    )
    print("SELECTED BANK", chosen, flush=True)


if __name__ == "__main__":
    main()
