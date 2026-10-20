"""Pilot gate before collecting any mask bank or training a generator."""

import numpy as np

from core import correct_mask, save_json, task
from pilot_legacy import fit


def main():
    spec = task(0, 0)
    records = []
    for scale in (0.35, 0.7, 1.25, 2.0):
        for kind in ("dense", "correct_support"):
            masks = (
                np.ones((2048, 8, 2), dtype=np.float32)
                if kind == "dense"
                else correct_mask(spec, 2048)
            )
            result = fit(
                masks,
                spec,
                8101,
                scale=scale,
                steps=2048,
                checkpoints=(512, 1024, 2048),
            )
            record = {
                "kind": kind,
                "scale": scale,
                "count": len(masks),
                "success": int((result["losses"] < 0.1).sum()),
                "quantiles": np.quantile(
                    result["losses"], [0, 0.1, 0.5, 0.9, 1]
                ).tolist(),
                "history_success": {
                    key: int((value < 0.1).sum())
                    for key, value in result["history"].items()
                },
                "seconds": result["seconds"],
            }
            records.append(record)
            print(record, flush=True)
            np.savez_compressed(
                f"tmp/sine_mask_vae/pilot_{kind}_{scale}.npz",
                **{
                    key: result[key]
                    for key in ("weights", "amplitude", "gradients", "losses")
                },
            )
    save_json("tmp/sine_mask_vae/pilot.json", {"task": spec, "fits": records})


if __name__ == "__main__":
    main()
