"""Store float32 fitting trajectories compactly, preserving every candidate."""

import gzip
import hashlib
import json

import numpy as np

from experiments.sine_ar_repair_v1.run import OUTPUT, dump


def main():
    manifest = []
    for source in sorted(OUTPUT.glob("*/*_history.json.gz")):
        with gzip.open(source, "rt") as stream:
            history = json.load(stream)
        destination = source.with_name(
            source.name.removesuffix(".json.gz") + ".npz"
        )
        np.savez_compressed(
            destination,
            step=np.array([r["step"] for r in history], dtype=np.int32),
            train=np.array([r["train"] for r in history], dtype=np.float32),
            population=np.array(
                [r["population"] for r in history], dtype=np.float32
            ),
        )
        manifest.append(
            {
                "source": str(source.relative_to(OUTPUT)),
                "source_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "packed": str(destination.relative_to(OUTPUT)),
                "packed_sha256": hashlib.sha256(
                    destination.read_bytes()
                ).hexdigest(),
                "steps": len(history),
                "source_bytes": source.stat().st_size,
                "packed_bytes": destination.stat().st_size,
                "dtype": "float32, as in the original Torch computation",
            }
        )
    dump(OUTPUT / "trajectory_manifest.json", manifest)
    print(
        json.dumps(
            {
                "histories": len(manifest),
                "source_bytes": sum(r["source_bytes"] for r in manifest),
                "packed_bytes": sum(r["packed_bytes"] for r in manifest),
            }
        )
    )


if __name__ == "__main__":
    main()
