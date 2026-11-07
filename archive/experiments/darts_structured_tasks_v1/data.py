"""Benchmark construction, with hidden truth kept out of learner archives."""

import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def teacher_predict(x, truth, specification):
    if specification["kind"] == "sine":
        projection = x @ truth["direction"]
        return (
            specification["low_amplitude"]
            * np.sin(specification["low_frequency"] * projection)
            + np.sin(specification["frequency"] * projection)
        ).astype(np.float32)
    indices = truth["indices"]
    values = np.r_[0.0, truth["numeric"][: specification["codes"]]]
    weights = values[np.abs(indices)] * np.sign(indices)
    bias, amplitude, offset = truth["numeric"][-3:]
    return (
        np.maximum(x @ weights + bias, 0).sum(-1)
        * amplitude
        / math.sqrt(specification["hidden"])
        + offset
    ).astype(np.float32)


def prepare(plan, task, frequency=None):
    name = task if task == "deepsets" else f"sine_{int(frequency)}"
    output = Path(plan["output"]) / name
    output.mkdir(parents=True, exist_ok=False)
    spec = {**plan[task], "kind": task, "name": name}
    rng = np.random.default_rng(plan["seed"] + (1 if task == "deepsets" else 2))
    if task == "deepsets":
        dimensions = spec["points"] * spec["features"]
        hidden = spec["points"] * spec["point_hidden"]
        spec.update({"inputs": dimensions, "hidden": hidden})
        permutation = rng.permutation(dimensions)
        indices = np.zeros((dimensions, hidden), dtype=np.int16)
        local = np.arange(1, spec["codes"] + 1).reshape(
            spec["features"], spec["point_hidden"]
        )
        for point in range(spec["points"]):
            rows = permutation[
                point * spec["features"] : (point + 1) * spec["features"]
            ]
            columns = np.arange(
                point * spec["point_hidden"],
                (point + 1) * spec["point_hidden"],
            )
            indices[np.ix_(rows, columns)] = local
        codes = np.array([0.8, -1.1, 1.3, 0.45])
        bias = 0.15
        deviations = np.linalg.norm(codes.reshape(2, 2), axis=0)
        means = [
            sigma * math.exp(-0.5 * (bias / sigma) ** 2) / math.sqrt(2 * math.pi)
            + bias * (1 + math.erf(bias / sigma / math.sqrt(2))) / 2
            for sigma in deviations
        ]
        offset = -spec["points"] * sum(means) / math.sqrt(hidden)
        truth = {
            "indices": indices,
            "numeric": np.r_[codes, bias, 1.0, offset].astype(np.float32),
            "permutation": permutation,
        }
    else:
        spec["frequency"] = float(frequency)
        labels = rng.permutation(spec["inputs"]) % spec["codes"] + 1
        codes = rng.normal(size=spec["codes"])
        direction = codes[labels - 1]
        direction /= np.linalg.norm(direction)
        truth = {
            "indices": labels[:, None].astype(np.int16),
            "direction": direction.astype(np.float32),
        }
    observations = {}
    # All frequencies share X, the low component, direction and split seeds.
    rng = np.random.default_rng(plan["seed"] + (100 if task == "deepsets" else 200))
    for split in ("support", "query", "selection"):
        x = rng.normal(size=(spec[split], spec["inputs"])).astype(np.float32)
        observations[split + "_x"] = x
        observations[split + "_y"] = teacher_predict(x, truth, spec)
    spec["target_variance"] = float(observations["support_y"].var())
    spec["target_mean"] = float(observations["support_y"].mean())
    np.savez_compressed(output / "source.npz", **observations)
    np.savez_compressed(output / "teacher_audit_only.npz", **truth)
    save_json(output / "specification.json", spec)
    save_json(output / "protocol.json", plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in BASE.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    save_json(
        output / "prepared.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "test_created": False,
            "source_sha256": digest(output / "source.npz"),
            "snapshot_sha256": {p.name: digest(p) for p in snapshot.glob("*.py")},
        },
    )
    return output


def create_test(output):
    output = Path(output)
    frozen = json.loads((output / "frozen.json").read_text())
    for name, expected in frozen["sha256"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Frozen artifact changed: {name}")
    plan = json.loads((output / "protocol.json").read_text())
    spec = json.loads((output / "specification.json").read_text())
    with np.load(output / "teacher_audit_only.npz") as archive:
        truth = dict(archive)
    rng = np.random.default_rng(plan["seed"] + 900000)
    x = rng.normal(size=(spec["test"], spec["inputs"])).astype(np.float32)
    data = {"test_x": x, "test_y": teacher_predict(x, truth, spec)}
    if spec["kind"] == "deepsets":
        base = rng.normal(size=(64, spec["points"], spec["features"]))
        permuted = np.stack(
            [
                np.stack([row[rng.permutation(spec["points"])] for _ in range(16)])
                for row in base
            ]
        ).reshape(64, 16, -1)
        observed = np.empty_like(permuted)
        observed[..., truth["permutation"]] = permuted
        data["permutation_x"] = observed.astype(np.float32)
    path = output / "heldout.npz"
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **data)
    save_json(
        output / "heldout_manifest.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "frozen_utc": frozen["frozen_utc"],
            "heldout_sha256": digest(path),
        },
    )
