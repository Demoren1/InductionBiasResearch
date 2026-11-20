"""Reweight code directions while preserving every original weight equality."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.paper_transfer_v2.common import CONFIG
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .common import OUT
from .contrasts import scatter_matrices


def build_conditioning(dataset, device):
    root = OUT / dataset
    folder = root / "conditioning"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "bank.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    settings = CONFIG["real"][dataset]
    layout = Layout(*settings["layout"])
    signals, noises = scatter_matrices(dataset, layout, device)
    known = root / "known"
    choices = json.loads((known / "choices.json").read_text())
    bank = json.loads((known / "bank.json").read_text())
    input_models = {
        bank[choices["family"][name]["candidate"]]["path"]
        for name in ("ward", "lloyd", "repartition")
    }
    result = {}
    for path in sorted(input_models):
        base = read_data(Path(path))["factor"]
        n_blocks = len(base)
        if n_blocks == 1:
            signals_for_blocks = signals.mean(0, keepdims=True)
            noises_for_blocks = noises.mean(0, keepdims=True)
        else:
            repeat = n_blocks // layout.sites
            signals_for_blocks = np.repeat(signals, repeat, axis=0)
            noises_for_blocks = np.repeat(noises, repeat, axis=0)
        spectra = []
        for factor, signal, noise in zip(
            base, signals_for_blocks, noises_for_blocks, strict=True
        ):
            a = factor.T @ signal @ factor
            b = factor.T @ noise @ factor
            cov = b + 0.03 * max(np.trace(b) / len(b), 1e-8) * np.eye(len(b))
            values, vectors = eigh(a, cov)
            scaled = np.maximum(values, 0)
            scaled /= max(scaled.max(), 1e-12)
            spectra.append((factor @ vectors, scaled))
        for floor in (0.03, 0.10, 0.30, 1.0):
            factors = []
            for directions, fitted in spectra:
                direction = directions * (floor + fitted) ** 0.5
                direction *= np.sqrt(len(fitted)) / np.linalg.norm(direction)
                factors.append(direction)
            name = f"conditioned_{Path(path).stem}_{floor:g}"
            output = folder / f"{name}.npz"
            atomic_npz(output, factor=np.stack(factors).astype(np.float32))
            result[name] = {
                "path": str(output),
                "family": "conditioned_groups",
            }
    save_json(manifest, result)
    return result
