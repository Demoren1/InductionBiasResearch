"""Matched source selection and target generation on two fixed geometries."""

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import torch

from experiments.paper_recovery_extension_v1.synthetic import calibrated
from experiments.paper_structure_benchmark_v1.fitting.adaptation import (
    signed_basis,
    values,
)
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .prepare import ORIGINAL, OUTPUT, ROOT


def source(number, device):
    torch.set_num_threads(2)
    root, output = (
        (ROOT / f"synthetic_{number}").resolve(),
        OUTPUT / f"synthetic_{number}",
    )
    boundary = StudyBoundary(root)
    boundary.install()
    sizes = json.loads((root / "protocol.json").read_text())
    h, o = sizes["hidden"], sizes["outputs"]
    arrays = [
        dict(np.load(root / "public" / f"{n}.npz"))
        for n in ("support", "numerical")
    ]
    x, y, vx, vy = [
        torch.tensor(a[k], device=device, dtype=torch.float64)
        for a, k in (
            (arrays[0], "x"),
            (arrays[0], "y"),
            (arrays[1], "x"),
            (arrays[1], "y"),
        )
    ]
    x, vx = x / 255, vx / 255
    center, scale = y.mean(1), y.std((1, 2), correction=0).clamp_min(1e-8)
    y, vy = (
        (y - center[:, None]) / scale[:, None, None],
        (vy - center[:, None]) / scale[:, None, None],
    )
    metadata = json.loads((root / "source/asrr/candidates.json").read_text())[
        "candidates"
    ]
    candidates = dict(np.load(root / "source/asrr/candidates.npz"))
    fitted, bases = [], []
    for row in metadata:
        basis = signed_basis(candidates[row["name"]], 4, device)
        basis /= basis.square().sum(-1).mean().sqrt().clamp_min(1e-12)
        parameter, _, _ = calibrated(basis, x, y, vx, vy, h, o, seed=761003)
        fitted.append(parameter)
        bases.append(basis)
        print(json.dumps({"source_candidate": row["name"]}), flush=True)
    boundary.role = "select"
    panel = dict(np.load(root / "public/structure.npz"))
    mx = torch.tensor(panel["x"], device=device, dtype=torch.float64) / 255
    my = torch.tensor(panel["y"], device=device, dtype=torch.float64)
    my = (my - center[:, None]) / scale[:, None, None]
    with torch.no_grad():
        risks = [
            float((values(mx, b, p[None], h, o)[0] - my).square().mean())
            for b, p in zip(bases, fitted)
        ]
    best = int(np.argmin(risks))
    np.savez_compressed(
        root / "fits/structures.npz",
        asrr=bases[best].cpu().numpy(),
        index=candidates[metadata[best]["name"]],
    )
    save_json(
        output / "source_selected.json",
        {
            "candidate": metadata[best]["name"],
            "source_M_losses": risks,
            "models_sha256": digest(root / "fits/structures.npz"),
            "frozen_utc": datetime.now(UTC).isoformat(),
        },
    )
    boundary.save(output / "source_access.json", "complete")


def prepare_target(number):
    from experiments.paper_structure_benchmark_v1.protocol.prepare import (
        outputs_for,
    )
    from experiments.paper_structure_benchmark_v1.protocol.prepare_general import (
        general_numbers,
    )

    root, output = ROOT / f"synthetic_{number}", OUTPUT / f"synthetic_{number}"
    frozen = json.loads((output / "source_selected.json").read_text())
    if digest(root / "fits/structures.npz") != frozen["models_sha256"]:
        raise RuntimeError("Changed synthetic source choice")
    case = f"case_{number:02d}"
    folder = root / "targets" / f"{case}_c0"
    folder.mkdir()
    with np.load(root / "fits/structures.npz") as data:
        np.savez_compressed(folder / "structures.npz", asrr=data["asrr"])
    sizes = json.loads((root / "protocol.json").read_text())
    save_json(folder / "sizes.json", sizes)
    rng = np.random.default_rng(26111499113 + number)
    code, bias, head, offset = general_numbers(
        rng, "random", number % 2, 4, sizes["outputs"]
    )
    teacher = np.load(ORIGINAL / "private" / f"{case}.npz")["index"]
    for name, count in (("support", 256), ("selection", 64), ("query", 2048)):
        x = rng.normal(size=(4, count, 16))
        y = np.stack(
            [
                outputs_for(
                    x[t], teacher, code[t], bias[t], head[t], offset[t]
                )
                for t in range(4)
            ]
        )
        if name == "query":
            np.savez_compressed(folder / "query_x.npz", x=x)
            np.savez_compressed(root / "private/answers.npz", y=y)
        else:
            np.savez_compressed(folder / f"{name}.npz", x=x, y=y)
    save_json(
        output / "target_frozen.json",
        {
            "source_choice": frozen,
            "query_labels_opened_for_fitting": False,
            "observations_sha256": {
                name: digest(folder / name)
                for name in ("support.npz", "selection.npz", "query_x.npz")
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("source", "prepare-target"))
    parser.add_argument("--number", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "source":
        source(args.number, torch.device(args.device))
    else:
        prepare_target(args.number)
