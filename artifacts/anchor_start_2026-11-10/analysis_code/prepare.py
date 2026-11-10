"""Generate paired observations; learner processes never import this module."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np

from archive.experiments.g34_theory_v1.general import (
    general_index,
    general_numbers,
)
from experiments.paper_structure_benchmark_v1.protocol.prepare import (
    outputs_for,
    prepare,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)


def setup(base):
    base.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).with_name("PROTOCOL.md"), base / "PROTOCOL.md")
    prepare(base / "standard", "confirmation", 3, 16, seed=27111054013)
    standard = base / "standard"
    protocol = json.loads((standard / "protocol.json").read_text())
    general = base / "general"
    for name in (
        "public",
        "private",
        "source",
        "dense",
        "fits",
        "metrics",
        "logs",
        "receipts",
        "targets",
    ):
        (general / name).mkdir(parents=True)
    general_protocol = copy.deepcopy(protocol)
    general_protocol["cases"] = []
    folds = protocol["folds"]
    for number in range(6):
        case = f"case_{number:02d}"
        variant = ("local", "quartet_mix", "random")[number // 2]
        replicate, outputs = number % 2, 1 + number % 2
        rng = np.random.default_rng(27111064019 + 100003 * number)
        index = general_index(rng, variant)
        code, bias, head, offset = general_numbers(
            rng, variant, replicate, 16, outputs
        )
        np.savez_compressed(general / "private" / f"{case}.npz", index=index)
        sizes = {"inputs": 16, "hidden": 16, "codes": 4, "outputs": outputs}
        save_json(general / "public" / f"{case}.json", sizes)
        general_protocol["cases"].append({"name": case, **sizes})
        panels = {name: {} for name in folds}
        for task in range(16):
            for name, count in folds.items():
                x = rng.normal(size=(count, 16))
                if name in ("fixed_train", "selection"):
                    x *= np.resize([0.5, 1, 2, 4], count)[:, None]
                clean = outputs_for(
                    x, index, code[task], bias[task], head[task], offset[task]
                )
                noise = rng.normal(size=clean.shape)
                for condition, sigma in enumerate((0.0, 0.2)):
                    panels[name][f"x_{condition}_{task}"] = x
                    panels[name][f"y_{condition}_{task}"] = (
                        clean + sigma * noise
                    )
        for condition in (0, 1):
            folder = general / "public" / f"{case}_c{condition}"
            folder.mkdir()
            for name, panel in panels.items():
                np.savez_compressed(
                    folder / f"{name}.npz",
                    **{
                        f"{key}_{task}": panel[f"{key}_{condition}_{task}"]
                        for task in range(16)
                        for key in ("x", "y")
                    },
                )
        code, bias, head, offset = general_numbers(
            rng, variant, replicate, 8, outputs
        )
        target = {}
        for task in range(8):
            for name, count in (
                ("support", 1024),
                ("selection", 256),
                ("query", 2048),
            ):
                x = rng.normal(size=(count, 16))
                target[f"{name}_x_{task}"] = x
                target[f"{name}_y_{task}"] = outputs_for(
                    x, index, code[task], bias[task], head[task], offset[task]
                )
        np.savez_compressed(
            general / "private" / f"{case}_targets.npz", **target
        )
    save_json(general / "protocol.json", general_protocol)
    for root in (standard, general):
        configuration = json.loads((root / "protocol.json").read_text())
        configuration["allowed_gpus"] = list(range(8))
        configuration["forbidden_gpus"] = []
        save_json(root / "protocol.json", configuration)
        for entry in configuration["cases"]:
            case = entry["name"]
            folder = root / "target_data" / case
            folder.mkdir(parents=True)
            with np.load(root / "private" / f"{case}_targets.npz") as target:
                for panel in ("support", "selection"):
                    np.savez_compressed(
                        folder / f"{panel}.npz",
                        **{
                            key: np.stack(
                                [
                                    target[f"{panel}_{key}_{t}"]
                                    for t in range(8)
                                ]
                            )
                            for key in ("x", "y")
                        },
                    )
                np.savez_compressed(
                    folder / "query_x.npz",
                    x=np.stack([target[f"query_x_{t}"] for t in range(8)]),
                )
        for method in ("full", "anchor_fixed", "anchor_free"):
            variant = root / "methods" / method
            variant.mkdir(parents=True)
            for name in ("public", "source"):
                (variant / name).symlink_to(
                    root / name, target_is_directory=True
                )
            for name in ("fits", "metrics", "receipts"):
                (variant / name).mkdir()
    project = Path(__file__).resolve().parents[2]
    files = [
        *Path(__file__).parent.glob("*.py"),
        Path(__file__).with_name("PROTOCOL.md"),
        *(project / "experiments/sharing_structure_research_v1").glob(
            "blind_*.py"
        ),
        project / "archive/experiments/g34_support_v1/bank.py",
        *(project / "experiments/paper_structure_benchmark_v1/fitting").glob(
            "*.py"
        ),
    ]
    save_json(
        base / "registration.json",
        {
            "code_sha256": {
                str(p.relative_to(project)): digest(p) for p in files
            },
            "seeds": {"standard": 27111054013, "general": 27111064019},
            "methods": ["full", "anchor_fixed", "anchor_free"],
            "cases": 12,
            "source_conditions": 24,
            "allowed_gpus": list(range(8)),
            "target_budget": 1024,
            "source_models_shared": True,
            "truth_opened": False,
        },
    )
    for path in files:
        target = base / "code_snapshot" / path.relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(json.dumps({"prepared": str(base), "source_jobs": 24}), flush=True)
