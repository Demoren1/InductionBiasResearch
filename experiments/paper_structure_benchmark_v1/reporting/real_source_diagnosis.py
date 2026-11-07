"""Describe real-source generalization and identifiability before any target access."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from ..fitting.adaptation import losses
from ..methods.bilevel import predict
from ..protocol.real_boundary import StudyBoundary
from ..fitting.real_dense import predict as dense_predict


def run(root):
    torch.set_num_threads(2)
    boundary = StudyBoundary(root, "select")
    boundary.install()
    protocol = json.loads((root / "protocol.json").read_text())
    d, h, o, kind = [protocol[key] for key in ("inputs", "hidden", "outputs", "loss")]
    panels = {
        name: dict(np.load(root / "public" / f"{name}.npz"))
        for name in ("support", "numerical", "structure")
    }
    dense = np.load(root / "source" / "dense" / "models.npz")
    parameter = torch.tensor(dense["parameters"])[None]
    result = {
        "data": {
            "tasks": len(parameter[0]),
            "input_dimensions": d,
            "support_per_task": panels["support"]["x"].shape[1],
            "first_layer_weights_per_task": d * h,
            "support_scalar_constraints_upper_bound": panels["support"]["x"].shape[1]
            * (o - 1 if kind == "cross_entropy" else o),
        },
        "dense": {},
        "methods": {},
        "native_before_common_refit": {},
    }
    for name, panel in panels.items():
        x, y = (
            torch.tensor(panel["x"], dtype=torch.float32) / 255,
            torch.tensor(panel["y"]),
        )
        with torch.no_grad():
            value = dense_predict(x, parameter, d, h, o)
            error = losses(value, y, kind)[0]
            result["dense"][name] = {
                "mean_loss": float(error.mean()),
                "per_task_loss": error.tolist(),
            }
            if kind == "cross_entropy":
                result["dense"][name]["accuracy"] = float(
                    (value[0].argmax(-1) == y.argmax(-1)).float().mean()
                )
    x = torch.tensor(panels["structure"]["x"], dtype=torch.float32) / 255
    y = torch.tensor(panels["structure"]["y"])
    for path in sorted((root / "fits").glob("*/selected.json")):
        metadata = json.loads(path.read_text())
        result["methods"][path.parent.name] = {
            "source_M_loss": metadata["source_M_loss"],
            "selected_sha256": metadata["selected_sha256"],
        }
    for method in ("msr", "yeh"):
        for k in protocol["independent_code_grid"]:
            path = root / "source" / f"{method}_k{k}" / "native.npz"
            if not path.exists():
                continue
            archive = np.load(path)
            basis = torch.tensor(archive["basis"])
            numeric = torch.tensor(archive["source_parameters"]).clone()
            scale, center = (
                torch.tensor(archive["normalization_scale"]),
                torch.tensor(archive["normalization_center"]),
            )
            numeric[..., k + h : -o] *= scale[None, :, None]
            numeric[..., -o:] = numeric[..., -o:] * scale[None, :, None] + center[None]
            scores = []
            with torch.no_grad():
                for begin in range(0, len(basis), 2):
                    value = predict(
                        x, basis[begin : begin + 2], numeric[begin : begin + 2], h, o
                    )
                    scores.extend(losses(value, y, kind).mean(1).tolist())
            result["native_before_common_refit"][f"{method}_k{k}"] = {
                "source_M_loss_per_configuration": scores,
                "minimum_source_M_loss": min(scores),
                "native_sha256": digest(path),
            }
    folder = root / "diagnosis"
    folder.mkdir(exist_ok=True)
    save_json(folder / "source_only.json", result)
    text = [
        "# Source-only diagnosis",
        "",
        "No private files, target tasks or true numerical/structural parameters were opened.",
        "",
        f"Each source has {result['data']['support_per_task']} labelled examples, at most {result['data']['support_scalar_constraints_upper_bound']} independent output constraints, versus {d * h} first-layer weights before biases/readout. These counts show why individually fitted dense directions can be underidentified; they do not prove that the shared-task problem is unidentifiable.",
        "",
        "## Frozen dense model",
        "",
        "| Panel | Mean loss |",
        "|---|---:|",
    ]
    for name, item in result["dense"].items():
        text.append(f"| {name} | {item['mean_loss']:.6g} |")
    text.extend(
        [
            "",
            "## Common-refit structures",
            "",
            "| Method/budget | Source M loss |",
            "|---|---:|",
        ]
    )
    text.extend(
        f"| {name} | {item['source_M_loss']:.6g} |"
        for name, item in result["methods"].items()
    )
    text.extend(
        [
            "",
            "## Native continuous controls before common refitting",
            "",
            "| Method/budget | Best source M loss |",
            "|---|---:|",
        ]
    )
    text.extend(
        f"| {name} | {item['minimum_source_M_loss']:.6g} |"
        for name, item in result["native_before_common_refit"].items()
    )
    text.extend(
        [
            "",
            "## Hypotheses and conditional next step",
            "",
            "H1: few-shot dense interpolation leaves individual directions poorly identified. H2: matching independent source neurons therefore aligns arbitrary fitting choices. H3: once a wrong discrete bank is frozen, a better codebook/readout solver cannot recover absent assignments. These hypotheses must be assessed from observable losses, direction dispersion and bank diversity, without a true mask.",
            "",
            "If the common-refit current method materially loses to the native/common-refit competitors, retain it and add a source-pooled functional discrete refinement. Evaluate candidate group changes through the task loss gradient and generalized Gauss-Newton diagonal, accept changes only after a true functional-loss decrease, and alternate with codebook/bias/readout optimization. Include label0 so connectivity can change. Select checkpoints on source M, then freeze before held-out task development/evaluation. If the current method remains competitive, do not activate this revision.",
        ]
    )
    (folder / "SOURCE_DIAGNOSIS.md").write_text("\n".join(text) + "\n")
    boundary.save(folder / "access_receipt.json", "complete")
    print(
        json.dumps(
            {
                "dense": {
                    key: value["mean_loss"] for key, value in result["dense"].items()
                },
                "native": {
                    key: value["minimum_source_M_loss"]
                    for key, value in result["native_before_common_refit"].items()
                },
                "common": {
                    key: value["source_M_loss"]
                    for key, value in result["methods"].items()
                },
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
