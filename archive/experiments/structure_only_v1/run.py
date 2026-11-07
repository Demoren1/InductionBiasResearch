"""Source calibration and paired evaluation of structure-only mask transfer."""

import argparse
import hashlib
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST, FashionMNIST

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / "structure_search_v2"
sys.path.insert(0, str(SHARED))
from distractor_fit import cases, evaluate_saved
from feature_fit import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    parser.add_argument("--stage", choices=["calibrate", "evaluate"], required=True)
    args = parser.parse_args()
    out = ROOT / "runs" / args.dataset
    shuffled = ROOT / "runs" / (args.dataset + "_shuffled")
    parent = SHARED / "runs" / ("distractor_" + args.dataset + "_v3")
    previous = SHARED / "runs" / ("distractor_" + args.dataset + "_v5")
    spec = json.loads((parent / "distractor_protocol.json").read_text())
    spec.update(
        active_features=64,
        binary_prior=True,
        structure_start=512,
        structure_every=16,
        sampling_temperature=1.0,
    )
    for directory in [out, shuffled]:
        shutil.copy2(parent / "source_splits.npz", directory / "source_splits.npz")
        (directory / "distractor_protocol.json").write_text(json.dumps(spec, indent=2))
    snapshot = out / ("source_" + args.stage + "_" + str(time.time_ns()))
    snapshot.mkdir()
    shutil.copy2(__file__, snapshot / "run.py")
    for name in [
        "feature_fit.py",
        "distractor_fit.py",
        "distractor_data.py",
        "distractor_prior.py",
    ]:
        shutil.copy2(SHARED / name, snapshot / name)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    dataset_type = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}[args.dataset]
    dataset = dataset_type(SHARED / "datasets", train=True)
    if args.stage == "calibrate":
        x, y, xv, yv, records, support, query = cases(dataset, out, spec)
        np.savez_compressed(
            out / "calibration_splits.npz", support=support, query=query
        )
        for directory, method in [
            (out, "bank_sample"),
            (out, "frequency"),
            (out, "task_consensus"),
            (out, "vae_mode"),
            (out, "vae_sample"),
            (out, "vae_z"),
            (shuffled, "frequency"),
            (shuffled, "vae_z"),
        ]:
            tuning_file = directory / "tuning.json"
            tuning = json.loads(tuning_file.read_text()) if tuning_file.exists() else []
            if any(r["method"] == method for r in tuning):
                continue
            options = list(
                itertools.product(
                    [0.001, 0.003, 0.01, 0.03], [0.0, 0.001, 0.01, 0.1, 1.0]
                )
            )
            data, history = fit(
                x, y, xv, yv, method, options, directory, spec, 1024, 2, device
            )
            scores = (
                data["query_loss"]
                .reshape(len(options), len(records), 2)
                .min(-1)
                .mean(-1)
            )
            for (rate, decay), score in zip(options, scores):
                tuning.append(
                    {
                        "method": method,
                        "lr": rate,
                        "wd": decay,
                        "source_query_bce": float(score),
                    }
                )
            tuning_file.write_text(json.dumps(tuning, indent=2))
            np.savez_compressed(directory / ("calibration_" + method + ".npz"), **data)
            (directory / ("calibration_" + method + "_history.json")).write_text(
                json.dumps(history)
            )
            print(
                directory.name,
                "TUNE",
                method,
                min(
                    (r for r in tuning if r["method"] == method),
                    key=lambda r: r["source_query_bce"],
                ),
                flush=True,
            )
        protocol = {
            "dataset": args.dataset,
            "steps": 1024,
            "restarts": 4,
            "split_seeds": 8,
            "methods": [
                "bank_sample",
                "frequency",
                "task_consensus",
                "vae_mode",
                "vae_sample",
                "vae_z",
                "untrained_vae_z",
                "shuffled_frequency",
                "shuffled_vae_z",
            ],
            "extractor_inputs": "only good source mask indices and task/lineage IDs",
            "excludes": [
                "input correlations",
                "functional utility",
                "source weights",
                "source gradients",
                "group count",
                "oracle layout",
            ],
            "source_numerical_calibration_uses_source_labels": True,
            "target_evaluation": "retrospective paired ablation on previous held-class splits; not a new blind test",
            "reused_baselines": [
                "dense_adam",
                "dense_sgd",
                "pretrained_adam",
                "frozen_pretrained",
                "random",
            ],
        }
        (out / "evaluation_protocol.json").write_text(json.dumps(protocol, indent=2))
    else:
        protocol = json.loads((out / "evaluation_protocol.json").read_text())
        frozen = out / "evaluation_frozen.json"
        if not frozen.exists():
            hashes = {}
            for directory in [out, shuffled]:
                for name in ["prior.pt", "good_bank.npz", "tuning.json"]:
                    path = directory / name
                    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
            hashes["protocol"] = hashlib.sha256(
                (out / "evaluation_protocol.json").read_bytes()
            ).hexdigest()
            frozen.write_text(json.dumps(hashes, indent=2))
        x, y, xv, yv, records, support, query = cases(
            dataset, out, spec, final=True, seeds=8
        )
        old_splits = np.load(previous / "target_splits.npz")
        if not (
            np.array_equal(support, old_splits["support"])
            and np.array_equal(query, old_splits["query"])
        ):
            raise ValueError(
                "Cannot reuse previous baselines with different target observations"
            )
        np.savez_compressed(out / "target_splits.npz", support=support, query=query)
        (out / "target_cases.json").write_text(json.dumps(records))
        test = dataset_type(SHARED / "datasets", train=False)
        from distractor_data import image_table

        pixels = image_table(test)
        for method in protocol["methods"]:
            if (out / (method + ".npz")).exists():
                continue
            directory = shuffled if method.startswith("shuffled_") else out
            actual = method.removeprefix("shuffled_")
            reference = {"untrained_vae_z": "vae_z"}.get(actual, actual)
            tuning = json.loads((directory / "tuning.json").read_text())
            selected_option = min(
                (r for r in tuning if r["method"] == reference),
                key=lambda r: r["source_query_bce"],
            )
            data, history = fit(
                x,
                y,
                xv,
                yv,
                actual,
                [(selected_option["lr"], selected_option["wd"])],
                directory,
                spec,
                1024,
                4,
                device,
            )
            accuracy, error = evaluate_saved(
                data, pixels, test.targets, records, spec, device
            )
            data.update(test_accuracy=accuracy, test_loss=error)
            np.savez_compressed(out / (method + ".npz"), **data)
            (out / (method + "_history.json")).write_text(json.dumps(history))
            choice = data["query_loss"].reshape(48, 4).argmin(1)
            values = accuracy.reshape(48, 4)[np.arange(48), choice]
            print(args.dataset, "RESULT", method, float(values.mean()), flush=True)
        baseline_hashes = {
            m: hashlib.sha256((previous / (m + ".npz")).read_bytes()).hexdigest()
            for m in protocol["reused_baselines"]
        }
        (out / "reused_baselines.json").write_text(
            json.dumps(
                {
                    "parent": str(previous.relative_to(ROOT.parent)),
                    "identical_target_observations": True,
                    "methods_sha256": baseline_hashes,
                },
                indent=2,
            )
        )
    (out / (args.stage + "_complete.json")).write_text(
        json.dumps({"stage": args.stage})
    )


if __name__ == "__main__":
    main()
