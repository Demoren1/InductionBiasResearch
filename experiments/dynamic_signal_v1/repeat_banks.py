"""Repeat the frozen primary comparison with independently rebuilt banks.

Each worker owns one seed and one GPU. Source collection, relational extraction,
source-only bank selection and MADE fitting are repeated; target HPO is frozen.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path("experiments/dynamic_signal_v1")
SEEDS = [2111481, 3111481, 4111481, 5111481, 6111481]


def execute(module, protocol, *arguments):
    command = [
        sys.executable,
        "-m",
        f"experiments.dynamic_signal_v1.{module}",
        "--protocol",
        str(protocol),
        *map(str, arguments),
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, check=True)


def prepare(replica):
    root = BASE / "runs" / f"bank_repeat_{replica}"
    if (root / "repetition_plan.json").exists():
        return root
    root.mkdir(parents=True, exist_ok=False)
    source_plan = json.loads((BASE / "protocol.json").read_text())
    target_plan = json.loads((BASE / "protocol_confirmation.json").read_text())
    for plan in [source_plan, target_plan]:
        plan.update(seed=SEEDS[replica - 1], run=str(root))
    for name, plan in [("source_protocol", source_plan), ("protocol", target_plan)]:
        (root / f"{name}.json").write_text(json.dumps(plan, indent=2) + "\n")

    # Frozen numerical settings, not new HPO. All future target choices use
    # exactly the settings selected in the independent confirmation experiment.
    frozen = BASE / "results/v2"
    for source in frozen.rglob("selection.json"):
        relative = source.relative_to(frozen)
        if "calibrat" not in str(relative):
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    # Source IMP originally used the developmental n8192 Adam calibration.
    for noise in range(3):
        size = 8192 if noise < 2 else 1024
        relative = Path("calibration_fit") / f"noise_{noise}_n{size}/selection.json"
        destination = root / "source_settings" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE / "results/v1" / relative, destination)
    plan = {
        "replica": replica,
        "seed": SEEDS[replica - 1],
        "all_seeds_fixed_in_advance": SEEDS,
        "endpoint": {"noise_id": 1, "support_size": 2048},
        "target_tasks": 16,
        "target_restarts": 8,
        "primary_methods": ["ar", "mask_unshared"],
        "source_bank_selection": "Three newly collected source noise banks; select on source tasks 8/9 across all noise levels, as in select_bank.py.",
        "frozen_settings": "results/v2 selections, including refined WSR and 4x-step dense controls; no new target HPO or method selection.",
        "source_recipe": "Same dense+IMP and relational pipeline; skip only unused legacy column-extractor refit.",
        "scope": "Five independent repetitions of the prespecified primary noise/support condition, with all thirteen controls; the other eleven conditions are not repeated.",
    }
    (root / "repetition_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return root


def rebuild_bank(root):
    source_protocol = root / "source_protocol.json"
    if not (root / "source").exists():
        execute("benchmark", source_protocol, "--only-role", "source")
    # source.py reads calibration_fit; temporarily expose the original source
    # settings, then restore the frozen confirmation settings for target fitting.
    for noise in range(3):
        size = 8192 if noise < 2 else 1024
        relative = Path("calibration_fit") / f"noise_{noise}_n{size}/selection.json"
        destination = root / relative
        saved = destination.read_bytes() if destination.exists() else None
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "source_settings" / relative, destination)
        collection = root / ("banks_large_tuned" if noise < 2 else "banks")
        collection /= f"noise_{noise}"
        try:
            if not (collection / "round_256.json").exists():
                execute(
                    "source",
                    source_protocol,
                    "--noise-id",
                    noise,
                    "--calibration-size",
                    size,
                    "--collection-only",
                )
        finally:
            if saved is not None:
                destination.write_bytes(saved)
            else:
                destination.unlink()
        if not (root / "banks_relations" / f"noise_{noise}/complete.json").exists():
            execute("relations", source_protocol, "--noise-id", noise)
    if not (root / "bank_selection/selection.json").exists():
        execute("select_bank", source_protocol)


def freeze_bank(root):
    if (root / "frozen_manifest.json").exists():
        return
    import torch

    from experiments.dynamic_signal_v1.prior import train_prior

    selection = json.loads((root / "bank_selection/selection.json").read_text())
    (root / "banks_relations").rename(root / "candidate_banks")
    chosen = root / "candidate_banks" / f"noise_{selection['chosen_id']}"
    (root / "banks_relations").mkdir()
    for noise in range(3):
        (root / "banks_relations" / f"noise_{noise}").symlink_to(
            chosen.resolve(), target_is_directory=True
        )
    prior = root / "priors/relations/noise_1"
    prior.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    train_prior(
        json.loads((root / "protocol.json").read_text()),
        chosen,
        prior,
        torch.device("cuda"),
    )
    for noise in [0, 2]:
        other = root / "priors/relations" / f"noise_{noise}"
        other.mkdir(parents=True, exist_ok=True)
        for name in ["made.pt", "frequencies.npz", "training.json"]:
            shutil.copyfile(prior / name, other / name)
    # Freeze source evidence, model and settings before generating target data.
    shutil.copyfile(root / "data_manifest.json", root / "source_data_manifest.json")
    hashes = {}
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.suffix in [".json", ".npz", ".pt"]
            and path != root / "data_manifest.json"
        ):
            hashes[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    for filename in [
        "fit.py",
        "source.py",
        "relations.py",
        "select_bank.py",
        "prior.py",
        "competitors.py",
        "run.py",
        "long_control.py",
        "weight_sharing.py",
        "benchmark.py",
        "repeat_banks.py",
    ]:
        path = BASE / filename
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "frozen_manifest.json").write_text(
        json.dumps({"chosen_bank": str(chosen), "hashes": hashes}, indent=2) + "\n"
    )


def run_target(root):
    protocol = root / "protocol.json"
    if not (root / "target").exists():
        execute("benchmark", protocol, "--only-role", "target")
    common = ["--stage", "target", "--noise-id", 1, "--size", 2048]
    stages = [
        (
            "prior",
            "priors/relations/noise_1/target_n2048",
            [
                "--bank-dir",
                root / "banks_relations/noise_1",
                "--name",
                "relations",
            ],
        ),
        ("competitors", "competitors/noise_1/target_n2048", []),
        ("run", "target_fit/noise_1_n2048", []),
        ("weight_sharing", "weight_sharing/noise_1/target_n2048", []),
        ("long_control", "long_control/noise_1/target_n2048", []),
    ]
    for module, directory, extra in stages:
        if not (root / directory / "complete.json").exists():
            execute(module, protocol, *common, *extra)
    (root / "complete.json").write_text('{"primary_comparison_complete": true}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica", type=int, choices=range(1, 6), required=True)
    args = parser.parse_args()
    root = prepare(args.replica)
    if not (root / "frozen_manifest.json").exists():
        rebuild_bank(root)
        freeze_bank(root)
    run_target(root)


if __name__ == "__main__":
    main()
