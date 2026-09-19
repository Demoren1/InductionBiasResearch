"""Run and summarize independent no-z generator restarts for the z ablation.

Every restart sees the same data for its training seed. Only the generator
initialization changes. A winner is selected by training-validation BCE; test
results are read only after that choice has been made.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "pattern/outputs/bilevel_mask/generated_sharing_no_z_multistart_corrected_20260919"
CURRENT_BASELINES = ROOT / "pattern/outputs/bilevel_mask/generated_sharing_z_current_code_20260919"
REFERENCES = ROOT / "pattern/outputs/bilevel_mask/generated_sharing_balanced_final_20260915"
PREVIOUS = ROOT / "mds/data/2026-09-19/z_importance_summary.json"
SUMMARY = ROOT / "mds/data/2026-09-19/no_z_multistart_summary.json"
FIGURE = ROOT / "mds/assets/2026-09-19/no_z_multistart_comparison.png"
SEEDS = tuple(range(42, 50))
RESTARTS = 16
STEPS = 150


def initialization_seed(seed: int, restart: int) -> int:
    return seed if restart == 0 else 1_000_000 + seed * 100 + restart


def result_path(output_root: Path, seed: int, restart: int) -> Path:
    return output_root / f"seed{seed}" / f"restart{restart:02d}" / "summary.json"


def run_one(output_root: Path, seed: int, restart: int, gpu: int) -> None:
    result = result_path(output_root, seed, restart)
    if result.exists():
        print(f"SKIP seed={seed} restart={restart}", flush=True)
        return
    output = result.parent
    if output.exists():
        raise FileExistsError(f"incomplete existing run: {output}")
    reference = REFERENCES / f"seed{seed}" / "training.pt"
    if not reference.exists():
        raise FileNotFoundError(reference)
    log = output_root / "logs" / f"seed{seed}_restart{restart:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1"})
    command = [sys.executable, "-m", "scripts.ablate_generated_sharing_z",
               "--output", str(output), "--reference", str(reference),
               "--policy", "coordinate_only", "--device", "cuda",
               "--outer-steps", str(STEPS),
               "--initialization-seed", str(initialization_seed(seed, restart))]
    with log.open("w") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"seed={seed} restart={restart} GPU={gpu} failed; see {log}")
    print(f"DONE seed={seed} restart={restart} GPU={gpu}", flush=True)


def run_all(output_root: Path, gpus: tuple[int, ...], seeds: tuple[int, ...],
            restarts: int) -> None:
    if not gpus or 3 in gpus:
        raise ValueError("choose at least one GPU and exclude GPU 3")
    jobs = [(seed, restart) for seed in seeds for restart in range(restarts)]
    def worker(gpu: int, assigned: list[tuple[int, int]]) -> None:
        for seed, restart in assigned:
            run_one(output_root, seed, restart, gpu)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs[index::len(gpus)])
                   for index, gpu in enumerate(gpus)]
        for future in futures:
            future.result()


def run_current_baselines(output_root: Path, gpus: tuple[int, ...]) -> None:
    """Remeasure z baselines after the permutation-cache speed improvement."""
    if not gpus or 3 in gpus:
        raise ValueError("choose at least one GPU and exclude GPU 3")
    jobs = [(seed, policy, steps) for seed in SEEDS
            for policy, steps in (("learned", 100), ("frozen_bank", 150))]

    def worker(gpu: int, assigned: list[tuple[int, str, int]]) -> None:
        for seed, policy, steps in assigned:
            output = output_root / f"{policy}_seed{seed}"
            if (output / "summary.json").exists():
                print(f"SKIP baseline={policy} seed={seed}", flush=True)
                continue
            if output.exists():
                raise FileExistsError(f"incomplete existing run: {output}")
            reference = REFERENCES / f"seed{seed}" / "training.pt"
            log = output_root / "logs" / f"{policy}_seed{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1"})
            command = [sys.executable, "-m", "scripts.ablate_generated_sharing_z",
                       "--output", str(output), "--reference", str(reference),
                       "--policy", policy, "--device", "cuda", "--outer-steps", str(steps)]
            with log.open("w") as stream:
                completed = subprocess.run(command, cwd=ROOT, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise RuntimeError(f"baseline={policy} seed={seed} GPU={gpu} failed; see {log}")
            print(f"DONE baseline={policy} seed={seed} GPU={gpu}", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs[index::len(gpus)])
                   for index, gpu in enumerate(gpus)]
        for future in futures:
            future.result()


def select_by_validation(records: list[dict]) -> dict:
    """Use the same within-training validation criterion as the z experiment."""
    return min(records, key=lambda row: (row["training"]["selected_training_validation_bce"],
                                         row["restart"]))


def stats(values: list[float]) -> dict:
    return {"mean": mean(values), "sd": stdev(values), "values": values}


def paired_stats(values: list[float]) -> dict:
    average = mean(values)
    margin = student_t.ppf(0.975, len(values) - 1) * stdev(values) / len(values) ** 0.5
    return {"mean": average, "ci95_t": [average - margin, average + margin],
            "no_z_wins": sum(value < 0 for value in values), "values": values}


def summarize(output_root: Path, seeds: tuple[int, ...] = SEEDS,
              restarts: int = RESTARTS) -> dict:
    previous = json.loads(PREVIOUS.read_text())
    by_seed = {row["seed"]: row for row in previous["per_seed"]}
    rows = []
    for seed in seeds:
        records = []
        for restart in range(restarts):
            record = json.loads(result_path(output_root, seed, restart).read_text())
            if record["policy"] != "coordinate_only" or record["seed"] != seed:
                raise ValueError(f"wrong policy or data seed at {seed}/{restart}")
            if record["config"]["outer_steps"] != STEPS:
                raise ValueError(f"wrong training duration at {seed}/{restart}")
            if record["training"]["initialization_seed"] != initialization_seed(seed, restart):
                raise ValueError(f"wrong initialization at {seed}/{restart}")
            record["restart"] = restart
            records.append(record)
        if len({row["reference_checkpoint_sha256"] for row in records}) != 1:
            raise ValueError(f"reference checkpoint changed at seed {seed}")
        if len({row["training"]["selected_training_validation_bce"] for row in records}) == 1:
            raise ValueError(f"all initializations produced the same result at seed {seed}")
        if abs(records[0]["test"]["mean_query_bce"] -
               by_seed[seed]["no_z_150"]["test_bce"]) > 1e-6:
            raise ValueError(f"original no-z run did not reproduce at seed {seed}")
        selected = select_by_validation(records)
        trajectory = [select_by_validation(records[:count]) for count in range(1, restarts + 1)]
        rows.append({
            "seed": seed,
            "selected_restart": selected["restart"],
            "selected_initialization_seed": selected["training"]["initialization_seed"],
            "selected_training_validation_bce": selected["training"]["selected_training_validation_bce"],
            "selected_outer_step": selected["training"]["selected_outer_step"],
            "selected_heldout_validation_bce": selected["validation"]["mean_query_bce"],
            "selected_test_bce": selected["test"]["mean_query_bce"],
            "selected_test_accuracy": selected["test"]["mean_query_accuracy"],
            "total_training_seconds": sum(row["training"]["training_seconds"] for row in records),
            "test_bce_by_number_of_starts": [row["test"]["mean_query_bce"] for row in trajectory],
            "all_restarts": [{"restart": row["restart"],
                              "training_validation_bce": row["training"]["selected_training_validation_bce"],
                              "heldout_validation_bce": row["validation"]["mean_query_bce"],
                              "test_bce": row["test"]["mean_query_bce"],
                              "test_accuracy": row["test"]["mean_query_accuracy"],
                              "training_seconds": row["training"]["training_seconds"]}
                             for row in records],
        })
    current = {}
    for policy, old_name in (("learned", "learned_z_100"),
                             ("frozen_bank", "sixteen_fixed_z_150")):
        results = [json.loads((CURRENT_BASELINES / f"{policy}_seed{seed}/summary.json").read_text())
                   for seed in seeds]
        for result in results:
            expected = by_seed[result["seed"]][old_name]["test_bce"]
            if abs(result["test"]["mean_query_bce"] - expected) > 1e-6:
                raise ValueError(f"current-code {policy} did not reproduce at seed {result['seed']}")
        current[policy] = stats([row["training"]["training_seconds"] for row in results])
    best_minus_learned = [row["selected_test_bce"] - by_seed[row["seed"]]["learned_z_100"]["test_bce"]
                          for row in rows]
    best_minus_single = [row["selected_test_bce"] - by_seed[row["seed"]]["no_z_150"]["test_bce"]
                         for row in rows]
    return {
        "protocol": {"data_seeds": list(seeds), "restarts_per_seed": restarts,
                     "outer_steps_per_restart": STEPS,
                     "latent_input": "absent: coordinate-only generator",
                     "varied_across_restarts": "generator initialization only",
                     "initialization_seed_rule": "data seed for restart 0; 1000000 + 100 * data seed + restart otherwise",
                     "selected_by": "minimum within-training validation BCE",
                     "test_used_for_selection": False,
                     "current_code_baselines_reproduced_exactly": True},
        "aggregate": {
            "no_z_best_of_16_bce": stats([row["selected_test_bce"] for row in rows]),
            "no_z_best_of_16_accuracy": stats([row["selected_test_accuracy"] for row in rows]),
            "no_z_total_training_seconds": stats([row["total_training_seconds"] for row in rows]),
            "no_z_single_current_training_seconds": stats([
                row["all_restarts"][0]["training_seconds"] for row in rows]),
            "learned_z_bce": previous["aggregate"]["learned_z_100"]["test_bce"],
            "learned_z_current_training_seconds": current["learned"],
            "sixteen_fixed_z_bce": previous["aggregate"]["sixteen_fixed_z_150"]["test_bce"],
            "sixteen_fixed_z_current_training_seconds": current["frozen_bank"],
            "single_no_z_bce": previous["aggregate"]["no_z_150"]["test_bce"],
            "best_no_z_minus_learned_z_bce": paired_stats(best_minus_learned),
            "best_no_z_minus_single_no_z_bce": paired_stats(best_minus_single),
        },
        "per_seed": rows,
    }


def plot(summary: dict, path: Path) -> None:
    rows = summary["per_seed"]
    seeds = [row["seed"] for row in rows]
    previous = json.loads(PREVIOUS.read_text())
    baseline = {row["seed"]: row for row in previous["per_seed"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    ax = axes[0]
    for label, values, color in (
        ("обучаемый z, 16 кодов", [baseline[s]["learned_z_100"]["test_bce"] for s in seeds], "#2466A8"),
        ("без z, один запуск", [baseline[s]["no_z_150"]["test_bce"] for s in seeds], "#928f8c"),
        ("без z, лучший из 16", [row["selected_test_bce"] for row in rows], "#DC6B28"),
    ):
        ax.plot(seeds, values, "o-", label=label, color=color, linewidth=1.8)
    ax.set(xlabel="Seed данных и обучения", ylabel="BCE на тесте (меньше — лучше)",
           title="Выбор запуска без z по валидации")
    ax.set_xticks(seeds)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    counts = [1, 2, 4, 8, 16]
    means = [mean(row["test_bce_by_number_of_starts"][count - 1] for row in rows)
             for count in counts]
    ax.plot(counts, means, "o-", color="#DC6B28", linewidth=2, label="без z, лучший по валидации")
    ax.axhline(previous["aggregate"]["learned_z_100"]["test_bce"]["mean"],
               color="#2466A8", linestyle="--", label="обучаемый z, 16 кодов")
    ax.axhline(previous["aggregate"]["sixteen_fixed_z_150"]["test_bce"]["mean"],
               color="#278569", linestyle=":", label="16 фиксированных кодов z")
    ax.set(xlabel="Независимые запуски без z", ylabel="Средняя BCE на тесте",
           title="Каждый запуск без z обучается отдельно")
    ax.set_xticks(counts)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--output-root", type=Path, default=OUTPUTS)
    run_parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 4, 5, 6, 7])
    run_parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    run_parser.add_argument("--restarts", type=int, default=RESTARTS)
    baseline_parser = sub.add_parser("baselines")
    baseline_parser.add_argument("--output-root", type=Path, default=CURRENT_BASELINES)
    baseline_parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 4, 5, 6, 7])
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--output-root", type=Path, default=OUTPUTS)
    report_parser.add_argument("--summary", type=Path, default=SUMMARY)
    report_parser.add_argument("--figure", type=Path, default=FIGURE)
    args = parser.parse_args()
    if args.command == "run":
        run_all(args.output_root, tuple(args.gpus), tuple(args.seeds), args.restarts)
    elif args.command == "baselines":
        run_current_baselines(args.output_root, tuple(args.gpus))
    else:
        summary = summarize(args.output_root)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
        plot(summary, args.figure)
        print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
