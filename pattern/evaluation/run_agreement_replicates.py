"""Run decoder agreement for several independent pairs of pattern VAEs.

The model initialization is the only intended difference between pairs.  All
pairs use the same task split, map partition, loader order, latent starts, and
search hyperparameters.  Each pair is kept in a self-contained directory that
is compatible with ``train_agreement_vaes.py`` and
``run_decoder_agreement.py``.

The default ``train-search`` stage deliberately stops before held-out task
evaluation.  It trains and freezes the VAEs, performs label-free agreement,
then creates a post-hoc structural aggregate.  Use ``--stage all`` to also run
the considerably more expensive fresh-MLP evaluation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_ROOT = ROOT / "outputs/decoder_agreement/multiseed32_tuned_20260912"
DEFAULT_PAIRS = tuple((seed, seed + 1) for seed in range(186, 250, 2))
SPLIT_SOURCE = "pattern/outputs/ood/split_seed_42/split.json"


def parse_pair(value: str) -> tuple[int, int]:
    try:
        left, right = value.split(":", maxsplit=1)
        pair = int(left), int(right)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("pairs must have the form SEED1:SEED2") from exc
    if pair[0] == pair[1]:
        raise argparse.ArgumentTypeError("the two VAE seeds in a pair must differ")
    return pair


def pair_protocol(pair: tuple[int, int], search_seed: int, n_starts: int,
                  steps: int, epochs: int, lr: float = 0.03,
                  temperature: float = 0.5, latent_radius: float = 12.0) -> dict:
    return {
        "experiment": "replicated agreement of two independently trained frozen VAE decoders",
        "split_source": SPLIT_SOURCE,
        "model_seeds": list(pair),
        "map_split_seed": 42,
        "vae": {
            "epochs": epochs,
            "beta": 0.1,
            "loss": "bce",
            "reduction": "sum",
            "top_frac": 0.1,
            "importance_name": "importance.pt",
            "latent_dim": 32,
            "hidden": 256,
        },
        "search": {
            "seed": search_seed,
            "n_starts": n_starts,
            "steps": steps,
            "lr": lr,
            "temperature": temperature,
            "latent_radius": latent_radius,
            "k_active": 32,
            "objective": "mean squared soft-top-K mask distance after exact Hungarian column assignment",
            "selection": "minimum agreement objective per start including initialization; gold and task data excluded",
        },
        "controls": [
            "initial independent latents for each decoder",
            f"random paired search with {steps + 1} proposals per start and the same latent radius",
            "uniform exact-32 random masks",
            "ideal support: evaluation reference only",
        ],
        "evaluation": {
            "patterns": ["0100", "1011", "0000", "0011"],
            "steps": 2000,
            "batch_size": 128,
            "seed": 42,
            "n_masks_per_method": n_starts,
            "paired_initial_weights_and_batches": True,
            "selection_uses_evaluation": False,
        },
        "replicate_design": {
            "varied": "VAE model/training initialization seed",
            "held_fixed": [
                "OOD task split",
                "selected importance-map tensor",
                "map train/validation partition and loader order",
                "latent-pair starts and random-search proposals",
                "all optimization hyperparameters",
            ],
            "independence_unit": "VAE pair, not latent start",
        },
    }


def write_immutable_json(path: Path, payload: dict) -> None:
    if path.exists():
        current = json.loads(path.read_text())
        if current != payload:
            raise FileExistsError(f"{path} exists with a different protocol")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[replicates] " + " ".join(command), flush=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def search_is_complete(pair_root: Path) -> bool:
    required = ("optimization.pt", "random_search.pt", "masks.pt", "search_provenance.json")
    present = [(pair_root / name).exists() for name in required]
    if any(present) and not all(present):
        raise FileExistsError(f"partial agreement-search output in {pair_root}")
    if not all(present):
        return False
    provenance = json.loads((pair_root / "search_provenance.json").read_text())
    if sha256_file(pair_root / "protocol.json") != provenance.get("protocol_sha256"):
        raise ValueError(f"protocol hash mismatch in {pair_root}")
    if sha256_file(pair_root / "masks.pt") != provenance.get("mask_sha256"):
        raise ValueError(f"mask hash mismatch in {pair_root}")
    for model in provenance.get("models", []):
        checkpoint = Path(model["checkpoint"])
        if sha256_file(checkpoint) != model.get("sha256"):
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
    return True


def prepare_protocols(args: argparse.Namespace) -> list[Path]:
    if len({seed for pair in args.pairs for seed in pair}) != 2 * len(args.pairs):
        raise ValueError("VAE seeds must not be reused across replicate pairs")
    suite = {
        "experiment": "pattern decoder agreement across VAE initializations",
        "pairs": [list(pair) for pair in args.pairs],
        "pair_directories": [f"pair_{a}_{b}" for a, b in args.pairs],
        "search_seed": args.search_seed,
        "n_starts_per_pair": args.n_starts,
        "search_steps": args.steps,
        "search_lr": args.lr,
        "search_temperature": args.temperature,
        "search_latent_radius": args.latent_radius,
        "vae_epochs": args.epochs,
        "split_source": SPLIT_SOURCE,
        "independence_unit": "VAE pair",
        "posthoc_metrics": "hard agreement and ideal-support IoU are computed only after masks.pt is frozen",
    }
    write_immutable_json(args.out_dir / "suite_protocol.json", suite)
    roots = []
    for pair in args.pairs:
        pair_root = args.out_dir / f"pair_{pair[0]}_{pair[1]}"
        write_immutable_json(
            pair_root / "protocol.json",
            pair_protocol(pair, args.search_seed, args.n_starts, args.steps, args.epochs,
                          args.lr, args.temperature, args.latent_radius),
        )
        roots.append(pair_root)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pairs", type=parse_pair, nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--search_seed", type=int, default=20260912)
    parser.add_argument("--n_starts", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--latent_radius", type=float, default=12.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--gpus", nargs="+", default=None,
        help="physical GPU ids; replicate pairs run concurrently and are assigned round-robin",
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "train", "search", "evaluate", "pair-report", "aggregate", "train-search", "all"],
        default="train-search",
    )
    args = parser.parse_args()
    if (args.n_starts <= 0 or args.steps < 0 or args.epochs <= 0
            or args.lr <= 0 or args.temperature <= 0 or args.latent_radius <= 0):
        raise ValueError(
            "n_starts, epochs, lr, temperature, and latent_radius must be positive; "
            "steps must be nonnegative")
    args.out_dir = args.out_dir.resolve()
    roots = prepare_protocols(args)
    if args.stage == "prepare":
        return

    train_stages = {"train", "train-search", "all"}
    search_stages = {"search", "train-search", "all"}
    evaluate_stages = {"evaluate", "all"}
    pair_report_stages = {"evaluate", "pair-report", "all"}

    if args.device == "cpu" and args.gpus:
        raise ValueError("--gpus is only valid with --device cuda")
    gpu_ids = args.gpus or [None]

    def run_pair(pair: tuple[int, int], pair_root: Path, gpu_id: str | None) -> None:
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        env.setdefault("MPLCONFIGDIR", str(args.out_dir / ".matplotlib"))
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        if args.stage in train_stages:
            run_logged(
                [sys.executable, "evaluation/train_agreement_vaes.py",
                 "--out_dir", str(pair_root), "--seeds", str(pair[0]), str(pair[1]),
                 "--epochs", str(args.epochs), "--device", args.device],
                pair_root / "train.log", env,
            )
        if args.stage in search_stages:
            if search_is_complete(pair_root):
                print(f"[replicates] {pair_root.name}: verified complete search; skipping", flush=True)
            else:
                run_logged(
                    [sys.executable, "evaluation/run_decoder_agreement.py",
                     "--out_dir", str(pair_root), "--stage", "search", "--device", args.device],
                    pair_root / "search.log", env,
                )
        if args.stage in evaluate_stages:
            run_logged(
                [sys.executable, "evaluation/run_decoder_agreement.py",
                 "--out_dir", str(pair_root), "--stage", "evaluate", "--device", args.device],
                pair_root / "evaluate.log", env,
            )
        if args.stage in pair_report_stages:
            run_logged(
                [sys.executable, "evaluation/report_decoder_agreement.py",
                 "--out_dir", str(pair_root)],
                pair_root / "report.log", env,
            )

    assignments = [(pair, pair_root) for pair, pair_root in zip(args.pairs, roots)]
    if args.device == "cpu" or len(gpu_ids) == 1:
        for assignment in assignments:
            run_pair(*assignment, gpu_ids[0])
    else:
        print(f"[replicates] running {len(assignments)} pairs over {len(gpu_ids)} GPUs", flush=True)
        # One serial queue per physical GPU prevents a faster worker from
        # starting the next wave on a GPU whose previous pair is still active.
        queues = [assignments[index::len(gpu_ids)] for index in range(len(gpu_ids))]

        def run_gpu_queue(gpu_id: str | None, queue: list[tuple[tuple[int, int], Path]]) -> None:
            for pair, pair_root in queue:
                run_pair(pair, pair_root, gpu_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = [executor.submit(run_gpu_queue, gpu_id, queue)
                       for gpu_id, queue in zip(gpu_ids, queues) if queue]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    if args.stage in {"search", "pair-report", "aggregate", "train-search", "all", "evaluate"}:
        aggregate_env = os.environ.copy()
        aggregate_env.setdefault("MPLCONFIGDIR", str(args.out_dir / ".matplotlib"))
        run_logged(
            [sys.executable, "evaluation/report_agreement_replicates.py",
             "--root", str(args.out_dir)],
            args.out_dir / "aggregate.log", aggregate_env,
        )


if __name__ == "__main__":
    main()
