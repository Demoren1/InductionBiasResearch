"""Reproducible, train-validation-only beta sweep for the motif-pair VAE/CVAE.

Each beta/seed run is isolated in its own output directory, so candidates can
be scheduled concurrently on user-specified idle GPUs without checkpoint races.
The selected beta is the *largest* value that passes the posterior-collapse
guard, not the one with the smallest reconstruction loss across beta values.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.cvae import read_split_provenance, task_ids  # noqa: E402


def _beta_name(beta: float) -> str:
    return f"beta_{beta:.8g}".replace("-", "m").replace(".", "p")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True,
                   help="current split.json; forwarded to every isolated training run")
    p.add_argument("--tasks", nargs="+", required=True, help="meta-train tasks only")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--betas", type=float, nargs="+", default=[1.0, .3, .1, .03, .01],
                   help="candidate values; they are canonicalized to decreasing order")
    p.add_argument("--variants", choices=("cvae", "vae"), nargs="+", default=["cvae"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--gpu_ids", type=str, nargs="+", required=True,
                   help="unique physical GPU ids; at most one isolated run is active per id")
    p.add_argument("--max_workers", type=int, default=None,
                   help="bounded parallel workers (default: number of --gpu_ids)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--top_frac", type=float, default=.1)
    p.add_argument("--importance_name", default="importance.pt")
    p.add_argument("--ckpt_root", type=Path, default=Path("outputs/checkpoints"))
    p.add_argument("--val_fraction", type=float, default=.15)
    p.add_argument("--active_kl_threshold", type=float, default=.01)
    p.add_argument("--active_mu_variance_threshold", type=float, default=1e-2)
    p.add_argument("--min_active_dims", type=int, default=2)
    p.add_argument("--min_val_kl", type=float, default=1.0)
    p.add_argument("--min_posterior_z0_recon_gap", type=float, default=0.0)
    p.add_argument("--tail_fraction", type=float, default=.10)
    p.add_argument("--tail_min_epochs", type=int, default=5)
    p.add_argument("--log_every", type=int, default=10)
    return p


def select_beta_run(runs: list[dict[str, Any]], *, expected_seeds: list[int] | None = None) -> dict[str, Any]:
    """Select largest guard-passing beta, then lowest internal val reconstruction.

    ``runs`` must contain only completed candidate summaries.  The function is
    pure to make the scientific selector easy to unit test.
    """
    # A lucky seed must not promote a beta whose other requested replicate
    # collapsed or failed.  Keeping ``None`` permissive preserves the small
    # library-call API used by existing callers; the production CLI always
    # passes its explicit seed list.
    required_seeds = None if expected_seeds is None else list(expected_seeds)
    if required_seeds is not None and (not required_seeds or len(set(required_seeds)) != len(required_seeds)):
        raise ValueError("expected_seeds must be a nonempty list of unique seeds")

    def stable(row: dict[str, Any]) -> bool:
        return (row.get("status") == "completed"
                and row.get("summary", {}).get("selection_status") == "eligible"
                and row.get("summary", {}).get("tail_stability", {}).get("stable") is True)

    by_beta: dict[float, list[dict[str, Any]]] = {}
    for row in runs:
        by_beta.setdefault(float(row["beta"]), []).append(row)
    eligible_by_beta: dict[float, list[dict[str, Any]]] = {}
    for beta, candidates in by_beta.items():
        if required_seeds is None:
            valid = [row for row in candidates if stable(row)]
            if valid:
                eligible_by_beta[beta] = valid
            continue
        by_seed = {int(row["seed"]): row for row in candidates}
        if len(by_seed) != len(candidates) or set(by_seed) != set(required_seeds):
            continue
        if all(stable(by_seed[seed]) for seed in required_seeds):
            eligible_by_beta[beta] = [by_seed[seed] for seed in required_seeds]
    policy = {
        "primary": "largest beta with a stable, guard-passing final training tail",
        "tie_break": "lowest internal-validation reconstruction, then lowest seed",
        "rationale": "minimizing reconstruction across beta values tends to choose beta near zero; "
                     "the largest stably non-collapsed beta retains the strongest usable prior; "
                     "initialization transients are explicitly rejected; every requested seed must pass",
        "ood_used": False,
        "required_seeds": required_seeds,
    }
    if not eligible_by_beta:
        return {
            "selection_status": "no_eligible_beta",
            "policy": policy,
            "reason": "No completed run had a fully guard-passing final tail; no checkpoint was promoted.",
            "candidates": runs,
        }
    selected_beta = max(eligible_by_beta)
    chosen = sorted(
        eligible_by_beta[selected_beta],
        key=lambda row: (float(row["summary"]["selected_val"]["recon"]), int(row["seed"])),
    )[0]
    return {
        "selection_status": "selected",
        "policy": policy,
        "selected": {
            "variant": chosen["variant"], "beta": chosen["beta"], "seed": chosen["seed"],
            "run_dir": chosen["run_dir"], "checkpoint": chosen["checkpoint"],
            "internal_val_recon": chosen["summary"]["selected_val"]["recon"],
            "internal_val_total": chosen["summary"]["selected_val"]["total"],
            "collapse_guard": chosen["summary"]["collapse_guard"],
            "tail_stability": chosen["summary"]["tail_stability"],
        },
        "candidates": runs,
    }


def _command(args, *, variant: str, beta: float, seed: int, run_dir: Path) -> list[str]:
    train = Path(__file__).with_name("train_cvae.py")
    return [
        sys.executable, str(train), "--split", str(args.split), "--tasks", *args.tasks,
        "--out_dir", str(run_dir),
        "--variant", variant, "--beta", str(beta), "--seed", str(seed),
        "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--lr", str(args.lr),
        "--latent_dim", str(args.latent_dim), "--hidden", str(args.hidden),
        "--top_frac", str(args.top_frac), "--importance_name", args.importance_name,
        "--ckpt_root", str(args.ckpt_root), "--val_fraction", str(args.val_fraction),
        "--active_kl_threshold", str(args.active_kl_threshold),
        "--active_mu_variance_threshold", str(args.active_mu_variance_threshold),
        "--min_active_dims", str(args.min_active_dims), "--min_val_kl", str(args.min_val_kl),
        "--min_posterior_z0_recon_gap", str(args.min_posterior_z0_recon_gap),
        "--tail_fraction", str(args.tail_fraction), "--tail_min_epochs", str(args.tail_min_epochs),
        "--log_every", str(args.log_every), "--device", "cuda",
    ]


def _run_candidates(args, variant: str, betas: list[float]) -> list[dict[str, Any]]:
    root = args.out_dir / variant
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(beta, seed, root / _beta_name(beta) / f"seed_{seed}")
            for beta in betas for seed in args.seeds]
    max_workers = args.max_workers or len(args.gpu_ids)
    if max_workers < 1 or max_workers > len(args.gpu_ids):
        raise ValueError("--max_workers must be in [1, number of unique --gpu_ids]")
    pending, running, records = list(jobs), [], []
    gpu_cursor = 0
    while pending or running:
        while pending and len(running) < max_workers:
            beta, seed, run_dir = pending.pop(0)
            run_dir.mkdir(parents=True, exist_ok=True)
            gpu_id = args.gpu_ids[gpu_cursor % len(args.gpu_ids)]
            gpu_cursor += 1
            log = (run_dir / "train.log").open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            process = subprocess.Popen(_command(args, variant=variant, beta=beta, seed=seed,
                                                run_dir=run_dir), stdout=log,
                                       stderr=subprocess.STDOUT, env=env)
            running.append((process, log, beta, seed, run_dir, gpu_id))
        process, log, beta, seed, run_dir, gpu_id = running.pop(0)
        returncode = process.wait()
        log.close()
        checkpoint = run_dir / "best.pt"
        selection = run_dir / "selection.json"
        record: dict[str, Any] = {
            "status": "completed" if returncode == 0 and selection.is_file() else "failed",
            "variant": variant, "beta": beta, "seed": seed, "gpu_id": gpu_id,
            "run_dir": str(run_dir.resolve()), "checkpoint": str(checkpoint.resolve()),
            "log": str((run_dir / "train.log").resolve()), "returncode": returncode,
        }
        if record["status"] == "completed":
            record["summary"] = json.loads(selection.read_text())
        else:
            record["reason"] = "train process failed; inspect train.log"
        records.append(record)
    return records


def main() -> None:
    args = parser().parse_args()
    if not args.betas or any(beta < 0 for beta in args.betas):
        raise ValueError("--betas must be a nonempty list of nonnegative values")
    if not 0 < args.top_frac <= 1 or not 0 < args.val_fraction < 1 or not 0 < args.tail_fraction <= 1:
        raise ValueError("--top_frac, --val_fraction, and --tail_fraction must lie in (0, 1)")
    if args.tail_min_epochs < 1:
        raise ValueError("--tail_min_epochs must be positive")
    if len(args.gpu_ids) != len(set(args.gpu_ids)):
        raise ValueError("--gpu_ids must not repeat a physical GPU")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds must not repeat a seed")
    args.split = args.split.resolve()
    args.split_provenance = read_split_provenance(args.split)
    if task_ids(args.tasks) != args.split_provenance["split_train_tasks"]:
        raise ValueError("--tasks must exactly equal --split train_tasks (including order)")
    betas = sorted(set(args.betas), reverse=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_selections = {}
    for variant in args.variants:
        records = _run_candidates(args, variant, betas)
        selection = select_beta_run(records, expected_seeds=args.seeds)
        selection.update({
            "variant": variant, "betas_evaluated_descending": betas,
            "seeds": args.seeds, "tasks": args.tasks, "top_frac": args.top_frac,
            "importance_name": args.importance_name,
            **args.split_provenance,
            "guard_thresholds": {
                "min_val_kl": args.min_val_kl,
                "min_active_dims": args.min_active_dims,
                "active_mu_variance_threshold": args.active_mu_variance_threshold,
                "min_posterior_z0_recon_gap": args.min_posterior_z0_recon_gap,
                "tail_fraction": args.tail_fraction,
                "tail_min_epochs": args.tail_min_epochs,
            },
        })
        target = args.out_dir / variant
        # This aggregate checkpoint belongs only to the current sweep.  Clear
        # it before handling an ineligible rerun so stale prior selections
        # cannot be consumed as though this sweep had promoted them.
        (target / "best.pt").unlink(missing_ok=True)
        (target / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        if selection["selection_status"] == "selected":
            source = Path(selection["selected"]["checkpoint"])
            shutil.copy2(source, target / "best.pt")
            print(f"[{variant}] selected beta={selection['selected']['beta']:g} -> {target / 'best.pt'}",
                  flush=True)
        else:
            print(f"[{variant}] no eligible beta; no checkpoint promoted", flush=True)
        all_selections[variant] = selection
    (args.out_dir / "selection.json").write_text(json.dumps(all_selections, indent=2) + "\n")
    if any(s["selection_status"] != "selected" for s in all_selections.values()):
        raise RuntimeError("at least one requested variant has no non-collapsed beta; see selection.json")


if __name__ == "__main__":
    main()
