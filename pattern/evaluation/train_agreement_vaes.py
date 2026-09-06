"""Train the two frozen VAE decoders used by the decoder-agreement experiment.

This trainer intentionally reads only the train patterns in an existing OOD
split.  It has no dependency on held-out pattern data, gold masks, or plots.

Example:
    CUDA_VISIBLE_DEVICES=2 python pattern/evaluation/train_agreement_vaes.py \
        --device cuda
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from models.cvae import CVAE, load_importance_maps, make_importance_loss, make_loaders  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT = ROOT / "outputs/ood/split_seed_42/split.json"
DEFAULT_OUT = ROOT / "outputs/decoder_agreement/seed_20260906"
MAP_SPLIT_SEED = 42
VAL_FRACTION = config.CVAE_VAL_FRACTION
BATCH_SIZE = 128
LR = 1e-3
BETA = 0.1
LOSS = "bce"
REDUCTION = "sum"
TOP_FRAC = 0.1
LATENT_DIM = 32
HIDDEN = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model: torch.nn.Module) -> dict:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def mean_history(total: float, recon: float, kl: float, n: int) -> dict:
    return {"total": total / n, "recon": recon / n, "kl": kl / n}


def check_complete(out_dir: Path, expected: dict) -> dict | None:
    """Return completed metadata only when it exactly matches this protocol."""
    ckpt = out_dir / "cvae_best.pt"
    meta_path = out_dir / "cvae_meta.pt"
    if not ckpt.exists() and not meta_path.exists():
        return None
    if not (ckpt.exists() and meta_path.exists()):
        raise FileExistsError(
            f"Partial output in {out_dir}; refusing to overwrite it. "
            "Remove it deliberately before restarting.")
    meta = torch.load(meta_path, weights_only=True, map_location="cpu")
    required = ["patterns", "seed", "map_split_seed", "latent_dim", "hidden",
                "beta", "loss", "reduction", "top_frac", "importance_name",
                "epochs", "dataset_files_sha256", "partition_sha256"]
    if not all(key in meta for key in required):
        raise FileExistsError(f"Existing metadata in {out_dir} is incomplete; refusing overwrite.")
    mismatches = {key: (meta.get(key), value) for key, value in expected.items()
                  if meta.get(key) != value}
    if mismatches:
        raise FileExistsError(
            f"Existing result in {out_dir} has a different protocol: {mismatches}")
    actual_hash = sha256_file(ckpt)
    if meta.get("checkpoint_sha256") != actual_hash:
        raise FileExistsError(f"Checkpoint hash mismatch in {out_dir}; refusing overwrite.")
    return meta


def noncollapse_diagnostics(model: CVAE, x: torch.Tensor, y: torch.Tensor,
                            device: torch.device, seed: int) -> dict:
    """Latent and decoder-variation diagnostics; no gold or task metrics."""
    model.eval()
    with torch.no_grad():
        n_posterior = min(512, x.size(0))
        xp, yp = x[:n_posterior].to(device), y[:n_posterior].to(device)
        c = model.condition(yp)
        mu, logvar = model.encode(xp, c)
        generator = torch.Generator(device=device).manual_seed(seed + 10_000)
        z = torch.randn(256, model.latent_dim, device=device, generator=generator)
        c_prior = torch.empty(256, 0, device=device)
        probabilities = model.importance(model.decode(z, c_prior))
        first = probabilities[0]
        pair_mse = (probabilities[1:] - first).pow(2).mean(dim=1)
        return {
            "posterior_examples": int(n_posterior),
            "posterior_mu_feature_std_mean": float(mu.std(dim=0).mean().cpu()),
            "posterior_mu_feature_abs_mean": float(mu.abs().mean().cpu()),
            "posterior_logvar_mean": float(logvar.mean().cpu()),
            "prior_samples": 256,
            "decoder_probability_feature_std_mean": float(probabilities.std(dim=0).mean().cpu()),
            "decoder_probability_feature_std_max": float(probabilities.std(dim=0).max().cpu()),
            "decoder_probability_global_std": float(probabilities.std().cpu()),
            "decoder_probability_mean": float(probabilities.mean().cpu()),
            "decoder_probability_min": float(probabilities.min().cpu()),
            "decoder_probability_max": float(probabilities.max().cpu()),
            "prior_vs_first_probability_mse_mean": float(pair_mse.mean().cpu()),
        }


def train_one(seed: int, x: torch.Tensor, y: torch.Tensor, expected: dict,
              out_dir: Path, device: torch.device) -> dict:
    result_dir = out_dir / f"vae_{seed}"
    complete = check_complete(result_dir, expected)
    if complete is not None:
        print(f"[agreement-vae] seed={seed}: verified existing checkpoint; skipping", flush=True)
        return {"seed": seed, "status": "verified_existing", "best_val": complete["best_val_loss"],
                "seconds": 0.0, "checkpoint_sha256": complete["checkpoint_sha256"]}

    result_dir.mkdir(parents=True, exist_ok=False)
    configure_seed(seed)
    # make_loaders uses an explicit generator, so both models get exactly the
    # same map partition and batch order, independent of the model seed.
    train_loader, val_loader, train_idx, val_idx = make_loaders(
        x, y, VAL_FRACTION, BATCH_SIZE, MAP_SPLIT_SEED)
    partition = {
        "map_split_seed": MAP_SPLIT_SEED,
        "val_fraction": VAL_FRACTION,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "loader_order_seed": MAP_SPLIT_SEED,
    }
    partition_hash = sha256_json(partition)
    if partition_hash != expected["partition_sha256"]:
        raise RuntimeError("Internal map partition changed between VAE seeds")

    model = CVAE(config.MASK_DIM, LATENT_DIM, HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = make_importance_loss(LOSS, REDUCTION)
    best_val = float("inf")
    best_epoch = 0
    train_history, val_history = [], []
    started = time.time()
    for epoch in range(1, expected["epochs"] + 1):
        model.train()
        sums = [0.0, 0.0, 0.0]
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, mu, logvar = model(xb, yb)
            total, recon, kl = loss_fn(logits, xb, mu, logvar, BETA)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for index, value in enumerate((total, recon, kl)):
                sums[index] += value.item() * xb.size(0)
        train_metrics = mean_history(*sums, len(train_loader.dataset))
        train_history.append(train_metrics)

        model.eval()
        sums = [0.0, 0.0, 0.0]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, mu, logvar = model(xb, yb)
                total, recon, kl = loss_fn(logits, xb, mu, logvar, BETA)
                for index, value in enumerate((total, recon, kl)):
                    sums[index] += value.item() * xb.size(0)
        val_metrics = mean_history(*sums, len(val_loader.dataset))
        val_history.append(val_metrics)
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            torch.save(cpu_state_dict(model), result_dir / "cvae_best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == expected["epochs"]:
            print(f"[agreement-vae] seed={seed} epoch={epoch:3d}/{expected['epochs']} "
                  f"train={train_metrics['total']:.4f} "
                  f"val={val_metrics['total']:.4f} "
                  f"recon={val_metrics['recon']:.4f} kl={val_metrics['kl']:.4f}", flush=True)

    model.load_state_dict(torch.load(result_dir / "cvae_best.pt", weights_only=True,
                                     map_location=device))
    diagnostics = noncollapse_diagnostics(model, x, y, device, seed)
    checkpoint_hash = sha256_file(result_dir / "cvae_best.pt")
    elapsed = time.time() - started
    meta = {
        **expected,
        "n_maps": int(x.size(0)),
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "per_pattern": expected["per_pattern"],
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "final_train": train_history[-1],
        "final_val": val_history[-1],
        "train_history": train_history,
        "val_history": val_history,
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "noncollapse_diagnostics": diagnostics,
    }
    torch.save(meta, result_dir / "cvae_meta.pt")
    with (result_dir / "trainer_summary.json").open("w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[agreement-vae] seed={seed}: best epoch={best_epoch} val={best_val:.4f}; "
          f"{elapsed:.1f}s; checkpoint_sha256={checkpoint_hash}", flush=True)
    return {"seed": seed, "status": "trained", "best_val": best_val,
            "best_epoch": best_epoch, "seconds": elapsed,
            "checkpoint_sha256": checkpoint_hash, "noncollapse": diagnostics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds contains duplicates")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(2)
    split = json.loads(args.split.read_text())
    patterns = split["train_patterns"]
    held_out = set(split["test_patterns"])
    if len(patterns) != 12 or set(patterns) & held_out:
        raise ValueError("Split must contain 12 disjoint train patterns")
    # Only the listed train patterns are supplied to this loader.
    x, y, per_pattern = load_importance_maps(patterns, config.CKPT_DIR,
                                             importance_name="importance.pt",
                                             top_frac=TOP_FRAC)
    dataset_files = {pat: sha256_file(config.pattern_dir(pat) / "importance.pt")
                     for pat in patterns}
    _, _, train_idx, val_idx = make_loaders(x, y, VAL_FRACTION, BATCH_SIZE, MAP_SPLIT_SEED)
    partition = {"map_split_seed": MAP_SPLIT_SEED, "val_fraction": VAL_FRACTION,
                 "train_indices": train_idx.tolist(), "val_indices": val_idx.tolist(),
                 "loader_order_seed": MAP_SPLIT_SEED}
    expected_base = {
        "patterns": patterns,
        "split_source": str(args.split),
        "split_sha256": sha256_file(args.split),
        "map_split_seed": MAP_SPLIT_SEED,
        "loader_order_seed": MAP_SPLIT_SEED,
        "partition_sha256": sha256_json(partition),
        "dataset_files_sha256": dataset_files,
        "dataset_selected_tensor_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        "latent_dim": LATENT_DIM,
        "hidden": HIDDEN,
        "beta": BETA,
        "loss": LOSS,
        "reduction": REDUCTION,
        "top_frac": TOP_FRAC,
        "importance_name": "importance.pt",
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "val_fraction": VAL_FRACTION,
        "per_pattern": per_pattern,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_log = args.out_dir / "train.log"
    summary_path = args.out_dir / "training.json"
    started = time.time()
    print(f"[agreement-vae] device={device}; CUDA_VISIBLE_DEVICES="
          f"{__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}; "
          f"patterns={patterns}; maps={x.size(0)}", flush=True)
    results = []
    for seed in args.seeds:
        expected = {**expected_base, "seed": seed}
        results.append(train_one(seed, x, y, expected, args.out_dir, device))
    summary = {"status": "complete", "device": str(device), "seeds": args.seeds,
               "elapsed_seconds": time.time() - started, "results": results,
               "shared_provenance": expected_base}
    if summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old.get("shared_provenance") != expected_base:
            raise FileExistsError(f"Existing {summary_path} has different provenance; refusing overwrite.")
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with run_log.open("a") as fh:
        fh.write(json.dumps({"event": "complete", "summary": summary}, sort_keys=True) + "\n")
    print(f"[agreement-vae] completed in {summary['elapsed_seconds']:.1f}s -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
