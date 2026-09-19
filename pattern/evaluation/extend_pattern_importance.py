"""Train an additional, independent masked-MLP bank for one pattern.

Outputs are separate from the historical bank. Each chunk can be resumed and
contains normalized importance maps and the validation losses used for top-10%
selection. The training recipe matches pattern/models/train.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from evaluation.importance import importance_from  # noqa: E402
from models.mlp import BatchedMaskedMLP, generate_masks, get_train_batch  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(pattern: str, out: Path, n_extra: int, chunk_size: int,
        steps: int, device: torch.device) -> None:
    if pattern not in config.PATTERNS or n_extra <= 0 or chunk_size <= 0 or steps <= 0:
        raise ValueError("invalid pattern or bank size")
    out.mkdir(parents=True, exist_ok=True)
    validation_path = config.val_path(pattern)
    validation = torch.load(validation_path, map_location="cpu", weights_only=False)
    x_val, y_val = validation["x"].to(device), validation["y"].to(device)
    protocol = {
        "pattern": pattern, "n_extra": n_extra, "chunk_size": chunk_size,
        "steps": steps, "batch_size": config.TRAIN_BATCH_SIZE,
        "lr": config.LR, "mask_probability": config.P,
        "validation_sha256": digest(validation_path),
        "seed_rule": "10_000_000 + int(pattern,2)*100_000 + chunk_index*1009",
        "importance": "abs(w1*mask)/per-model max abs(w1*mask)",
    }
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise FileExistsError(f"different bank protocol at {protocol_path}")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    torch.set_num_threads(2)
    for offset in range(0, n_extra, chunk_size):
        chunk = offset // chunk_size
        count = min(chunk_size, n_extra - offset)
        path = out / f"chunk_{chunk:03d}.pt"
        seed = 10_000_000 + int(pattern, 2) * 100_000 + chunk * 1009
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=True)
            if (existing["seed"] != seed or existing["importance"].shape !=
                    (count, config.SEQ_LEN, config.H)):
                raise FileExistsError(f"unexpected existing bank chunk: {path}")
            print(f"[bank] {pattern} chunk {chunk}: verified existing", flush=True)
            continue
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        masks = generate_masks(count, config.SEQ_LEN, config.H, config.P, seed)
        model = BatchedMaskedMLP(count, config.SEQ_LEN, config.H).to(device)
        model.load_masks(masks)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
        start = time.time()
        for step in range(steps):
            xb, yb = get_train_batch(pattern, config.TRAIN_BATCH_SIZE, seed + step)
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.binary_cross_entropy_with_logits(pred, yb[:, None].expand_as(pred))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = model.val_loss(x_val, y_val, config.VAL_BATCH_SIZE).cpu()
            val_acc = model.val_acc(x_val, y_val, config.VAL_BATCH_SIZE).cpu()
            importance = importance_from(model.w1.detach().cpu(), masks)
        payload = {"pattern": pattern, "seed": seed, "steps": steps,
                   "importance": importance, "val_loss": val_loss,
                   "val_acc": val_acc}
        torch.save(payload, path)
        print(f"[bank] {pattern} chunk {chunk} n={count} "
              f"val_bce={val_loss.mean():.5f} elapsed={time.time()-start:.1f}s "
              f"-> {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-extra", type=int, default=6000)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    run(args.pattern, args.out, args.n_extra, args.chunk_size, args.steps,
        torch.device(args.device))


if __name__ == "__main__":
    main()
