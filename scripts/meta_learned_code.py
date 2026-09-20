"""Learn a four-dimensional code for each known meta_pattern length.

The generated objects, task splits, adaptation procedure, and checkpoint
criterion match meta_pattern. The only architecture change is replacing its
observed scalar length input with four trainable codes shared by tasks of the
same known length. This script does not use held-out lengths 5 or 7 in training
or checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from meta_pattern.common import (dataset, save_checkpoint, seed_for, setup,
                                 source_hashes, task_splits, validation, write_json)
from meta_pattern.config import Config
from meta_pattern.models import _normalize_columns, adapt_v, forward_with_u
from meta_pattern.train import stable_clip_grad_norm_


TRAIN_LENGTHS = (3, 4, 6, 8)
LATENT_DIM = 4


class CodeFullUGenerator(nn.Module):
    """An unfactorized U decoder with a trainable code per known length."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        output_size = (config.seq_len + 1) * config.hidden * config.rank1 + (config.hidden + 1) * config.rank2
        layers: list[nn.Module] = [nn.Linear(LATENT_DIM, config.width), nn.SiLU()]
        for _ in range(1, config.generator_depth):
            layers.extend((nn.Linear(config.width, config.width), nn.SiLU()))
        layers.append(nn.Linear(config.width, output_size))
        self.network = nn.Sequential(*layers)
        initial = torch.zeros(len(TRAIN_LENGTHS), LATENT_DIM)
        initial[:, 0] = torch.tensor([(length - 5.5) / 2.5 for length in TRAIN_LENGTHS])
        noise = torch.randn(len(TRAIN_LENGTHS), LATENT_DIM) * 0.05
        noise[:, 0] = 0
        self.codes = nn.Parameter(initial + noise)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.shape != (LATENT_DIM,):
            raise ValueError(f"expected latent shape ({LATENT_DIM},), got {tuple(z.shape)}")
        output = self.network(z).reshape(-1)
        first_size = (self.config.seq_len + 1) * self.config.hidden * self.config.rank1
        u1 = output[:first_size].reshape(-1, self.config.rank1)
        u2 = output[first_size:].reshape(self.config.hidden + 1, self.config.rank2)
        return _normalize_columns(u1), _normalize_columns(u2)

    def forward(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if length not in TRAIN_LENGTHS:
            raise ValueError(f"length {length} has no learned code")
        return self.decode(self.codes[TRAIN_LENGTHS.index(length)])


def training_config(seed: int, *, outer_steps: int = 1000, inner_steps: int = 50,
                    support_size: int = 1024, query_size: int = 1024,
                    tasks_per_step: int = 4, validate_every: int = 100) -> Config:
    return Config(seed=seed, method="generator", train_lengths=TRAIN_LENGTHS,
                  all_unseen_patterns=True, data_seed=20260906,
                  width=64, generator_depth=2, rank1=16, rank2=4,
                  inner_optimizer="adam", inner_steps=inner_steps, inner_lr=0.1,
                  init_scale=0.1, outer_steps=outer_steps, tasks_per_step=tasks_per_step,
                  support_size=support_size, query_size=query_size, batch_size=min(128, support_size),
                  validate_every=validate_every, val_tasks_per_length=0)


def train(output: Path, seed: int, device: str, *, outer_steps: int = 1000,
          inner_steps: int = 50, support_size: int = 1024, query_size: int = 1024,
          tasks_per_step: int = 4, validate_every: int = 100) -> Path:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config = training_config(seed, outer_steps=outer_steps, inner_steps=inner_steps,
                             support_size=support_size, query_size=query_size,
                             tasks_per_step=tasks_per_step, validate_every=validate_every)
    setup(seed, device)
    model = CodeFullUGenerator(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.outer_lr)
    splits = task_splits(config)
    train_by_length = {length: [task for task in splits["train"] if task.length == length]
                       for length in TRAIN_LENGTHS}
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output / "protocol.json", {
        "config": config.to_dict(), "model": "CodeFullUGenerator", "latent_dim": LATENT_DIM,
        "initial_code": "normalized length in coordinate zero; N(0,0.05) in remaining coordinates",
        "train_lengths": list(TRAIN_LENGTHS), "held_lengths": [5, 7],
        "task_splits": {name: [task.pattern for task in tasks] for name, tasks in splits.items()},
        "base_source_sha256": source_hashes(), "script_sha256": script_hash,
        "device": device, "checkpoint_selection": "known-length validation BCE",
    })
    best = float("inf")
    started = time.monotonic()

    def checkpoint(step: int) -> None:
        nonlocal best
        model.eval()
        score, rows = validation(config, model, device)
        state = {"step": step, "config": config.to_dict(), "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "val_bce": score,
                 "script_sha256": script_hash}
        save_checkpoint(output / "latest.pt", state)
        if score < best:
            best = score
            save_checkpoint(output / "best.pt", state)
        with (output / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, "bce": score, "tasks": rows}) + "\n")
        print(f"VALIDATE seed={seed} step={step} bce={score:.6f} best={best:.6f}", flush=True)
        model.train()

    checkpoint(0)
    for step in range(1, config.outer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for slot in range(config.tasks_per_step):
            length = TRAIN_LENGTHS[((step - 1) * config.tasks_per_step + slot) % len(TRAIN_LENGTHS)]
            episode_seed = seed_for("train", config.data_seed, step, slot)
            task = random.Random(episode_seed).choice(train_by_length[length])
            support = dataset(config, task, config.support_size, seed_for(episode_seed, "support"),
                              "support", device)
            query = dataset(config, task, config.query_size, seed_for(episode_seed, "query"),
                            "query", device)
            u = model(length)
            v = adapt_v(u, support["x"], support["y"], steps=config.inner_steps,
                        lr=config.inner_lr, seed=seed_for(episode_seed, "v"),
                        create_graph=True, batch_size=config.batch_size,
                        optimizer=config.inner_optimizer, init_scale=config.init_scale)
            loss = F.binary_cross_entropy_with_logits(
                forward_with_u(query["x"], u, v, seq_len=config.seq_len, hidden=config.hidden),
                query["y"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite meta loss at step {step}")
            (loss / config.tasks_per_step).backward()
            total += float(loss.detach()) / config.tasks_per_step
        norm = stable_clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, "query_bce": total,
                                     "gradient_norm_before_clip": norm}) + "\n")
        if step == 1 or step % 10 == 0:
            print(f"TRAIN seed={seed} step={step} query_bce={total:.6f} grad={norm:.6g}", flush=True)
        if step % config.validate_every == 0 or step == config.outer_steps:
            checkpoint(step)
    write_json(output / "done.json", {"best_val_bce": best,
                                      "seconds": time.monotonic() - started,
                                      "trained_steps": config.outer_steps})
    return output / "best.pt"


def load(checkpoint: Path, device: str) -> tuple[dict, Config, CodeFullUGenerator]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("script_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("learned-code script changed since training")
    config = Config(**state["config"])
    setup(config.seed, device)
    model = CodeFullUGenerator(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return state, config, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outer-steps", type=int, default=1000)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--support-size", type=int, default=1024)
    parser.add_argument("--query-size", type=int, default=1024)
    parser.add_argument("--tasks-per-step", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=100)
    args = parser.parse_args()
    train(args.out, args.seed, args.device, outer_steps=args.outer_steps,
          inner_steps=args.inner_steps, support_size=args.support_size,
          query_size=args.query_size, tasks_per_step=args.tasks_per_step,
          validate_every=args.validate_every)


if __name__ == "__main__":
    main()
