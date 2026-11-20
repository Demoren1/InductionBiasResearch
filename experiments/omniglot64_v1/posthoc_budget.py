"""Predict from Omniglot supports. Only the scoring module opens target answers."""

import argparse
import json
import multiprocessing as mp

import numpy as np
import torch

from . import posthoc_model as model
from .posthoc_source import PLAN, ROOT, WIDTHS, D, H, sha, write_json

BATCH = 16
CURVE = (0, 10, 25, 50, 100, 150, 200, 300, 450, 600)


def config(family, width, blocks, lr, step):
    return f"{family}.{width}.{blocks}.{round(lr * 1000)}.{step}"


def unpack(name):
    family, width, blocks, rate, step = name.split(".")
    return family, int(width), int(blocks), int(rate) / 1000, int(step)


class Run:
    def __init__(self, stage, shot, offset, gpu, bank, choices):
        self.stage, self.shot, self.offset = stage, shot, offset
        self.device, self.bank, self.choices = f"cuda:{gpu}", bank, choices
        folder = ROOT / stage / str(shot)
        sx, sy, qx = (
            torch.tensor(
                np.load(folder / f"{key}.npy", mmap_mode="r")[offset : offset + BATCH],
                device=self.device,
            )
            for key in ("sx", "sy", "qx")
        )
        self.sx, self.sy, self.qx = sx.float() / 255, sy.long(), qx.float() / 255
        self.dense_x = model.inputs(self.sx), model.inputs(self.qx)
        self.mapping_cache = {}
        self.out = {}
        self.seed = {"teacher_dev": 118113, "policy_dev": 183619, "test": 223427}[
            stage
        ] + 997 * offset

    def compact(self, width, blocks, shuffle=None):
        key = width, blocks, shuffle
        if key not in self.mapping_cache:
            labels = self.bank[f"w{width}_b{blocks}"]
            if shuffle is not None:
                labels = labels[
                    :, np.random.default_rng(390001 + shuffle).permutation(D)
                ]
            mapping = model.indicators(labels, self.device)
            self.mapping_cache[key] = (
                mapping,
                (model.inputs(self.sx, mapping), model.inputs(self.qx, mapping)),
                labels,
            )
        return self.mapping_cache[key]

    @torch.no_grad()
    def record(self, key, state, inputs, labels=None, deploy=False):
        sx, qx = inputs
        logits = model.predict(qx, state)
        self.out[key] = logits.cpu().numpy()
        if self.stage != "test":
            support = model.predict(sx, state)
            ce = -support.log_softmax(-1).gather(-1, self.sy[..., None]).mean(1)[:, 0]
            recall = (support.argmax(-1) == self.sy).float().mean(1)
            self.out[key + "_support"] = torch.stack((ce, recall), 1).cpu().numpy()
        if deploy and self.stage == "test" and self.offset == 0:
            path = ROOT / "models" / f"{self.shot}shot_{key}.npz"
            arrays = {
                name: value[0].cpu().numpy()
                for name, value in zip(("w", "b", "v", "a"), state)
            }
            if labels is not None:
                arrays["groups"] = labels
            np.savez(path, **arrays)

    def scratch(self, width, blocks, groups, salt):
        return model.start(
            len(self.sx), blocks, groups, width, self.seed + salt, self.device
        )

    def sweep(self, family, width, blocks, initial, inputs, steps, target=None):
        for rate in PLAN["lr"]:
            checkpoints = [s for s in CURVE if s <= max(steps)]

            def save(step, state, rate=rate):
                self.record(config(family, width, blocks, rate, step), state, inputs)

            model.fit(initial, inputs[0], self.sy, rate, checkpoints, save, target)

    def fixed(
        self, key, chosen, initial, inputs, target=None, labels=None, deploy=False
    ):
        _, _, _, lr, steps = unpack(chosen)
        state = model.fit(
            initial, inputs[0], self.sy, lr, [steps], lambda *_: None, target
        )
        self.record(key, state, inputs, labels, deploy)
        return state

    def process(self):
        dense = self.dense_x
        teacher_start = self.scratch(H, 1, D, H)
        if self.stage == "teacher_dev":
            self.sweep("teacher", H, 1, teacher_start, dense, PLAN["teacher_steps"])
            return
        teacher = self.fixed(
            "teacher", self.choices["teacher"], teacher_start, dense, deploy=True
        )
        target = model.predict(dense[0], teacher)
        aligned = model.align_teacher(
            teacher, torch.tensor(self.bank["anchor"], device=self.device)
        )
        for width in WIDTHS:
            groups = PLAN["budgets"][str(width)] // H
            narrow = self.scratch(width, 1, D, width)
            pruned = model.repair(
                model.prune(teacher, dense[0], width), dense[0], target
            )
            if self.stage == "policy_dev":
                self.sweep("narrow", width, 1, narrow, dense, PLAN["scratch_steps"])
                self.sweep("prune", width, 1, pruned, dense, PLAN["warm_steps"], target)
                for blocks in (1, 4):
                    mapping, inputs, _ = self.compact(width, blocks)
                    raw = model.project(teacher if blocks == 1 else aligned, mapping)
                    warm = model.repair(raw, inputs[0], target)
                    fresh = self.scratch(H, blocks, groups, 3000 + width)
                    self.sweep(
                        "asrr", width, blocks, warm, inputs, PLAN["warm_steps"], target
                    )
                    self.sweep(
                        "fresh", width, blocks, fresh, inputs, PLAN["scratch_steps"]
                    )
            else:
                plan = self.choices[str(width)]
                self.fixed(
                    f"narrow_{width}", plan["narrow"], narrow, dense, deploy=True
                )
                self.fixed(
                    f"prune_{width}", plan["prune"], pruned, dense, target, deploy=True
                )
                _, _, blocks, _, _ = unpack(plan["fresh"])
                _, inputs, labels = self.compact(width, blocks)
                fresh = self.scratch(H, blocks, groups, 3000 + width)
                self.fixed(f"fresh_{width}", plan["fresh"], fresh, inputs)
                _, _, blocks, _, _ = unpack(plan["asrr"])
                if plan["fresh_same_index"] == plan["fresh"]:
                    self.out[f"fresh_same_{width}"] = self.out[f"fresh_{width}"]
                else:
                    _, inputs, _ = self.compact(width, blocks)
                    fresh = self.scratch(H, blocks, groups, 3000 + width)
                    self.fixed(
                        f"fresh_same_{width}", plan["fresh_same_index"], fresh, inputs
                    )
                for shuffle in (None, 0, 1, 2, 3):
                    mapping, inputs, labels = self.compact(width, blocks, shuffle)
                    raw = model.project(teacher if blocks == 1 else aligned, mapping)
                    key = (
                        f"asrr_{width}"
                        if shuffle is None
                        else f"shuffle_{width}_{shuffle}"
                    )
                    if shuffle is None:
                        self.record(f"raw_{width}", raw, inputs)
                    warm = model.repair(raw, inputs[0], target)
                    self.fixed(
                        key,
                        plan["asrr"],
                        warm,
                        inputs,
                        target,
                        labels,
                        deploy=shuffle is None,
                    )


def worker(args):
    gpu, stage, jobs, choices = args
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    bank = {}
    if stage != "teacher_dev":
        with np.load(ROOT / "bank.npz") as data:
            bank = dict(data)
    for shot, offset in jobs:
        out = ROOT / stage / str(shot) / f"pred_{offset:04d}.npz"
        run = Run(stage, shot, offset, gpu, bank, choices.get(str(shot), {}))
        run.process()
        np.savez(out, **run.out)
    return len(jobs)


def run(stage, gpus):
    name = {
        "teacher_dev": None,
        "policy_dev": "teacher_frozen.json",
        "test": "policy_frozen.json",
    }[stage]
    choices = json.loads((ROOT / name).read_text())["choices"] if name else {}
    if stage == "test":
        (ROOT / "models").mkdir(exist_ok=True)
    jobs = []
    for shot in PLAN["target_shots"]:
        folder = ROOT / stage / str(shot)
        count = len(np.load(folder / "sy.npy", mmap_mode="r"))
        jobs += [
            (shot, i)
            for i in range(0, count, BATCH)
            if not (folder / f"pred_{i:04d}.npz").exists()
        ]
    with mp.get_context("spawn").Pool(len(gpus)) as pool:
        pool.map(
            worker,
            [(gpu, stage, jobs[i :: len(gpus)], choices) for i, gpu in enumerate(gpus)],
        )
    paths = sorted((ROOT / stage).glob("*/pred_*.npz"))
    write_json(
        ROOT / stage / "predictions_sealed.json",
        {
            "config_sha256": sha(ROOT / name)
            if name
            else sha(ROOT / "registration.json"),
            "bank_sha256": sha(ROOT / "bank.npz") if name else None,
            "files": {str(p.relative_to(ROOT / stage)): sha(p) for p in paths},
        },
    )
    print(stage, "sealed", len(paths), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("teacher_dev", "policy_dev", "test"))
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    args = parser.parse_args()
    run(args.stage, args.gpus)
