"""Output-only experimental driver; private generation stays in oracle workers."""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.darts_structured_tasks_v1.evaluate import align_indices

from .blind_contract import digest, save_json
from .blind_relu import (
    averaging_budget,
    estimate_noise,
    orient_planes,
    recover_planes,
    scale_quantizations,
)
from .shared_structure_data import teacher as synthetic_teacher

BASE = Path(__file__).resolve().parent


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    for name in ("private", "public", "source", "code_snapshot", "logs"):
        (root / name).mkdir()
    cases = [
        {
            "name": "pattern_0110",
            "kind": "pattern",
            "inputs": 11,
            "hidden": 8,
            "codes": 2,
            "outputs": 1,
        },
        {
            "name": "deepsets",
            "kind": "sets",
            "inputs": 16,
            "hidden": 16,
            "codes": 4,
            "outputs": 2,
        },
        {
            "name": "general_random",
            "kind": "synthetic",
            "inputs": 16,
            "hidden": 16,
            "codes": 4,
            "outputs": 1,
        },
    ]
    plan = {
        "stage": "G33 exploratory removal of teacher priors",
        "registered_utc": datetime.now(UTC).isoformat(),
        "seed": 26110633041,
        "sources": 4,
        "cases": cases,
        "private_noise_conditions": [0.0, 0.2],
        "learner_allowlist": ["inputs", "hidden", "codes", "outputs"],
        "learner_interface": "Only sizes and independent repeated output observations. No b, sigma, teacher I/C/W2, roles, edges, coefficient range or generation seed.",
        "bias_model": "Separate learned bias per hidden neuron and output offsets. No common-b constraint in learner.",
        "observation_precision": "Gaussian observation means are simulated exactly in distribution; every constituent repeat is counted. Noise estimate uses32 independent raw repeats at32 random inputs.",
        "selection": "Exploratory pilot; truth is used only after saved learner output, never by blind_relu. Freeze a new independent cohort after modifying the method.",
        "resource_policy": "CPU for small linear algebra and output queries; GPU2 only if packed nonlinear fits are justified. Never GPU6/7; do not touch foreign jobs.",
    }
    for number, case in enumerate(cases):
        rng = np.random.default_rng(plan["seed"] + 100003 * number)
        d, h, k, o = [case[name] for name in ("inputs", "hidden", "codes", "outputs")]
        index = np.zeros((d, h), dtype=np.int16)
        if case["kind"] == "pattern":
            for neuron in range(h):
                index[neuron : neuron + 4, neuron] = [-1, 2, 2, -1]
            code = rng.uniform(0.8, 1.3, (plan["sources"], k))
            bias = np.full((plan["sources"], h), -3.0)
        elif case["kind"] == "sets":
            order = rng.permutation(d)
            for point in range(8):
                index[
                    np.ix_(order[2 * point : 2 * point + 2], [2 * point, 2 * point + 1])
                ] = [[1, 2], [3, 4]]
            code = rng.normal(size=(plan["sources"], k))
            bias = np.full((plan["sources"], h), 0.15)
        else:
            index, code = synthetic_teacher(
                {"seed": plan["seed"], "sources": plan["sources"], "codes": k},
                {"family": 33921, "variant": "random"},
            )
            bias = np.full((plan["sources"], h), 0.15)
        index = index[:, rng.permutation(h)]
        readout = rng.uniform(0.5, 1.5, (plan["sources"], h, o)) * rng.choice(
            (-1, 1), (plan["sources"], h, o)
        )
        offset = rng.normal(0, 0.3, (plan["sources"], o))
        np.savez_compressed(
            root / "private" / f"{case['name']}.npz",
            index=index,
            codes=code,
            bias=bias,
            readout=readout,
            offset=offset,
        )
        save_json(
            root / "public" / f"{case['name']}.json",
            {key: case[key] for key in plan["learner_allowlist"]},
        )
    for name in ("blind_relu.py", "blind_study.py"):
        shutil.copy2(BASE / name, root / "code_snapshot" / name)
    plan["code_sha256"] = {
        name: digest(BASE / name) for name in ("blind_relu.py", "blind_study.py")
    }
    save_json(root / "protocol.json", plan)
    print(json.dumps({"prepared": str(root), "cases": len(cases)}), flush=True)


def oracle_worker(connection, private_path, task, sigma, seed):
    with np.load(private_path) as raw:
        index, code = raw["index"], raw["codes"][task]
        weight = np.sign(index) * np.r_[0, code][np.abs(index)]
        bias, readout, offset = (
            raw["bias"][task],
            raw["readout"][task],
            raw["offset"][task],
        )
    rng = np.random.default_rng(seed)
    while True:
        message = connection.recv()
        if message is None:
            return
        x, repeats, replicas = message
        clean = (
            np.maximum(x @ weight + bias, 0) @ readout / math.sqrt(weight.shape[1])
            + offset
        )
        response = clean[None] + rng.normal(
            size=(replicas, *clean.shape)
        ) * sigma / math.sqrt(repeats)
        connection.send(response)


class OutputObservations:
    def __init__(self, connection):
        self.connection = connection
        self.requests = []
        self.output_count = 0

    def __call__(self, x, *, repeats, replicas):
        x = np.asarray(x, dtype=np.float64)
        if repeats < 1 or replicas < 1 or not np.isfinite(x).all():
            raise ValueError("Invalid output request")
        self.connection.send((x, repeats, replicas))
        result = self.connection.recv()
        self.output_count += len(x) * repeats * replicas
        self.requests.append(
            {
                "points": len(x),
                "repeats": repeats,
                "replicas": replicas,
                "x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            }
        )
        return result


def learn_task(observe, sizes, seed, rays, precision_factor=1):
    rng = np.random.default_rng(seed)
    variance, signal, noise_trace = estimate_noise(observe, sizes["inputs"], rng)
    planes, plane_trace = recover_planes(
        observe,
        sizes,
        variance,
        signal,
        seed + 31,
        rays=rays,
        precision_factor=precision_factor,
    )
    if not planes:
        return {}, {"complete": False, "noise": noise_trace, "planes": plane_trace}
    count = 4096
    x = (
        rng.normal(size=(count, sizes["inputs"]))
        * np.resize(np.array([0.5, 1, 2, 4]), count)[:, None]
    )
    repeats = averaging_budget(variance, math.sqrt(signal) * 1e-3, 2**20)
    y = observe(x, repeats=repeats, replicas=1)[0]
    normal, bias, coefficient, orientation = orient_planes(planes, x, y)
    uncertainty = np.array([row["uncertainty"] for row in planes])
    candidates = scale_quantizations(normal, uncertainty, sizes["codes"], seed + 73)
    arrays = {
        "normal": normal,
        "bias": bias,
        "coefficient": coefficient,
        "fit_x": x,
        "fit_y": y,
        "uncertainty": uncertainty,
    }
    for number, candidate in enumerate(candidates):
        index = candidate["index"]
        padded = np.zeros((sizes["inputs"], sizes["hidden"]), dtype=np.int16)
        padded[:, : index.shape[1]] = index
        arrays[f"index_{number}"] = padded
        arrays[f"codes_{number}"] = candidate["codes"]
        arrays[f"scales_{number}"] = candidate["scales"]
    return arrays, {
        "complete": len(planes) == sizes["hidden"] and bool(candidates),
        "noise": noise_trace,
        "planes": plane_trace,
        "recovered_planes": len(planes),
        "orientation": orientation,
        "candidates": [
            {
                "number": number,
                "projection_mse": c["projection_mse"],
                "groups": len(np.unique(np.abs(c["index"][c["index"] != 0]))),
            }
            for number, c in enumerate(candidates)
        ],
        "public_sizes": sizes,
        "output_observations": observe.output_count,
        "requests": observe.requests,
    }


def run(root, case_name, condition, task, rays, precision_factor=1, *, audit=True):
    plan = json.loads((root / "protocol.json").read_text())
    sizes = json.loads((root / "public" / f"{case_name}.json").read_text())
    if set(sizes) != set(plan["learner_allowlist"]):
        raise ValueError("Unexpected teacher metadata in public sizes")
    output = root / "source" / f"{case_name}_c{condition}_t{task}"
    if output.exists() and not (output / "frozen.json").exists():
        output.rename(output.with_name(output.name + "_failed_" + str(time.time_ns())))
    output.mkdir(exist_ok=False)
    (output / "executed_code_snapshot").mkdir()
    for name in ("blind_relu.py", "blind_study.py"):
        shutil.copy2(BASE / name, output / "executed_code_snapshot" / name)
    ctx = mp.get_context("spawn")
    client, server = ctx.Pipe()
    process = ctx.Process(
        target=oracle_worker,
        args=(
            server,
            root / "private" / f"{case_name}.npz",
            task,
            plan["private_noise_conditions"][condition],
            plan["seed"] + 1000003 * task + 73 * condition,
        ),
    )
    process.start()
    observe = OutputObservations(client)
    started = time.monotonic()
    try:
        arrays, trace = learn_task(
            observe, sizes, 330000 + task * 1009, rays, precision_factor
        )
    except Exception as error:
        save_json(
            output / "failure.json",
            {
                "error": repr(error),
                "output_observations": observe.output_count,
                "requests": observe.requests,
            },
        )
        raise
    finally:
        client.send(None)
        process.join(timeout=10)
    np.savez_compressed(output / "learned.npz", **arrays)
    trace["seconds"] = time.monotonic() - started
    trace["precision_factor"] = precision_factor
    trace["output_observations"] = observe.output_count
    trace["requests"] = observe.requests
    save_json(output / "trace.json", trace)
    save_json(
        output / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "learned_sha256": digest(output / "learned.npz"),
            "trace_sha256": digest(output / "trace.json"),
        },
    )
    if not audit:
        print(
            json.dumps(
                {
                    "case": case_name,
                    "condition": condition,
                    "task": task,
                    "complete": trace["complete"],
                    "seconds": trace["seconds"],
                    "output_observations": observe.output_count,
                }
            ),
            flush=True,
        )
        return
    with np.load(root / "private" / f"{case_name}.npz") as private:
        truth = private["index"]
    audit = []
    for name, index in arrays.items():
        if name.startswith("index_"):
            metrics, _, _ = align_indices(index, truth, sizes["codes"])
            audit.append({"candidate": name, **metrics})
    save_json(output / "post_freeze_audit.json", {"rows": audit})
    print(
        json.dumps(
            {
                "case": case_name,
                "condition": condition,
                "task": task,
                "complete": trace["complete"],
                "planes": trace.get("recovered_planes", 0),
                "best_structural_errors": min(
                    (r["categorical_errors"] for r in audit), default=None
                ),
                "seconds": trace["seconds"],
                "output_observations": observe.output_count,
            }
        ),
        flush=True,
    )


def batch_job(job):
    run(*job, audit=False)


def collect_panels(root, case_name, condition):
    plan = json.loads((root / "protocol.json").read_text())
    sizes = json.loads((root / "public" / f"{case_name}.json").read_text())
    panels = {name: {} for name in ("selection", "meta", "test")}
    requests = []
    for task in range(plan["sources"]):
        trace = json.loads(
            (
                root / "source" / f"{case_name}_c{condition}_t{task}" / "trace.json"
            ).read_text()
        )
        variance = np.array(trace["noise"]["variance_estimate"])
        signal = trace["noise"]["signal_variance_estimate"]
        repeats = averaging_budget(variance, math.sqrt(signal) * 1e-3, 2**20)
        rng = np.random.default_rng(33100007 + 1009 * task)
        ctx = mp.get_context("spawn")
        client, server = ctx.Pipe()
        process = ctx.Process(
            target=oracle_worker,
            args=(
                server,
                root / "private" / f"{case_name}.npz",
                task,
                plan["private_noise_conditions"][condition],
                plan["seed"] + 1000003 * task + 73 * condition + 99991,
            ),
        )
        process.start()
        observe = OutputObservations(client)
        try:
            for name, count in (("selection", 1024), ("meta", 2048), ("test", 4096)):
                x = rng.normal(size=(count, sizes["inputs"]))
                if name == "selection":
                    x *= np.resize(np.array([0.5, 1, 2, 4]), count)[:, None]
                y = observe(x, repeats=repeats, replicas=1)[0]
                panels[name][f"x_{task}"] = x
                panels[name][f"y_{task}"] = y
        finally:
            client.send(None)
            process.join(timeout=10)
        requests.append(
            {
                "task": task,
                "output_observations": observe.output_count,
                "requests": observe.requests,
            }
        )
    output = root / "public" / f"{case_name}_c{condition}"
    output.mkdir(exist_ok=False)
    for name, data in panels.items():
        np.savez_compressed(output / f"{name}.npz", **data)
    save_json(
        output / "observations.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "rows": requests,
            "sha256": {p.name: digest(p) for p in output.glob("*.npz")},
        },
    )
    print(json.dumps({"panels": str(output)}), flush=True)


def fixed_panels(root, case_name, condition, train_points):
    plan = json.loads((root / "protocol.json").read_text())
    sizes = json.loads((root / "public" / f"{case_name}.json").read_text())
    panels = {name: {} for name in ("fixed_train", "selection", "meta", "test")}
    requests = []
    for task in range(plan["sources"]):
        rng = np.random.default_rng(33500007 + 1009 * task)
        ctx = mp.get_context("spawn")
        client, server = ctx.Pipe()
        process = ctx.Process(
            target=oracle_worker,
            args=(
                server,
                root / "private" / f"{case_name}.npz",
                task,
                plan["private_noise_conditions"][condition],
                plan["seed"] + 1000003 * task + 73 * condition + 999983,
            ),
        )
        process.start()
        observe = OutputObservations(client)
        try:
            for name, count in (
                ("fixed_train", train_points),
                ("selection", 2048),
                ("meta", 4096),
                ("test", 8192),
            ):
                x = rng.normal(size=(count, sizes["inputs"]))
                if name in ("fixed_train", "selection"):
                    x *= np.resize(np.array([0.5, 1, 2, 4]), count)[:, None]
                panels[name][f"x_{task}"] = x
                panels[name][f"y_{task}"] = observe(x, repeats=1, replicas=1)[0]
        finally:
            client.send(None)
            process.join(timeout=10)
        requests.append(
            {
                "task": task,
                "output_observations": observe.output_count,
                "requests": observe.requests,
            }
        )
    output = root / "public" / f"{case_name}_c{condition}"
    output.mkdir(exist_ok=False)
    for name, data in panels.items():
        np.savez_compressed(output / f"{name}.npz", **data)
    save_json(
        output / "observations.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "mode": "fixed_examples_no_repeats_no_active_training_queries",
            "rows": requests,
            "sha256": {p.name: digest(p) for p in output.glob("*.npz")},
        },
    )
    print(
        json.dumps({"fixed_panels": str(output), "train_points": train_points}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "run", "batch", "panels", "fixed")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", default="pattern_0110")
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--rays", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--precision-factor", type=int, default=1)
    parser.add_argument("--train-points", type=int, default=16384)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root.resolve())
    elif args.action == "batch":
        plan = json.loads((args.root / "protocol.json").read_text())
        jobs = [
            (
                args.root.resolve(),
                case["name"],
                condition,
                task,
                args.rays,
                args.precision_factor,
            )
            for case in plan["cases"]
            for condition in range(len(plan["private_noise_conditions"]))
            for task in range(plan["sources"])
            if not (
                args.root
                / "source"
                / f"{case['name']}_c{condition}_t{task}"
                / "frozen.json"
            ).exists()
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(batch_job, jobs))
    elif args.action == "panels":
        collect_panels(args.root.resolve(), args.case, args.condition)
    elif args.action == "fixed":
        fixed_panels(args.root.resolve(), args.case, args.condition, args.train_points)
    else:
        run(
            args.root.resolve(),
            args.case,
            args.condition,
            args.task,
            args.rays,
            args.precision_factor,
        )


if __name__ == "__main__":
    main()
