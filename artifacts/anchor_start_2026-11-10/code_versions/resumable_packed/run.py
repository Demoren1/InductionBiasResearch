"""Execute the paired anchor-start control with memory-bounded GPU batches."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    load_sizes,
    save_json,
)

PROJECT = Path(__file__).resolve().parents[2]
METHODS = ("full", "anchor_fixed", "anchor_free")
MODULE = "experiments.anchor_start_ablation_v1.run"


class ObservationBoundary:
    """Keep scratch-path private arrays sealed even outside the repository."""

    def __init__(self, base):
        self.base = base.resolve()
        self.role = "train"
        self.opened = set()
        self.violations = []

    def audit(self, event, arguments):
        if event != "open" or not arguments or isinstance(arguments[0], int):
            return
        try:
            path = Path(os.fsdecode(arguments[0])).resolve()
        except (TypeError, ValueError, OSError):
            return
        if path.is_relative_to(self.base):
            local = path.relative_to(self.base)
            forbidden = "private" in local.parts or path.name == "test.npz"
            forbidden |= path.name == "meta.npz" and self.role != "select"
            if forbidden:
                self.violations.append(str(path))
                raise PermissionError(f"Sealed observations: {path}")
            self.opened.add(str(path))
        elif path.is_relative_to(PROJECT / "artifacts"):
            self.violations.append(str(path))
            raise PermissionError(f"Historical audit data: {path}")


def bound_numerical_batches():
    """Split independent GN models without changing their update equations."""
    import torch

    from experiments.sharing_structure_research_v1 import (
        blind_dense_extract as extract,
        blind_dense_repair as repair,
        blind_joint_bank as joint,
        blind_numeric as numeric,
    )

    available = torch.cuda.mem_get_info()[0]
    if available >= 2 * 1024**3:
        size = 16
    elif available >= 1024**3:
        size = 8
    else:
        size = min(int(os.environ.get("IBR_GN_BATCH", "4")), 2)
    fit_size = 64 if available >= 1536 * 1024**2 else 32
    if available < 512 * 1024**2:
        fit_size = 8
    original_polish = extract.polish
    original_fit = numeric.fit_batch

    def polish(
        initial, x, y, vx, vy, dimensions, hidden, outputs, device, steps
    ):
        results = []
        for begin in range(0, len(initial), size):
            part = slice(begin, begin + size)
            results.append(
                original_polish(
                    initial[part],
                    x[part],
                    y[part],
                    vx[part],
                    vy[part],
                    dimensions,
                    hidden,
                    outputs,
                    device,
                    steps,
                )
            )
        return tuple(
            np.concatenate([r[k] for r in results], axis=0 if k < 2 else 1)
            for k in range(4)
        )

    def fit(indices, initial, x, y, vx, vy, codes, device, *, steps=32):
        results = []
        for begin in range(0, len(initial), fit_size):
            part = slice(begin, begin + fit_size)
            results.append(
                original_fit(
                    indices[part],
                    initial[part],
                    x[part],
                    y[part],
                    vx[part],
                    vy[part],
                    codes,
                    device,
                    steps=steps,
                )
            )
        trace = {
            key: np.concatenate([r[1][key] for r in results], axis=axis)
            for key, axis in (
                ("validation_mse", 0),
                ("selected_step", 0),
                ("train_history", 1),
                ("validation_history", 1),
            )
        }
        trace.update(
            seconds=sum(r[1]["seconds"] for r in results), device=str(device)
        )
        return np.concatenate([r[0] for r in results]), trace

    extract.polish = repair.polish = polish
    joint.fit_batch = fit
    return {
        "polish_models": size,
        "refit_models": fit_size,
        "free_memory_at_start_bytes": available,
    }


def source(root, case, condition, device):
    import torch
    from experiments.sharing_structure_research_v1 import (
        blind_dense,
        blind_dense_extract,
        blind_dense_repair,
        blind_joint_bank,
    )

    stages = (
        ("adam", lambda: blind_dense.run(root, case, condition, device, 2500)),
        (
            "polish",
            lambda: blind_dense_extract.run(
                root, case, condition, device, 32, polish_only=True
            ),
        ),
        (
            "revival",
            lambda: blind_dense_repair.run(root, case, condition, device, 2),
        ),
        (
            "descriptors",
            lambda: blind_dense_extract.run(
                root, case, condition, device, 0, from_repair=True
            ),
        ),
        (
            "anchors",
            lambda: blind_joint_bank.initialize_bank(
                root, case, condition, device
            ),
        ),
    )
    timing = {}
    for name, operation in stages:
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        timing[name] = time.perf_counter() - started
        print(json.dumps({"stage": name, "seconds": timing[name]}), flush=True)
    return {"stages": timing}


def propose(root, case, condition):
    from experiments.sharing_structure_research_v1 import (
        blind_joint_bank as joint,
    )
    from archive.experiments.g34_support_v1.bank import assign

    sizes = load_sizes(root, case)
    codes = sizes["codes"]
    key = f"{case}_c{condition}"
    base = dict(np.load(root / "fits" / key / "fitted.npz"))
    sources = [
        dict(np.load(root / "source" / f"{key}_t{t}" / "learned.npz"))
        for t in range(base["parameters"].shape[1])
    ]
    ranked = base["validation_mse"].mean(1).argsort()[:4]
    for method in METHODS:
        variant = root / "methods" / method
        folder = variant / "fits" / key
        folder.mkdir()
        shutil.copy2(root / "fits" / key / "fitted.npz", folder / "fitted.npz")
        started = time.perf_counter()
        if method == "full":
            calls = [0]
            operation = joint.joint_assign

            def counted(*args, **kwargs):
                calls[0] += 1
                return operation(*args, **kwargs)

            joint.joint_assign = counted
            try:
                joint.propose(variant, case, condition)
            finally:
                joint.joint_assign = operation
            record = json.loads((folder / "joint_bank.json").read_text())
            record["projection_calls"] = calls[0]
            record["method"] = method
            save_json(folder / "joint_bank.json", record)
            continue
        bank, origins = {}, {}
        for number in ranked:
            anchor = base["indices"][number]
            mappings = [
                joint.align_source(
                    anchor,
                    base["parameters"][number, t, :codes],
                    item["normal"],
                    item["uncertainty"],
                )
                for t, item in enumerate(sources)
            ]
            normal = np.stack(
                [s["normal"][:, m] for s, m in zip(sources, mappings)]
            )
            uncertainty = np.stack(
                [s["uncertainty"][m] for s, m in zip(sources, mappings)]
            )
            result = (
                joint.joint_assign(
                    anchor, anchor != 0, normal, uncertainty, codes
                )
                if method == "anchor_fixed"
                else assign(
                    anchor, anchor != 0, normal, uncertainty, codes, "free"
                )
            )
            if result is None:
                continue
            index, code, scales, score = result[:4]
            if not np.isfinite(score):
                continue
            for reference in range(len(sources)):
                oriented = (
                    index * np.where(scales[reference] >= 0, 1, -1)[None]
                )
                canonical = canonical_index(oriented.astype(np.int16), codes)
                identity = canonical.tobytes()
                if identity in bank and bank[identity][0] <= score:
                    continue
                bank[identity] = (score, canonical, np.abs(code))
                origins[identity] = {
                    "anchor": int(number),
                    "reference": reference,
                    "seed": "anchor",
                    "votes": 0,
                }
        keys = sorted(bank, key=lambda value: bank[value][0])
        np.savez_compressed(
            folder / "joint_bank.npz",
            indices=np.stack([bank[k][1] for k in keys])
            if keys
            else np.empty((0, *base["indices"].shape[1:]), dtype=np.int16),
            score=np.array([bank[k][0] for k in keys]),
            code_guesses=np.stack([bank[k][2] for k in keys])
            if keys
            else np.empty((0, len(sources), codes)),
        )
        save_json(
            folder / "joint_bank.json",
            {
                "method": method,
                "anchors": ranked.tolist(),
                "projection_calls": len(ranked),
                "starts_per_anchor": 1,
                "support_proposals_per_anchor": 1,
                "origins": [origins[k] for k in keys],
                "seconds": time.perf_counter() - started,
                "teacher_access": False,
            },
        )
    return {}


def fit(root, case, condition, method, device, boundary):
    from experiments.sharing_structure_research_v1 import (
        blind_joint_bank as joint,
    )

    variant = root / "methods" / method
    joint.fit(variant, case, condition, device)
    boundary.role = "select"
    joint.select(variant, case, condition)
    return {}


def target(root, case, cache, device):
    import torch
    from experiments.paper_structure_benchmark_v1.fitting.adaptation import (
        feature,
        fit_adam,
        signed_basis,
        values,
    )
    from experiments.paper_structure_benchmark_v1.fitting.numerical import (
        fit_features,
    )

    folder = root / "target_cache" / cache
    sizes = load_sizes(root, case)
    hidden, outputs, codes = [sizes[k] for k in ("hidden", "outputs", "codes")]
    index = np.load(folder / "index.npy")
    basis = signed_basis(index, codes, device)
    arrays = [
        dict(np.load(root / "target_data" / case / name))
        for name in ("support.npz", "selection.npz", "query_x.npz")
    ]
    support, validation, query = arrays
    x, y, vx, vy, qx = [
        torch.tensor(a, device=device, dtype=torch.float32)
        for a in (
            support["x"],
            support["y"],
            validation["x"],
            validation["y"],
            query["x"],
        )
    ]
    center = y.mean(1)
    scale = y.std((1, 2), correction=0).clamp_min(1e-8)
    y = (y - center[:, None]) / scale[:, None, None]
    vy = (vy - center[:, None]) / scale[:, None, None]
    initial, trace = fit_adam(
        x, y, vx, vy, basis.float(), hidden, outputs, seed=764025
    )
    adam_prediction = values(qx, basis.float(), initial[:1], hidden, outputs)[
        0
    ]
    keep, tasks = initial.shape[:2]
    feat = (
        feature(x.double(), basis)[None]
        .expand(keep, -1, -1, -1, -1)
        .flatten(0, 1)
    )
    vf = (
        feature(vx.double(), basis)[None]
        .expand(keep, -1, -1, -1, -1)
        .flatten(0, 1)
    )
    numerical, gn_trace = fit_features(
        feat,
        vf,
        initial.double().flatten(0, 1),
        y.double().repeat(keep, 1, 1),
        vy.double().repeat(keep, 1, 1),
        codes,
        steps=48,
    )
    losses = gn_trace["numerical_selection_mse"].reshape(keep, tasks)
    numerical = numerical.reshape(keep, tasks, -1)
    selected = numerical[losses.argmin(0), torch.arange(tasks, device=device)]
    prediction = values(qx.double(), basis, selected[None], hidden, outputs)[0]
    np.savez_compressed(
        folder / "predictions.npz",
        prediction=(prediction * scale[:, None, None] + center[:, None])
        .detach()
        .cpu()
        .numpy(),
        adam_prediction=(
            adam_prediction * scale[:, None, None] + center[:, None]
        )
        .detach()
        .cpu()
        .numpy(),
        parameters=selected.detach().cpu().numpy(),
        center=center.cpu().numpy(),
        scale=scale.cpu().numpy(),
    )
    np.savez_compressed(
        folder / "curves.npz",
        **{
            **{
                f"adam_{k}": v
                for k, v in trace.items()
                if isinstance(v, np.ndarray)
            },
            **{
                f"gn_{k}": v
                for k, v in gn_trace.items()
                if isinstance(v, np.ndarray)
            },
        },
    )
    return {"prediction_sha256": digest(folder / "predictions.npz")}


def worker(base, cohort, case, condition, phase, method, cache):
    runner_sha256 = digest(Path(__file__))
    import torch

    torch.set_num_threads(2)
    boundary = ObservationBoundary(base)
    sys.addaudithook(boundary.audit)
    root = base / cohort
    key = f"{case}_c{condition}_{phase}_{method or cache or 'shared'}"
    metric = root / "metrics" / f"{key}.json"
    started = time.perf_counter()
    device = torch.device("cuda:0") if phase != "propose" else None
    batching = None
    if device is not None:
        torch.zeros(1, device=device)
        torch.cuda.reset_peak_memory_stats()
        batching = bound_numerical_batches()
    status, details = "failed", {}
    try:
        if phase == "source":
            details = source(root, case, condition, device)
        elif phase == "propose":
            details = propose(root, case, condition)
        elif phase == "fit":
            details = fit(root, case, condition, method, device, boundary)
        else:
            details = target(root, case, cache, device)
        status = "complete"
    finally:
        if device is not None:
            torch.cuda.synchronize()
        record = {
            "status": status,
            "phase": phase,
            "method": method,
            "case": case,
            "condition": condition,
            "wall_seconds": time.perf_counter() - started,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "batching": batching,
            "runner_sha256": runner_sha256,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if device is not None
            else 0,
            "violations": boundary.violations,
            **details,
        }
        save_json(metric, record)
        save_json(
            root / "receipts" / f"{key}.json",
            {
                "status": status,
                "violations": boundary.violations,
                "opened": sorted(boundary.opened),
            },
        )
        print(json.dumps(record), flush=True)


def coordinate(base, *, resume=False):
    """Schedule this experiment's GPU and CPU stages, including foreign load."""
    jobs, live, done, failed = {}, {}, set(), {}
    retries = {}
    targets, target_map = {}, {}
    for cohort in ("standard", "general"):
        protocol = json.loads((base / cohort / "protocol.json").read_text())
        for case in protocol["cases"]:
            for condition in (0, 1):
                identity = f"{cohort}_{case['name']}_c{condition}"
                jobs[identity + "_source"] = {
                    "cohort": cohort,
                    "case": case["name"],
                    "condition": condition,
                    "phase": "source",
                    "method": "",
                    "cache": "",
                    "cpu": False,
                    "dependency": None,
                }
                jobs[identity + "_propose_"] = {
                    **jobs[identity + "_source"],
                    "phase": "propose",
                    "cpu": True,
                    "dependency": identity + "_source",
                }
                for method in METHODS:
                    jobs[identity + "_propose_fit_" + method] = {
                        **jobs[identity + "_source"],
                        "phase": "fit",
                        "method": method,
                        "dependency": identity + "_propose_",
                    }

    def register_target(identity, job):
        root = base / job["cohort"]
        key = f"{job['case']}_c{job['condition']}"
        selected = root / "methods" / job["method"] / "fits" / key
        index = np.load(selected / "joint_selected.npz")[
            "joint_functional_index"
        ].astype(np.int16)
        code = hashlib.sha256(index.tobytes()).hexdigest()[:20]
        cache = f"{job['case']}_{code}"
        target_map[identity] = {
            "cohort": job["cohort"],
            "case": job["case"],
            "condition": job["condition"],
            "method": job["method"],
            "cache": cache,
        }
        target_id = f"{job['cohort']}_{cache}"
        if target_id not in targets:
            folder = root / "target_cache" / cache
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / "index.npy").exists():
                np.save(folder / "index.npy", index)
            targets[target_id] = cache
            jobs[target_id] = {
                **job,
                "phase": "target",
                "cache": cache,
                "method": "",
                "dependency": identity,
            }

    def metric_path(job):
        suffix = job["method"] or job["cache"] or "shared"
        name = (
            f"{job['case']}_c{job['condition']}_{job['phase']}_{suffix}.json"
        )
        if job["phase"] == "target":
            existing = sorted(
                (base / job["cohort"] / "metrics").glob(
                    f"{job['case']}_c*_target_{job['cache']}.json"
                )
            )
            if existing:
                return existing[0]
        return base / job["cohort"] / "metrics" / name

    started = time.monotonic()
    if resume:
        previous = json.loads((base / "state.json").read_text())
        packing = json.loads((base / "repacking.json").read_text())
        parent_stat = Path(f"/proc/{packing['paused_controller_pid']}/stat")
        started = float(parent_stat.read_text().split()[21]) / os.sysconf(
            "SC_CLK_TCK"
        )
        retries = previous.get("retries", {})
        for identity, job in list(jobs.items()):
            path = metric_path(job)
            if (
                path.exists()
                and json.loads(path.read_text())["status"] == "complete"
            ):
                done.add(identity)
                if job["phase"] == "fit":
                    register_target(identity, job)
        for identity, entry in previous["live"].items():
            if identity in done or identity in packing["restart_jobs"]:
                continue
            live[identity] = {**entry, "process": None, "stream": None}
        for identity, job in list(jobs.items()):
            if job["phase"] != "target":
                continue
            path = metric_path(job)
            if (
                path.exists()
                and json.loads(path.read_text())["status"] == "complete"
            ):
                done.add(identity)
        print(
            json.dumps(
                {
                    "resumed": len(done),
                    "adopted": len(live),
                    "repacking": packing["restart_jobs"],
                }
            ),
            flush=True,
        )
    while len(done) + len(failed) < len(jobs) or live:
        for identity, entry in list(live.items()):
            if entry["process"] is None:
                process_stat = Path(f"/proc/{entry['pid']}/stat")
                try:
                    alive = process_stat.read_text().split()[2] != "Z"
                except FileNotFoundError:
                    alive = False
                if alive:
                    continue
                path = metric_path(jobs[identity])
                status = (
                    0
                    if path.exists()
                    and json.loads(path.read_text())["status"] == "complete"
                    else 1
                )
            else:
                status = entry["process"].poll()
            if status is None:
                continue
            if entry["stream"] is not None:
                entry["stream"].close()
            del live[identity]
            if status:
                failure_text = Path(entry["log"]).read_text()
                if "out of memory" in failure_text.lower() and not retries.get(
                    identity
                ):
                    retries[identity] = {
                        "gpu": entry["gpu"],
                        "log": entry["log"],
                    }
                    shutil.copy2(entry["log"], entry["log"] + ".oom")
                    job = jobs[identity]
                    if job["phase"] == "source":
                        root = base / job["cohort"]
                        key = f"{job['case']}_c{job['condition']}"
                        for folder in (
                            root / "dense" / key,
                            root / "fits" / key,
                            *(root / "source").glob(f"{key}_t*"),
                        ):
                            if folder.exists():
                                shutil.rmtree(folder)
                    continue
                failed[identity] = {"exit_code": status, "log": entry["log"]}
                continue
            done.add(identity)
            job = jobs[identity]
            if job["phase"] in ("source", "propose"):
                for method in [""] if job["phase"] == "source" else METHODS:
                    following = (
                        "propose" if job["phase"] == "source" else "fit"
                    )
                    name = (
                        identity.rsplit("_", 1)[0] + f"_{following}_{method}"
                    )
                    jobs[name] = {
                        **job,
                        "phase": following,
                        "method": method,
                        "cpu": following == "propose",
                        "dependency": identity,
                    }
            elif job["phase"] == "fit":
                register_target(identity, job)
        occupied = {
            gpu: sum(e["gpu"] == gpu for e in live.values())
            for gpu in range(8)
        }
        cpu_live = sum(e["gpu"] is None for e in live.values())
        gpu_free = {
            int(row.split(",")[0]): int(row.split(",")[1])
            for row in subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
        }
        for identity, job in list(jobs.items()):
            if identity in done or identity in failed or identity in live:
                continue
            if job["dependency"] and job["dependency"] not in done:
                continue
            if job["cpu"]:
                if cpu_live >= 6:
                    continue
                gpu = None
            else:
                limit = {gpu: (1 if gpu in (2, 3) else 2) for gpu in range(8)}
                choices = [
                    g
                    for g in range(8)
                    if occupied[g] < limit[g] and gpu_free[g] >= 1100
                ]
                if not choices:
                    continue
                gpu = min(choices, key=lambda g: occupied[g])
            log = base / "logs" / f"{identity}.log"
            log.parent.mkdir(exist_ok=True)
            stream = log.open("w")
            environment = os.environ.copy()
            environment.update(
                OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                IBR_GN_BATCH="2"
                if gpu in (2, 3) or identity in retries
                else "4",
                CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
            )
            arguments = [
                sys.executable,
                "-m",
                MODULE,
                "--base",
                str(base),
                "--phase",
                job["phase"],
                "--cohort",
                job["cohort"],
                "--case",
                job["case"],
                "--condition",
                str(job["condition"]),
            ]
            if job["method"]:
                arguments.extend(("--method", job["method"]))
            if job["cache"]:
                arguments.extend(("--cache", job["cache"]))
            process = subprocess.Popen(
                arguments,
                cwd=PROJECT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            live[identity] = {
                "process": process,
                "stream": stream,
                "gpu": gpu,
                "pid": process.pid,
                "log": str(log),
            }
            if gpu is None:
                cpu_live += 1
            else:
                occupied[gpu] += 1
                gpu_free[gpu] -= 1100
            print(
                json.dumps(
                    {"launched": identity, "gpu": gpu, "pid": process.pid}
                ),
                flush=True,
            )
        state = {
            "pid": os.getpid(),
            "seconds": time.monotonic() - started,
            "done": sorted(done),
            "failed": failed,
            "retries": retries,
            "total": len(jobs),
            "live": {
                k: {
                    f: v
                    for f, v in e.items()
                    if f not in ("process", "stream")
                }
                for k, e in live.items()
            },
        }
        save_json(base / "state.json", state)
        save_json(base / "target_map.json", target_map)
        if not live and failed:
            break
        time.sleep(5)
    if failed:
        raise RuntimeError(f"Failed jobs: {failed}")
    files = [
        *base.glob("*/methods/*/fits/*/joint_selected.npz"),
        *base.glob("*/target_cache/*/predictions.npz"),
    ]
    save_json(
        base / "frozen.json",
        {
            "files": {str(p.relative_to(base)): digest(p) for p in files},
            "source_choices_frozen": True,
            "target_predictions_frozen": True,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=(
            "prepare",
            "run",
            "resume",
            "source",
            "propose",
            "fit",
            "target",
            "report",
        ),
        required=True,
    )
    parser.add_argument("--cohort", default="standard")
    parser.add_argument("--case", default="case_00")
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--method", default="")
    parser.add_argument("--cache", default="")
    arguments = parser.parse_args()
    base = arguments.base.resolve()
    if arguments.phase == "prepare":
        from .prepare import setup

        setup(base)
    elif arguments.phase in ("run", "resume"):
        coordinate(base, resume=arguments.phase == "resume")
    elif arguments.phase == "report":
        from .report import report

        report(base)
    else:
        worker(
            base,
            arguments.cohort,
            arguments.case,
            arguments.condition,
            arguments.phase,
            arguments.method,
            arguments.cache,
        )


if __name__ == "__main__":
    main()
