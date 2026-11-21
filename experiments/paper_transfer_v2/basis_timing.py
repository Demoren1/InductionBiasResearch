"""Time the selected shared bases with source examples already prepared."""

import argparse
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.groups import PLAN as FAMILY_PLAN
from experiments.asrr_benchmark_families_v1.msr import (
    fit_source as fit_archived_msr,
)
from experiments.asrr_benchmark_families_v1.msr import source_panels
from experiments.asrr_transfer_refinement_v1.common import OUT as REFINEMENT
from experiments.asrr_transfer_refinement_v1.common import (
    SEED as REFINEMENT_SEED,
)
from experiments.asrr_transfer_refinement_v1.unified_global import HAR_LAYOUT
from experiments.asrr_universal_v1.fixed_blocks import (
    STAGE,
    basis_inputs,
    fit_groups,
)
from experiments.asrr_universal_v1.har import har_source
from experiments.asrr_universal_v1.run import classification_source
from experiments.asrr_universal_v1.structure import spectral_basis
from experiments.aug_miniimagenet_v1 import convergence as mini
from experiments.aug_miniimagenet_v1 import panels as mini_panels
from experiments.aug_miniimagenet_v1.study import parallel, source_arrays
from experiments.paper_structure_benchmark_v1.methods.real_competitors import (
    run as fit_har_basis,
)
from experiments.real_group_transfer_v1.optimization import Layout, fit_dense
from experiments.real_group_transfer_v1.study import SEED as FC100_SEED
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .bilevel import stratified_sources, train_msr, train_yeh
from .common import CONFIG, OUT, SEED
from .source import dense_source, source_tensors

RESULTS = Path("artifacts/paper_basis_timings_2026-11-21")
LOCAL = RESULTS.parent / "local" / RESULTS.name
HAR_ROOT = LOCAL.parent / "paper_asrr_2026-11-17/har"
HAR_ARCHIVE = RESULTS.parent / "paper_recovery_extension_2026-11-14/har"
MINI_NAME = "aug_miniimagenet"
DATASETS = (*CONFIG["real"], "har", MINI_NAME)
METHODS = ("asrr", "yeh_soft", "msr")
MINI_POLICIES = mini.RESULTS / "test/primary_choices.json"


def elapsed(device, operation):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = operation()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def dataset_spec(dataset):
    if dataset == MINI_NAME:
        return mini_panels.LAYOUT, mini_panels.SHOT, mini_panels.CODES
    if dataset == "har":
        return HAR_LAYOUT, 3, 128
    spec = CONFIG["real"][dataset]
    return Layout(*spec["layout"]), spec["shot"], spec["codes"]


def time_bilevel(dataset, method, device):
    layout, shot, codes = dataset_spec(dataset)
    rank = codes // layout.hidden
    if dataset == MINI_NAME:
        path = MINI_POLICIES
        candidate = json.loads(path.read_text())[method]["candidate"]
        support, _, views, labels = source_arrays(device)
        data = (
            support,
            torch.cat(tuple(views.values()), 1),
            labels.repeat(1, 2),
        )
    else:
        path = OUT / dataset / f"{method}_choice.json"
        candidate = Path(json.loads(path.read_text())["model"]).stem
        data = None

    warm = False
    if candidate.startswith("archive_b"):
        match = re.fullmatch(r"archive_b(\d+)_r(\d+)", candidate)
        factors, rank = (int(value) for value in match.groups())
        panel = source_panels(dataset, device, layout)
        _, seconds = elapsed(
            device,
            lambda: fit_archived_msr(
                dataset, factors, rank, device, panel=panel
            ),
        )
    else:
        if data is None:
            data = stratified_sources(dataset, device, layout)
        if method == "msr":
            match = re.fullmatch(
                r"(?:msr_)?f(\d+)_rate([.\d]+)(?:_seed(\d+))?_step(\d+)",
                candidate,
            )
            factors, rate, seed, steps = match.groups()
            _, seconds = elapsed(
                device,
                lambda: next(
                    train_msr(
                        data,
                        layout,
                        int(factors),
                        rank,
                        float(rate),
                        shot,
                        device,
                        checkpoints=(int(steps),),
                        seed_offset=int(seed or 0),
                    )
                ),
            )
        else:
            match = re.fullmatch(
                r"yeh_soft_(warm[01]|dense|cold)_rate([.\d]+)_step(\d+)",
                candidate,
            )
            init, rate, steps = match.groups()
            warm = init in ("warm1", "dense")
            exemplar = None
            if warm:
                if dataset == MINI_NAME:
                    fits = read_data(mini_panels.WORK / "dense_sources.npz")
                    params, ce = fits["parameters"], fits["ce"]
                else:
                    params, ce = dense_source(dataset, device)
                exemplar = params[ce[:, 0].argmin(), 0, : layout.connections]
            groups = layout.sites if layout.sites > 1 else 16
            _, seconds = elapsed(
                device,
                lambda: next(
                    train_yeh(
                        data,
                        layout,
                        exemplar,
                        groups,
                        codes,
                        float(rate),
                        warm,
                        shot,
                        device,
                        checkpoints=(int(steps),),
                    )
                ),
            )
    return {
        "candidate": candidate,
        "basis_seconds": seconds,
        "choice_sha256": digest(path),
        "uses_dense_source": warm,
    }


def time_har_bilevel(method, device):
    native = "yeh" if method == "yeh_soft" else "msr"
    selected = (
        HAR_ARCHIVE / "source_selection" / f"{method}_k128/selected.json"
    )
    registered = (
        HAR_ARCHIVE / "source_provenance" / f"{native}_k128/registered.json"
    )
    choice = json.loads(selected.read_text())["candidate"]
    plan = json.loads(registered.read_text())
    settings = plan["settings"][int(choice.removeprefix("r"))]
    with tempfile.TemporaryDirectory(
        prefix=f"har_{method}_", dir=LOCAL
    ) as tmp:
        root = Path(tmp)
        for relative in (
            "protocol.json",
            "public/support.npz",
            "public/numerical.npz",
            "source/dense/models.npz",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HAR_ROOT / relative, path)
        _, seconds = elapsed(
            device,
            lambda: fit_har_basis(
                root,
                native,
                128,
                device,
                plan["steps"],
                settings=[settings],
                basis_path=root / "basis.npz",
                profile=False,
            ),
        )
    return {
        "candidate": choice,
        "settings": settings,
        "basis_seconds": seconds,
        "choice_sha256": digest(selected),
        "uses_dense_source": settings["initialization"] == "dense_adam",
    }


def time_dense_sources(dataset, layout, device):
    if dataset == "har":
        receipt = HAR_ROOT / "source/dense/polish_access_receipt.json"
        return json.loads(receipt.read_text())["seconds"]
    if dataset == MINI_NAME:
        ordered, sy, views, qy = source_arrays(device)
        panel = (
            ordered.flatten(1, 2),
            sy,
            torch.cat(tuple(views.values()), 1),
            qy.repeat(1, 2),
        )
        steps = mini_panels.PLAN["dense_source"]["steps"]
        restarts = mini_panels.PLAN["dense_source"]["restarts"]
        count, seed, stride = 8, mini_panels.SEED + 53, 0
    else:
        panel = source_tensors(dataset, device)
        if dataset == "mnist_torus":
            spec = CONFIG["mnist"]
            steps, restarts = spec["source_steps"], spec["source_restarts"]
            count, seed, stride = 1, SEED + 339, 0
        elif dataset == "fc100":
            steps, restarts = 800, 3
            count, seed, stride = 8, FC100_SEED, 0
        else:
            spec = FAMILY_PLAN[dataset]
            steps, restarts = spec["source_steps"], spec["source_restarts"]
            count, seed, stride = len(panel[0]), spec["seed"], 991
    batches = np.array_split(np.arange(len(panel[0])), count)

    def fit_all():
        for i, rows in enumerate(batches):
            fit_dense(
                tuple(value[rows] for value in panel),
                layout,
                steps=steps,
                restarts=restarts,
                device=device,
                seed=seed + stride * i,
            )

    return elapsed(device, fit_all)[1]


def time_asrr(dataset, device):
    layout, shot, codes = dataset_spec(dataset)
    rank = codes // layout.hidden
    folder = REFINEMENT / dataset / STAGE
    if dataset == MINI_NAME:
        policy = json.loads(MINI_POLICIES.read_text())["asrr"]
        candidate = policy["candidate"].removeprefix("asrr_")
        folder = mini_panels.WORK / "models/asrr"
    else:
        policy = json.loads((folder / "choices.json").read_text())
        row = (
            policy["selected"]
            if dataset == "har"
            else policy["family"][policy["selected"]]
        )
        candidate = row["candidate"]
    spec = json.loads((folder / "bank.json").read_text())[candidate]
    weight = spec["group_weight"]

    def build():
        if dataset == MINI_NAME:
            ordered, sy, queries, qy = source_arrays(device)
            saved = read_data(mini_panels.WORK / "dense_sources.npz")
            profiles = source_profiles(
                saved["parameters"][np.argsort(saved["ce"][:, 0])], layout
            )
            data = ordered.flatten(1, 2), sy, queries, qy
        elif dataset == "har":
            profiles, data, ordered = har_source(HAR_ROOT, device)
        else:
            profiles, data, ordered = classification_source(
                dataset, layout, device
            )
        dense, paired, guides, allocation = basis_inputs(
            profiles, data, ordered, layout, rank, shot
        )
        groups = fit_groups(dense, paired, rank, spec["gradient_weight"])
        guide = np.repeat(
            guides[spec["moment_weight"]], len(allocation), axis=0
        )
        basis = np.asarray(
            [
                spectral_basis(g, u, weight)
                for g, u in zip(groups, guide, strict=True)
            ],
            np.float32,
        )
        return basis, groups.astype(np.float32), guide.astype(np.float32)

    (basis, groups, guide), seconds = elapsed(device, build)
    frozen = read_data(spec["path"])["factor"]
    if not np.allclose(basis, frozen, atol=1e-5):
        left = basis @ basis.transpose(0, 2, 1)
        right = frozen @ frozen.transpose(0, 2, 1)
        if not np.allclose(left, right, atol=2e-5):
            raise ValueError(
                f"Basis replay changed the learned subspace on {dataset}"
            )

    def shuffle():
        seed = mini_panels.SEED if dataset == MINI_NAME else REFINEMENT_SEED
        for i in range(CONFIG["target"]["shuffle_replicates"]):
            rng = np.random.default_rng(seed + 621_173 + i)
            np.asarray(
                [
                    spectral_basis(g[rng.permutation(len(g))], u, weight)
                    for g, u in zip(groups, guide, strict=True)
                ],
                np.float32,
            )

    _, shuffle_seconds = elapsed(device, shuffle)
    return {
        "candidate": candidate,
        "basis_seconds": seconds,
        "dense_seconds": time_dense_sources(dataset, layout, device),
        "shuffle_seconds": shuffle_seconds,
        "uses_dense_source": True,
        "model_sha256": digest(spec["path"]),
    }


def measure(job):
    dataset, method, gpu = job
    path = RESULTS / f"{dataset}_{method}.json"
    if path.exists():
        return
    device = mini_panels.device_for(gpu)
    if method == "asrr":
        row = time_asrr(dataset, device)
    elif dataset == "har":
        row = time_har_bilevel(method, device)
    else:
        row = time_bilevel(dataset, method, device)
    record = {
        "dataset": dataset,
        "method": method,
        "device": torch.cuda.get_device_name(device),
        **row,
    }
    save_json(path, record)
    print(record, flush=True)


def run(jobs, gpus):
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    jobs = [
        (d, m) for d, m in jobs if not (RESULTS / f"{d}_{m}.json").exists()
    ]
    for start in range(0, len(jobs), len(gpus)):
        batch = jobs[start : start + len(gpus)]
        parallel(
            measure, [(*job, gpu) for job, gpu in zip(batch, gpus)], len(batch)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=METHODS
    )
    args = parser.parse_args()
    run([(d, m) for d in args.datasets for m in args.methods], args.gpus)
