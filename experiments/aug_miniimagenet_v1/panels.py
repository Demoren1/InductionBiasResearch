"""Sample paired clean and transformed queries on the Ravi class split."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.msr_reproduction_v1.images_data import augment_query
from experiments.paper_transfer_v2.common import freeze
from experiments.real_group_transfer_v1.optimization import Layout
from experiments.sharing_structure_research_v1.blind_contract import digest

PLAN = json.loads(Path(__file__).with_name("protocol.json").read_text())
SEED = PLAN["seed"]
LAYOUT = Layout(*PLAN["layout"])
SHOT = PLAN["test"]["support_per_class"]
CODES = PLAN["coefficients"]
OUT = Path("artifacts/aug_miniimagenet_2026-11-21")
WORK = OUT.parent / "local" / OUT.name
IMAGES = Path(
    "artifacts/local/msr_reproduction_2026-11-14/images/miniimagenet"
)
SPLIT = {"source": "train", "dev": "val", "test": "test"}


def device_for(gpu):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    return device


@torch.no_grad()
def local_features(images, device, generator=None):
    """Pool each of the sixteen 21-by-21 RGB patches to 4-by-4 RGB."""
    result = []
    for chunk in np.array_split(images, (len(images) + 383) // 384):
        rgb = torch.as_tensor(chunk, dtype=torch.float32, device=device) / 255
        if generator is not None:
            rgb = augment_query(rgb, generator)
        patches = rgb.reshape(len(rgb), 3, 4, 21, 4, 21)
        patches = patches.permute(0, 2, 4, 1, 3, 5).flatten(0, 2)
        pooled = adaptive_avg_pool2d(patches, (4, 4)).permute(0, 2, 3, 1)
        features = pooled.reshape(len(rgb), LAYOUT.sites, LAYOUT.fan_in)
        result.append((features * 2 - 1).half().cpu().numpy())
    return np.concatenate(result)


def panel_job(job):
    phase, begin, count, gpu, directory, root_seed = job
    path = directory / f"inputs_{begin:04d}.npz"
    if path.exists():
        return path
    spec = PLAN[phase]
    support, query = spec["support_per_class"], spec["query_per_class"]
    data = np.load(IMAGES / f"{SPLIT[phase]}.npy", mmap_mode="r")
    seed = root_seed + 100_003 * list(SPLIT).index(phase) + 997 * begin
    rng = np.random.default_rng(seed)
    classes = np.stack(
        [
            rng.choice(len(data), LAYOUT.outputs, replace=False)
            for _ in range(count)
        ]
    )
    draws = np.stack(
        [
            rng.choice(data.shape[1], support + query, replace=False)
            for _ in range(count * LAYOUT.outputs)
        ]
    ).reshape(count, LAYOUT.outputs, -1)
    device = device_for(gpu)
    generator = torch.Generator(device=device).manual_seed(seed + 11)
    pixels = data[classes[..., None], draws[..., :support]].reshape(
        -1, 3, 84, 84
    )
    sx = local_features(pixels, device)
    pixels = data[classes[..., None], draws[..., support:]].reshape(
        -1, 3, 84, 84
    )
    clean = local_features(pixels, device)
    transformed = local_features(pixels, device, generator)
    sy = np.broadcast_to(
        np.arange(LAYOUT.outputs)[None, :, None],
        (count, LAYOUT.outputs, support),
    )
    atomic_npz(
        path.with_name(f"draws_{begin:04d}.npz"),
        classes=classes,
        draws=draws,
        seed=seed,
    )
    atomic_npz(
        path,
        sx=sx.reshape(
            count, LAYOUT.outputs, support, LAYOUT.sites, LAYOUT.fan_in
        ),
        sy=sy,
        qx=transformed.reshape(count, -1, LAYOUT.sites, LAYOUT.fan_in),
        qx_clean=clean.reshape(count, -1, LAYOUT.sites, LAYOUT.fan_in),
    )
    return path


def panel_shard(jobs):
    return [panel_job(job) for job in jobs]


def prepare(phase, gpus, *, work=WORK, output=OUT, seed=SEED, episodes=None):
    output.mkdir(parents=True, exist_ok=True)
    spec = PLAN[phase]
    count = spec["episodes"] if episodes is None else episodes
    freeze(
        output / "protocol.json",
        {**PLAN, "seed": seed, phase: {**spec, "episodes": count}},
    )
    directory = work / phase
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        return
    if phase == "test" and not (output / "choices.json").exists():
        raise ValueError(
            "Freeze all development choices before making test episodes"
        )
    batch = (
        max(1, count // len(gpus))
        if phase == "source"
        else PLAN["panel_batch"]
    )
    jobs = [
        (
            phase,
            start,
            min(batch, count - start),
            gpus[i % len(gpus)],
            directory,
            seed,
        )
        for i, start in enumerate(range(0, count, batch))
    ]
    workers = min(len(jobs), len(gpus))
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as pool:
        shards = pool.map(
            panel_shard, [jobs[i :: len(gpus)] for i in range(workers)]
        )
        all_paths = sorted(path for shard in shards for path in shard)
    query = PLAN[phase]["query_per_class"]
    qy = np.broadcast_to(
        np.repeat(np.arange(LAYOUT.outputs), query),
        (count, LAYOUT.outputs * query),
    )
    answers = directory / "answers.npz"
    atomic_npz(answers, qy=qy, units=np.arange(count))
    if phase == "source":
        arrays = [read_data(path) for path in all_paths]
        atomic_npz(
            work / "source.npz",
            qy=qy,
            **{
                key: np.concatenate([a[key] for a in arrays])
                for key in arrays[0]
            },
        )
    freeze(
        directory / "manifest.json",
        {
            "episodes": count,
            "raw_split": SPLIT[phase],
            "answer_sha256": digest(answers),
            "inputs": {path.name: digest(path) for path in all_paths},
        },
    )
    print(f"{phase}: prepared {count} paired episodes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phases", nargs="+", choices=SPLIT)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()
    for phase in args.phases:
        prepare(phase, args.gpus)
