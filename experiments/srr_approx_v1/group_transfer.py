"""Source-only group proposals and paired, fresh-code transfer measurements."""

import argparse
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.asrr_anchor_init_v1.transfer import public_panel
from experiments.omniglot64_v1.source import BANKS as OMNI_BANK
from experiments.omniglot64_v1.source import ROOT as OMNI_SOURCE
from experiments.paper_asrr_v1.extract import source_profiles
from experiments.paper_recovery_extension_v1.adapter import feature, fit_adam, values
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.core import canonical


OUT = Path("artifacts/asrr_group_recovery_2026-11-19")
LOCAL = OUT.parent / "local" / OUT.name
MNIST_SOURCE = OUT.parent / "local/paper_asrr_2026-11-17/mnist"
MNIST_BANK = OUT.parent / "local/asrr_tuning_2026-11-17/banks/mnist/models"
SETTINGS = {"mnist16": (16, 2048), "omniglot10": (64, 4096)}
SEED = 191127
SHARD = 24


def pivots(gram, count, rng):
    """Randomly pivoted Cholesky, sampling by unexplained source variance."""
    residual = gram.diagonal().copy()
    factor = np.zeros((len(gram), count))
    chosen = []
    for k in range(count):
        np.maximum(residual, 0, out=residual)
        residual[chosen] = 0
        if residual.sum() <= 1e-15:
            rest = np.setdiff1d(np.arange(len(gram)), chosen)
            chosen.extend(rng.choice(rest, count - k, replace=False).tolist())
            break
        i = rng.choice(len(gram), p=residual / residual.sum())
        factor[:, k] = (gram[:, i] - factor[:, :k] @ factor[i, :k]) / np.sqrt(
            residual[i]
        )
        residual -= factor[:, k] ** 2
        chosen.append(i)
    return np.array(chosen)


def representatives(covariance, profiles, codes, blend):
    """Propose per-neuron representatives and groups at three coverages."""
    _, inputs, hidden = profiles.shape
    count = codes // hidden
    rng = np.random.default_rng(SEED)
    views = {
        kind: np.zeros((inputs, hidden), dtype=np.int16)
        for kind in ("pooled", "partial", "isolated")
    }
    for h in range(hidden):
        z = profiles[:, :, h].T
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        unit = z / np.maximum(norm, 1e-12)
        gram = covariance * ((1 - blend) + blend * (unit @ unit.T))
        sd = np.sqrt(np.maximum(gram.diagonal(), 1e-16))
        selected = pivots(gram, count, rng)
        corr = gram[:, selected] / (sd[:, None] * sd[selected])
        nearest = corr.argmax(axis=1)
        confidence = corr[np.arange(inputs), nearest]
        nearest[selected] = np.arange(count)
        labels = h * count + nearest + 1
        views["pooled"][:, h] = labels
        views["partial"][:, h] = np.where(confidence >= 0.5, labels, 0)
        views["partial"][selected, h] = labels[selected]
        views["isolated"][selected, h] = labels[selected]
    return {k: canonical(v) for k, v in views.items()}


def prepare(setting):
    hidden, codes = SETTINGS[setting]
    origin = MNIST_SOURCE if setting == "mnist16" else OMNI_SOURCE
    if setting == "mnist16":
        with np.load(origin / "source/asrr/profiles.npz") as data:
            profiles = data["consensus"]
    else:
        _, profiles, _, _ = source_profiles(
            origin / "source/dense/models.npz", 784, hidden
        )
    with np.load(origin / "public/support.npz") as data:
        images = data["x"]
    # Equal task weight. The deterministic cap limits redundant MNIST examples.
    rng = np.random.default_rng(SEED)
    samples = rng.choice(images.shape[1], min(images.shape[1], 1000), replace=False)
    images = images[:, samples].astype(float) / 255
    images -= images.mean(axis=1, keepdims=True)
    flat = images.reshape(-1, images.shape[-1])
    covariance = flat.T @ flat / len(flat)
    bank = {}
    for name, blend in (("joint", 0.5), ("input", 0.0)):
        for key, value in representatives(covariance, profiles, codes, blend).items():
            bank[f"{name}_{key}"] = value
    if setting == "mnist16":
        refs = {"asrr": MNIST_BANK / "asrr_input_m2048.npz"}
    else:
        manifest = (
            OUT.parent / "omniglot64_gradient_profiles_2026-11-19/choices_frozen.json"
        )
        policies = read_json(manifest)["policies"]
        names = {
            "asrr": "four_partitions",
            "uniform": "balanced_random_mask_7611189541",
            "wsr": "wsr_step_75",
        }
        refs = {}
        for name, key in names.items():
            row = policies[key]
            path = OMNI_BANK / f"{row['name']}.npz"
            if digest(path) != row["model_sha256"]:
                raise ValueError(f"Archived model differs for {key}")
            refs[name] = path
    for key, path in refs.items():
        with np.load(path) as data:
            bank[key] = data["index"]
    for name, index in bank.items():
        occupied = np.unique(np.abs(index[index != 0]))
        if not np.array_equal(occupied, np.arange(1, codes + 1)):
            raise ValueError(f"{name}: unused code")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{setting}_indices.npz", **bank)
    save_json(
        OUT / f"{setting}_plan.json",
        {
            "eligible": [key for key in bank if key.startswith("joint_")],
            "controls": list(refs) + ["input_selected_coverage"],
            "selection": "highest mean development query accuracy, then lowest CE, then name",
            "development_seed": SEED + 10,
            "development_episodes": 96,
            "target_seeds": [SEED + 100 + i for i in range(8)],
            "target_episodes_per_seed": 96,
            "ref_files": {k: str(v) for k, v in refs.items()},
            "source_dense_sha256": digest(origin / "source/dense/models.npz"),
            "source_support_sha256": digest(origin / "public/support.npz"),
            "index_archive_sha256": digest(OUT / f"{setting}_indices.npz"),
        },
    )


def cached_basis(index, device):
    """Cache group sums, including fixed zero and cross-neuron shared codes."""
    inputs, hidden = index.shape
    partitions, keys, codes_by_block = [], [], []
    for column in index.T:
        active = column != 0
        ids, inverse = np.unique(abs(column[active]), return_inverse=True)
        first_seen, ordered = np.unique(inverse, return_index=True)
        by_first = first_seen[np.argsort(ordered)]
        labels = np.full(inputs, -1)
        labels[active] = np.argsort(by_first)[inverse]
        matrix = np.zeros((inputs, len(ids)), dtype=np.float32)
        matrix[np.flatnonzero(active), labels[active]] = np.sign(column[active])
        key = (labels.tobytes(), np.sign(column).tobytes())
        if not keys or keys[-1] != key:
            keys.append(key)
            partitions.append(matrix)
            codes_by_block.append([])
        codes_by_block[-1].append(ids[by_first] - 1)
    # Unequal consecutive blocks may always be expanded to one per neuron.
    lengths = [len(block) for block in codes_by_block]
    if len(set(lengths)) > 1:
        partitions = [p for p, n in zip(partitions, lengths) for _ in range(n)]
        codes_by_block = [[ids] for block in codes_by_block for ids in block]
    largest = max(p.shape[1] for p in partitions)
    groups = np.stack(
        [np.pad(p, ((0, 0), (0, largest - p.shape[1]))) for p in partitions], axis=1
    )
    lookup = np.stack(
        [
            np.pad(np.stack(ids, axis=-1), ((0, largest - len(ids[0])), (0, 0)))
            for ids in codes_by_block
        ]
    )
    return {
        "index": torch.as_tensor(index, device=device).long(),
        "scale": 1.0,
        "codes": int(abs(index).max()),
        "input_group_map": torch.as_tensor(groups, device=device),
        "input_group_codes": torch.as_tensor(lookup, device=device).long(),
    }


def predict(panel, index, setting, seed, device):
    hidden = index.shape[1]
    basis = cached_basis(index, device)
    if setting == "mnist16":
        settings = [{"lr": 0.003, "quantile": 0.9, "restart": i} for i in range(16)]
        options = dict(fixed_step=50, keep=16, preserve_setting_order=True)
    else:
        settings = [
            dict(lr=lr, quantile=q, restart=r)
            for lr, q, r in itertools.product(
                (0.001, 0.003, 0.01, 0.03), (0.1, 0.5, 0.9), range(2)
            )
        ]
        options = dict(steps=500, keep=4)
    predictions = []
    for begin in range(0, len(panel["train_x"]), SHARD):
        p = {
            k: torch.as_tensor(v[begin : begin + SHARD], device=device).float()
            for k, v in panel.items()
            if k != "query_y"
        }
        outputs = p["train_y"].shape[-1]
        params, _ = fit_adam(
            p["train_x"],
            p["train_y"],
            p.get("valid_x"),
            p.get("valid_y"),
            basis,
            hidden,
            outputs,
            loss="cross_entropy",
            seed=seed + begin,
            settings=settings,
            **options,
        )
        with torch.no_grad():
            logits = values(
                p["query_x"],
                basis,
                params,
                hidden,
                outputs,
                feature(p["query_x"], basis),
            )
            predictions.append(logits.softmax(-1).mean(0).cpu().numpy())
    return np.concatenate(predictions)


def metrics(probability, truth):
    accuracy = (probability.argmax(-1) == truth.argmax(-1)).mean(-1)
    ce = -(np.log(np.maximum(probability, 1e-30)) * truth).sum(-1).mean(-1)
    return np.stack([accuracy, ce], axis=1)


def read_json(path):
    return json.loads(path.read_text())


def shared_with_exceptions(base, residual, blocks, one_group, rng, weight):
    """Keep shared coarse codes and untie entries with balanced input coverage."""
    inputs, hidden = base.shape
    budget = len(np.unique(base))
    coarse = np.empty_like(base)
    used = 0
    for columns in np.array_split(np.arange(hidden), blocks):
        if one_group:
            labels = np.zeros(inputs, dtype=int)
        else:
            _, labels = np.unique(base[:, columns[0]], return_inverse=True)
        coarse[:, columns] = used + labels[:, None] + 1
        used += labels.max() + 1
    reserve = np.zeros_like(base, dtype=bool)
    priority = rng.random(base.shape)
    for label in range(1, used + 1):
        locations = np.flatnonzero(coarse == label)
        anchor = locations[np.argmin(priority.ravel()[locations])]
        reserve.flat[anchor] = True
    counts = np.full(hidden, (budget - used) // hidden)
    counts[rng.permutation(hidden)[: (budget - used) % hidden]] += 1
    coverage = np.zeros(inputs, dtype=int)
    noise = rng.gumbel(size=base.shape)
    for h in rng.permutation(hidden):
        # First spend codes on inputs with the lowest coverage. Residual source
        # disagreement is a proposal score within each equal-coverage tier.
        priority = noise[:, h] + weight * np.log1p(residual[:, h])
        available = np.flatnonzero(~reserve[:, h])
        order = np.lexsort((-priority[available], coverage[available]))
        take = available[order[: counts[h]]]
        coarse[take, h] = used + np.arange(1, len(take) + 1)
        coverage[take] += 1
        used += len(take)
    if used != budget or len(np.unique(coarse)) != budget:
        raise ValueError("Non-exhaustive exception allocation")
    return canonical(coarse)


def signed_candidates(setting):
    """Release uncertain sign constraints while preserving each code partition."""
    hidden, _ = SETTINGS[setting]
    archive = OUT / f"{setting}_indices.npz"
    with np.load(archive) as data:
        bank = dict(data)
    if setting == "mnist16":
        with np.load(MNIST_SOURCE / "source/asrr/profiles.npz") as data:
            profiles = data["consensus"]
    else:
        _, profiles, _, _ = source_profiles(
            OMNI_SOURCE / "source/dense/models.npz", 784, hidden
        )
    base = bank["asrr"]
    if np.any(base <= 0):
        raise ValueError("The reference must have unsigned input groups")
    residual = np.empty(base.shape)
    shared = np.empty(base.shape, dtype=bool)
    for h in range(hidden):
        _, labels, counts = np.unique(
            base[:, h], return_inverse=True, return_counts=True
        )
        z = profiles[:, :, h].T
        centers = np.zeros((len(counts), len(profiles)))
        np.add.at(centers, labels, z)
        error = np.sum((z - (centers / counts[:, None])[labels]) ** 2, axis=1)
        # Singleton signs are a code gauge, and contribute no group disagreement.
        residual[:, h] = error / max(np.mean(error), 1e-12)
        shared[:, h] = counts[labels] > 1
    rng = np.random.default_rng(SEED + 1)
    uniforms = rng.random(base.shape)
    for pct in (5, 15, 30, 50):
        flipped = (uniforms < np.minimum(0.5, pct / 100 * residual)) & shared
        uniform = np.zeros_like(flipped)
        for h in range(hidden):
            valid = np.flatnonzero(shared[:, h])
            order = np.argsort(uniforms[valid, h])
            uniform[valid[order[: flipped[:, h].sum()]], h] = True
        for name, changed in (("adaptive", flipped), ("iid", uniform)):
            bank[f"{name}_{pct}"] = canonical(base * np.where(changed, -1, 1))
    coarse_variants = (("root", 1, True), ("single", 1, False), ("four", 4, False))
    for kind, blocks, one_group in coarse_variants:
        for name, weight in (("shared", 1.0), ("uniform_shared", 0.0)):
            rng = np.random.default_rng(SEED + 2)
            bank[f"{name}_{kind}"] = shared_with_exceptions(
                base, residual, blocks, one_group, rng, weight
            )
    np.savez_compressed(OUT / f"{setting}_extended_indices.npz", **bank)
    initial = read_json(OUT / f"{setting}_plan.json")
    plan = dict(
        initial,
        eligible=[f"adaptive_{p}" for p in (5, 15, 30, 50)],
        initial_winner=read_json(OUT / f"{setting}_frozen.json")["winner"],
        ablation="same number of flipped nonsingletons per neuron, uniform placement",
        shared_candidates=["shared_root", "shared_single", "shared_four"],
        shared_ablation="same group types and balanced coverage with neutral proposal scores",
        final_selection="maximum development accuracy over the three rounds, then CE and name",
        index_archive_sha256=digest(OUT / f"{setting}_extended_indices.npz"),
    )
    save_json(OUT / f"{setting}_extended_plan.json", plan)


def study(setting, phase, gpus, extended=False, shared=False, contrast=False):
    source_tag = (
        f"{setting}_contrast"
        if contrast
        else f"{setting}_extended"
        if extended or shared
        else setting
    )
    tag = f"{setting}_shared" if shared else source_tag
    plan = read_json(OUT / f"{source_tag}_plan.json")
    archive = OUT / f"{source_tag}_indices.npz"
    if digest(archive) != plan["index_archive_sha256"]:
        raise ValueError("Candidate archive has changed")
    with np.load(archive) as data:
        bank = dict(data)
    folder = LOCAL / setting
    dev = phase == "development"
    if contrast:
        if dev:
            raise ValueError("The centering hypothesis has no development selection")
        winner = "centered"
        methods = [winner, "anchor_native", "asrr"]
        seeds = plan["target_seeds"]
        count = plan["target_episodes_per_seed"]
    elif dev:
        methods = (
            plan["shared_candidates"]
            if shared
            else plan["eligible"] + ([] if extended else ["asrr"])
        )
        seeds = [plan["development_seed"]]
        count = plan["development_episodes"]
    else:
        winner = read_json(OUT / f"{tag}_frozen.json")["winner"]
        control = (
            winner.replace("joint_", "input_").replace("adaptive_", "iid_")
            if not winner.startswith("shared_")
            else "uniform_" + winner
        )
        methods = list(dict.fromkeys([winner, control, *plan["ref_files"].keys()]))
        if extended or shared:
            earlier = [plan["initial_winner"]]
            if shared:
                earlier.append(
                    read_json(OUT / f"{setting}_extended_frozen.json")["winner"]
                )
            methods += [k for k in dict.fromkeys(earlier) if k not in methods]
        if shared:
            anchor_file = OUT / f"{setting}_anchor_indices.npz"
            with np.load(anchor_file) as anchors:
                methods.extend(anchors.files)
                bank.update(dict(anchors))
            save_json(
                OUT / f"{setting}_target_methods.json",
                {
                    "primary_anchor": "anchor_native versus asrr",
                    "primary_selected": f"{winner} versus asrr",
                    "all": methods,
                    "anchor_sha256": digest(anchor_file),
                    "bank_sha256": digest(archive),
                },
            )
        seeds = plan["target_seeds"]
        count = plan["target_episodes_per_seed"]
    jobs, records = [], {}
    for number, seed in enumerate(seeds):
        key = f"{'contrast' if contrast else phase}_{number:02d}"
        original = OUT / f"{setting}_{phase}_panels.json"
        record = (
            read_json(original)[key]
            if (extended or shared) and original.exists()
            else public_panel(
                setting, key, count, seed, directory=folder, target=not dev
            )
        )
        records[key] = record
        jobs.extend((key, name, seed + 2000) for name in methods)
    save_json(OUT / f"{tag}_{phase}_panels.json", records)
    torch.set_num_threads(1)

    def worker(slot):
        device = f"cuda:{gpus[slot]}"
        torch.cuda.set_device(device)
        for key, name, seed in jobs[slot :: len(gpus)]:
            with np.load(folder / f"{key}.npz") as data:
                panel = {k: data[k] for k in data.files if k != "query_y"}
            pred = predict(panel, bank[name], setting, seed, device)
            np.save(folder / f"{key}_{name}.npy", pred)
            print(f"{setting} {key} {name}", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, range(len(gpus))))
    rows = {name: [] for name in methods}
    hashes = {}
    for key in records:
        for name in methods:
            hashes[f"{key}_{name}"] = digest(folder / f"{key}_{name}.npy")
    save_json(OUT / f"{tag}_{phase}_predictions.json", hashes)
    for key, record in records.items():
        with np.load(record["path"] if dev else record["answer_path"]) as data:
            truth = data["query_y" if dev else "y"]
        for name in methods:
            rows[name].append(metrics(np.load(folder / f"{key}_{name}.npy"), truth))
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    np.savez_compressed(OUT / f"{tag}_{phase}_episodes.npz", **rows)
    summary = {k: v.mean(axis=0).tolist() for k, v in rows.items()}
    if dev:
        eligible = methods if shared else plan["eligible"]
        winner = min(eligible, key=lambda k: (-summary[k][0], summary[k][1], k))
        if shared:
            previous = read_json(OUT / f"{setting}_extended_frozen.json")
            pilot = previous["winner"]
            summary[pilot] = previous["metrics"][pilot]
            winner = min(
                (winner, pilot), key=lambda k: (-summary[k][0], summary[k][1], k)
            )
        elif extended:
            pilot = plan["initial_winner"]
            summary[pilot] = read_json(OUT / f"{setting}_frozen.json")["metrics"][pilot]
            winner = min(
                (winner, pilot), key=lambda k: (-summary[k][0], summary[k][1], k)
            )
        selection = {"winner": winner, "metrics": summary}
        save_json(OUT / f"{tag}_frozen.json", selection)
        print(selection, flush=True)
    else:
        differences = {}
        for name in methods[1:]:
            delta = rows[winner][:, 0] - rows[name][:, 0]
            half = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
            differences[name] = [delta.mean(), delta.mean() - half, delta.mean() + half]
        report = {
            "episodes": len(rows[winner]),
            "accuracy_ce": summary,
            "winner": winner,
            "paired_difference_and_ci95": differences,
        }
        if shared:
            anchors = {}
            for name in (
                "asrr",
                "anchor_aligned",
                "independent_single_fit",
                "uniform",
                "wsr",
            ):
                if name in rows:
                    delta = rows["anchor_native"][:, 0] - rows[name][:, 0]
                    half = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
                    anchors[name] = [
                        delta.mean(),
                        delta.mean() - half,
                        delta.mean() + half,
                    ]
            report["anchor_native_minus_reference"] = anchors
        save_json(OUT / f"{tag}_results.json", report)
        print(report, flush=True)


def reference_controls(gpus):
    """Score all four archived uniform masks and the archived selected WSR fit."""
    setting = "omniglot10"
    original = read_json(OUT / f"{setting}_shared_target_panels.json")
    choices = OUT.parent / "omniglot64_gradient_profiles_2026-11-19/choices_frozen.json"
    policies = read_json(choices)["policies"]
    keys = [
        "wsr_step_75",
        *(f"balanced_random_mask_{s}" for s in range(7611189542, 7611189545)),
    ]
    refs = {}
    for key in keys:
        row = policies[key]
        path = OMNI_BANK / f"{row['name']}.npz"
        if digest(path) != row["model_sha256"]:
            raise ValueError(f"Archived reference differs for {key}")
        with np.load(path) as data:
            refs[key] = data["index"]
    save_json(
        OUT / "omniglot10_reference_plan.json",
        {
            "archive": str(choices),
            "choices": {k: policies[k] for k in keys},
            "already_scored": "uniform is archived balanced mask 7611189541",
            "correction": "The WSR model ID is 104 for the selected step 75. Model ID 075 in the initial run is a different candidate.",
        },
    )
    folder = LOCAL / setting
    seeds = read_json(OUT / f"{setting}_extended_plan.json")["target_seeds"]
    jobs = [
        (key, name, seed + 2000)
        for key, seed in zip(original, seeds, strict=True)
        for name in keys
    ]
    torch.set_num_threads(1)

    def worker(slot):
        device = f"cuda:{gpus[slot]}"
        torch.cuda.set_device(device)
        for key, name, seed in jobs[slot :: len(gpus)]:
            if digest(Path(original[key]["path"])) != original[key]["sha256"]:
                raise ValueError("A frozen episode file has changed")
            with np.load(original[key]["path"]) as data:
                panel = {k: data[k] for k in data.files if k != "query_y"}
            p = predict(panel, refs[name], setting, seed, device)
            np.save(folder / f"{key}_{name}.npy", p)
            print(key, name, flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, range(len(gpus))))
    hashes = {
        f"{key}_{name}": digest(folder / f"{key}_{name}.npy")
        for key in original
        for name in keys
    }
    save_json(OUT / "omniglot10_reference_predictions.json", hashes)
    rows = {name: [] for name in keys}
    for key, record in original.items():
        with np.load(record["answer_path"]) as data:
            y = data["y"]
        for name in keys:
            rows[name].append(metrics(np.load(folder / f"{key}_{name}.npy"), y))
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    with np.load(OUT / "omniglot10_shared_target_episodes.npz") as data:
        original_rows = dict(data)
    rows["mean_four_uniform_masks"] = np.mean(
        [original_rows["uniform"], *(rows[name] for name in keys[1:])], axis=0
    )
    np.savez_compressed(OUT / "omniglot10_reference_episodes.npz", **rows)
    report = {}
    for key, v in rows.items():
        delta = v[:, 0] - original_rows["asrr"][:, 0]
        half = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
        report[key] = {
            "accuracy_ce": v.mean(axis=0).tolist(),
            "minus_asrr": [delta.mean(), delta.mean() - half, delta.mean() + half],
        }
    save_json(OUT / "omniglot10_reference_results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "signs", "development", "target", "controls")
    )
    parser.add_argument("setting", choices=SETTINGS)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--contrast", action="store_true")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.setting)
    elif args.phase == "signs":
        signed_candidates(args.setting)
    elif args.phase == "controls":
        reference_controls(args.gpus)
    else:
        study(
            args.setting,
            args.phase,
            args.gpus,
            args.extended,
            args.shared,
            args.contrast,
        )
