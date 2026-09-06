"""Independently audit and report the multi-task CVAE/MLP search.

This runner is intentionally post-hoc: it never optimizes a latent or trains
an MLP.  It re-decodes every saved latent with the frozen source checkpoints,
checks the completed evaluation shards, and then aggregates predeclared
comparisons.  The reported uncertainty is conditional on the checkpoints and
the two held-out gaps in each profile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from data.generate import configure_compute_device, ideal_mask  # noqa: E402
from evaluation.eval_generated_masks import _load_model  # noqa: E402
from evaluation.oracle_ideal import hard_topk, soft_topk  # noqa: E402
from evaluation.structural import best_permutation_iou  # noqa: E402


DEFAULT_OUT = ROOT / "outputs/combined_task_agreement/20260906"
EXPECTED_METHODS = ["task_only", "combined_0p1", "combined_1", "combined_10",
                    "agreement_30", "agreement_1000", "prior", "ideal"]
N_ORDINALS = 64
N_SHARDS = 2
N_TASKS_PER_GAP = 8


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_gap_tasks(spec: dict[str, Any], gap: int) -> list[str]:
    tasks = list(spec["tasks"][str(gap)])
    _require(len(tasks) == N_TASKS_PER_GAP and len(set(tasks)) == len(tasks),
             f"gap {gap}: expected eight distinct motif tasks")
    _require(all(config.parse_task(task).gap == gap for task in tasks),
             f"gap {gap}: task has a mismatched condition")
    return tasks


def _freeze(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_frozen(model: torch.nn.Module, before: dict[str, torch.Tensor], label: str) -> None:
    _require(not any(parameter.grad is not None for parameter in model.parameters()),
             f"{label}: frozen checkpoint received a gradient")
    for name, value in model.state_dict().items():
        _require(torch.equal(value, before[name]), f"{label}: checkpoint changed at {name}")


@torch.no_grad()
def _decode(model: torch.nn.Module, z: torch.Tensor, condition: torch.Tensor,
            *, temperature: float, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode [method, ordinal, latent] independently of the search runner."""
    _require(z.ndim == 3, f"latent tensor must be [method, ordinal, latent], got {z.shape}")
    methods, ordinals, latent = z.shape
    _require(latent == int(getattr(model, "latent_dim", -1)), "saved latent width mismatches checkpoint")
    c = condition.expand(methods * ordinals, -1)
    logits = model.decode(z.reshape(methods * ordinals, latent), c)
    _require(logits.shape == (methods * ordinals, config.MASK_DIM),
             f"decoder output has unexpected shape {logits.shape}")
    soft = soft_topk(logits, k, temperature).reshape(methods, ordinals, config.SEQ_LEN, config.H)
    hard = hard_topk(logits, k).reshape(methods, ordinals, config.SEQ_LEN, config.H)
    return soft.cpu(), hard.cpu()


def _check_binary_topk(hard: torch.Tensor, label: str) -> None:
    _require(hard.shape[-2:] == (config.SEQ_LEN, config.H), f"{label}: wrong mask shape")
    _require(bool(((hard == 0) | (hard == 1)).all()), f"{label}: hard mask is not binary")
    _require(bool((hard.sum((-1, -2)) == config.K_ACTIVE).all()),
             f"{label}: hard mask does not have exactly {config.K_ACTIVE} edges")


def _check_soft(soft: torch.Tensor, label: str) -> None:
    _require(soft.shape[-2:] == (config.SEQ_LEN, config.H), f"{label}: wrong soft-mask shape")
    _require(bool(torch.isfinite(soft).all()), f"{label}: soft mask has non-finite entries")
    _require(bool((soft >= 0).all() and (soft <= 1).all()), f"{label}: soft mask is outside [0,1]")
    expected = torch.full_like(soft.sum((-1, -2)), float(config.K_ACTIVE))
    torch.testing.assert_close(soft.sum((-1, -2)), expected, rtol=0, atol=3e-4,
                               msg=f"{label}: soft top-k cardinality")


def _assert_close(actual: torch.Tensor, saved: torch.Tensor, label: str, *, cpu: bool) -> None:
    _require(actual.shape == saved.shape, f"{label}: shape {actual.shape} != saved {saved.shape}")
    # CUDA is the production path; CPU re-decoding is a valid independent
    # audit witness and can differ very slightly in sigmoid/top-k arithmetic.
    torch.testing.assert_close(actual, saved, rtol=1e-5, atol=2e-5 if cpu else 1e-7, msg=label)


def _search_methods(saved: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate saved tensor layouts and return z, soft, hard in method order."""
    combined = saved["combined"]
    _require(set(combined) >= {"z", "soft", "hard", "history"}, "combined result is incomplete")
    z = combined["z"]
    _require(z.shape[:2] == (4, 2), f"combined z has wrong leading shape {z.shape}")
    _require(z.shape[2] in (32, 64), "combined z has unexpected ordinal count")
    _require(len(combined["history"]) == 30, "combined history is incomplete")
    for outer, row in enumerate(combined["history"]):
        for key in ("task_bce", "pair_mse", "objective", "task_grad_norm", "pair_grad_norm"):
            _require(key in row and len(row[key]) == 4, f"history outer {outer}: missing {key}")
    controls = saved["controls"]
    _require(set(controls) == {"agreement_30", "agreement_1000"}, "unexpected agreement controls")
    _require(set(saved["prior"]) >= {"soft", "hard"}, "prior masks are missing")
    method_z = torch.cat((z, controls["agreement_30"]["z"], controls["agreement_1000"]["z"],
                          saved["initial_z"].unsqueeze(0)), dim=0)
    _require(method_z.shape[:2] == (7, 2), f"saved latent methods have wrong shape {method_z.shape}")
    _require(bool(torch.isfinite(method_z).all()), "saved z contains non-finite entries")
    _require(bool((method_z.norm(dim=-1) <= 8.00002).all()), "saved z exceeds radius 8")
    soft = torch.cat((combined["soft"], controls["agreement_30"]["soft"],
                      controls["agreement_1000"]["soft"], saved["prior"]["soft"]), dim=0)
    hard = torch.cat((combined["hard"], controls["agreement_30"]["hard"],
                      controls["agreement_1000"]["hard"], saved["prior"]["hard"]), dim=0)
    _require(soft.shape[:2] == (7, 2) and hard.shape[:2] == (7, 2),
             "saved method masks have wrong leading dimensions")
    _check_soft(soft, "saved masks")
    _check_binary_topk(hard, "saved masks")
    return method_z, soft, hard


def _audit_shard(root: Path, protocol: dict[str, Any], profile: str, gap: int, shard: int,
                 models: list[torch.nn.Module], frozen: list[dict[str, torch.Tensor]],
                 device: torch.device) -> dict[str, Any]:
    spec, settings = protocol["profiles"][profile], protocol["settings"]
    tasks = _as_gap_tasks(spec, gap)
    destination = root / profile / f"gap{gap}_shard{shard}"
    for filename in ("search.pt", "evaluation_seed0.pt", "evaluation_seed1.pt", "evaluation_seed2.pt", "done.json"):
        _require((destination / filename).is_file(), f"missing required artifact: {destination / filename}")
    done = json.loads((destination / "done.json").read_text())
    search_path = destination / "search.pt"
    saved = torch.load(search_path, weights_only=True, map_location="cpu")
    metadata = saved.get("metadata")
    start, stop = shard * 32, (shard + 1) * 32
    expected_metadata = {"protocol_sha256": sha(root / "protocol.json"), "profile": profile,
                         "gap": gap, "start": start, "stop": stop, "tasks": tasks,
                         "methods": EXPECTED_METHODS}
    for key, value in expected_metadata.items():
        _require(metadata.get(key) == value, f"{destination}: search metadata {key} differs")
    _require(isinstance(metadata.get("torch_version"), str), f"{destination}: missing torch version")
    _require(done.get("decoder_unchanged") is True, f"{destination}: worker did not attest frozen decoders")
    _require(done.get("search_sha256") == sha(search_path),
             f"{destination}: done search hash differs from artifact")
    for key, value in expected_metadata.items():
        _require(done.get(key) == value, f"{destination}: done metadata {key} differs")
    method_z, saved_soft, saved_hard = _search_methods(saved)
    parent = torch.load(spec["prior"], weights_only=True, map_location="cpu")
    parent_gaps = parent["gaps"].tolist()
    _require(gap in parent_gaps, f"{destination}: gap missing from parent latent bank")
    parent_index = parent_gaps.index(gap)
    expected_initial = torch.stack([
        parent["latents"]["prior"][key].reshape(len(parent_gaps), N_ORDINALS, -1)[parent_index, start:stop]
        for key in ("z1", "z2")
    ])
    _require(torch.equal(saved["initial_z"], expected_initial),
             f"{destination}: initial_z is not the exact parent prior block")
    condition = models[0].condition([tasks[0]], device=device)
    _require(torch.equal(condition, models[1].condition([tasks[0]], device=device)),
             f"{destination}: source checkpoints disagree on condition encoding")
    redecode_soft, redecode_hard = [], []
    # Decode all 7 non-gold methods.  The loop keeps each source decoder an
    # independent witness and does not reuse the search implementation.
    for decoder, model in enumerate(models):
        soft, hard = _decode(model, method_z[:, decoder].to(device), condition,
                             temperature=float(settings["temperature"]), k=config.K_ACTIVE)
        _assert_close(soft, saved_soft[:, decoder], f"{destination}: decoder {decoder} soft re-decode",
                      cpu=device.type == "cpu")
        _require(torch.equal(hard, saved_hard[:, decoder]),
                 f"{destination}: decoder {decoder} hard re-decode differs")
        redecode_soft.append(soft)
        redecode_hard.append(hard)
    decoded_soft = torch.stack(redecode_soft, dim=1)
    decoded_hard = torch.stack(redecode_hard, dim=1)
    _check_soft(decoded_soft, f"{destination}: re-decoded masks")
    _check_binary_topk(decoded_hard, f"{destination}: re-decoded masks")
    target = ideal_mask(tasks[0]).float()
    gold = target[None, None, None].expand(1, 2, stop - start, -1, -1).clone()
    hard = torch.cat((decoded_hard, gold), dim=0)
    _check_binary_topk(hard, f"{destination}: masks including ideal")
    expected_hard_hash = tensor_sha(hard)
    _require(done.get("hard_masks_sha256") == expected_hard_hash,
             f"{destination}: done hard-mask hash differs from independently decoded masks")
    evaluations = []
    expected_eval_metadata = {**metadata, "search_sha256": sha(search_path),
                              "hard_masks_sha256": expected_hard_hash}
    for seed in settings["evaluation_seeds"]:
        row = torch.load(destination / f"evaluation_seed{seed}.pt", weights_only=True, map_location="cpu")
        _require(row.get("metadata") == expected_eval_metadata,
                 f"{destination}: evaluation seed {seed} provenance differs")
        _require(row.get("seed") == seed, f"{destination}: evaluation seed label differs")
        for metric in ("bce", "acc"):
            value = row.get(metric)
            _require(isinstance(value, torch.Tensor), f"{destination}: missing tensor {metric}")
            _require(value.shape == (len(EXPECTED_METHODS), 2, N_TASKS_PER_GAP, stop - start),
                     f"{destination}: {metric} has wrong shape {value.shape}")
            _require(bool(torch.isfinite(value).all()), f"{destination}: {metric} has non-finite values")
        evaluations.append({"seed": seed, "bce": row["bce"].cpu(), "acc": row["acc"].cpu()})
    _assert_frozen(models[0], frozen[0], f"{profile}/gap{gap}/decoder0")
    _assert_frozen(models[1], frozen[1], f"{profile}/gap{gap}/decoder1")
    iou_to_ideal = torch.tensor([[[best_permutation_iou(mask, target)["iou"]
                                   for mask in hard[method, decoder]]
                                  for decoder in range(2)]
                                 for method in range(len(EXPECTED_METHODS))], dtype=torch.float64)
    pair_iou = torch.tensor([[best_permutation_iou(hard[method, 0, ordinal], hard[method, 1, ordinal])["iou"]
                              for ordinal in range(stop - start)]
                             for method in range(len(EXPECTED_METHODS))], dtype=torch.float64)
    return {"destination": str(destination), "tasks": tasks, "hard": hard, "iou": iou_to_ideal,
            "pair_iou": pair_iou, "evaluations": evaluations, "history": saved["combined"]["history"],
            "search_sha256": sha(search_path), "hard_masks_sha256": expected_hard_hash}


def _crossed_ci(deltas: torch.Tensor, *, seed: int) -> list[float]:
    """Cross pairs and ordinals; each sampled pair retains both profile gaps."""
    _require(deltas.shape == (8, 2, N_ORDINALS), f"expected [pair, gap, ordinal], got {deltas.shape}")
    generator = torch.Generator().manual_seed(seed)
    draws = 10_000
    pair_draw = torch.randint(8, (draws, 8), generator=generator)
    ordinal_draw = torch.randint(N_ORDINALS, (draws, N_ORDINALS), generator=generator)
    resampled = deltas[pair_draw[:, :, None], :, ordinal_draw[:, None, :]].mean((1, 2, 3))
    return torch.quantile(resampled, torch.tensor([.025, .975], dtype=torch.float64)).tolist()


def _profile_comparisons(acc: torch.Tensor, tasks: list[str], profile: str) -> dict[str, Any]:
    # acc is [seed, method, decoder, task, ordinal].  Decoder and evaluation
    # seeds are paired nuisance replication, averaged before the CI.
    _require(acc.shape == (3, 8, 2, 16, N_ORDINALS), f"bad aggregated accuracy shape {acc.shape}")
    pairs = sorted({(config.parse_task(task).a, config.parse_task(task).b) for task in tasks})
    gaps = sorted({config.parse_task(task).gap for task in tasks})
    _require(len(pairs) == 8 and len(gaps) == 2, f"{profile}: expected 8 motif pairs and 2 gaps")
    ordered = {(config.parse_task(task).a, config.parse_task(task).b, config.parse_task(task).gap): i
               for i, task in enumerate(tasks)}
    comparisons: dict[str, Any] = {}
    for candidate in ("combined_0p1", "combined_1", "combined_10"):
        for reference in ("task_only", "agreement_30"):
            c, r = EXPECTED_METHODS.index(candidate), EXPECTED_METHODS.index(reference)
            per_task = (acc[:, c] - acc[:, r]).mean((0, 1))  # [task, ordinal]
            grid = torch.stack([torch.stack([per_task[ordered[a, b, gap]] for gap in gaps])
                                for a, b in pairs])
            key = f"{candidate}_minus_{reference}"
            comparisons[key] = {
                "accuracy_delta": float(grid.mean()),
                "crossed_bootstrap_95": _crossed_ci(grid.double(), seed=20260906 + c * 101 + r),
                "scope": "conditional on fixed checkpoints and held-out gaps; paired delta averages both decoders and all three evaluation seeds before resampling 8 motif pairs and 64 ordinals; each pair draw retains its two gaps",
            }
    return comparisons


def _gradient_diagnostic(histories: list[list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    lambdas = (0., .1, 1., 10.)
    for index, weight in enumerate(lambdas):
        task = torch.tensor([row[outer]["task_grad_norm"][index] for row in histories for outer in range(30)],
                            dtype=torch.float64)
        pair = torch.tensor([row[outer]["pair_grad_norm"][index] for row in histories for outer in range(30)],
                            dtype=torch.float64)
        entry = {"lambda": weight, "task_gradient_norm_mean": float(task.mean()),
                 "pair_gradient_norm_mean": float(pair.mean())}
        if weight:
            scaled = weight * pair
            entry.update(lambda_pair_gradient_norm_mean=float(scaled.mean()),
                         task_over_lambda_pair_gradient_norm_mean=float((task / scaled.clamp_min(1e-15)).mean()),
                         task_over_lambda_pair_gradient_norm_median=float(torch.median(task / scaled.clamp_min(1e-15))))
        else:
            entry["lambda_pair_gradient_norm_mean"] = 0.
            entry["task_over_lambda_pair_gradient_norm_mean"] = None
        result[EXPECTED_METHODS[index]] = entry
    return result


def _metrics(acc: torch.Tensor, bce: torch.Tensor, iou: torch.Tensor, pair_iou: torch.Tensor,
             tasks: list[str]) -> dict[str, Any]:
    # iou/pair_iou are fixed-mask quantities and do not repeat across target tasks.
    rows: dict[str, Any] = {}
    for method_index, method in enumerate(EXPECTED_METHODS):
        rows[method] = {
            "mean_accuracy": float(acc[:, method_index].double().mean()),
            "mean_bce": float(bce[:, method_index].double().mean()),
            "mean_iou_to_ideal": float(iou[method_index].mean()),
            "max_iou_to_ideal": float(iou[method_index].max()),
            "exact_ideal_count": int((iou[method_index] == 1).sum()),
            "mean_pair_iou": float(pair_iou[method_index].mean()),
        }
    by_gap: dict[str, Any] = {}
    for gap in sorted({config.parse_task(task).gap for task in tasks}):
        indexes = [i for i, task in enumerate(tasks) if config.parse_task(task).gap == gap]
        by_gap[str(gap)] = {method: {
            "mean_accuracy": float(acc[:, method_index, :, indexes].double().mean()),
            "mean_bce": float(bce[:, method_index, :, indexes].double().mean()),
            "mean_iou_to_ideal": float(iou[method_index, :, indexes[0] // 8].mean()),
            "max_iou_to_ideal": float(iou[method_index, :, indexes[0] // 8].max()),
            "exact_ideal_count": int((iou[method_index, :, indexes[0] // 8] == 1).sum()),
            "mean_pair_iou": float(pair_iou[method_index, indexes[0] // 8].mean()),
        } for method_index, method in enumerate(EXPECTED_METHODS)}
    return {"methods": rows, "by_gap": by_gap}


def _plot(root: Path, profiles: dict[str, Any]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = EXPECTED_METHODS
    figure, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for row, profile in enumerate(("interp", "extrap")):
        gaps = sorted(profiles[profile]["gaps"])
        for column, gap in enumerate(gaps):
            for offset, metric in enumerate(("mean_accuracy", "mean_iou_to_ideal")):
                axis = axes[row, column * 2 + offset]
                values = [profiles[profile]["metrics"]["by_gap"][str(gap)][method][metric] for method in names]
                axis.bar(range(len(names)), values, color=["#4C78A8" if name.startswith("combined") else "#777777" for name in names])
                axis.set_title(f"{profile}, gap {gap}: {'accuracy' if metric == 'mean_accuracy' else 'IoU to ideal'}")
                axis.set_ylim(0, 1.02)
                axis.set_xticks(range(len(names)), [name.replace("combined_", "c_") for name in names], rotation=55, ha="right", fontsize=8)
                axis.grid(axis="y", alpha=.25)
    figure.suptitle("Frozen-CVAE masks after supervised adaptation on target motif tasks", y=1.01)
    figure.tight_layout()
    path = root / "comparison_accuracy_iou.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# Combined task-loss and agreement audit", "",
             "All saved latents were independently re-decoded from the frozen source checkpoints. "
             "This is supervised adaptation on the eight target tasks per held-out gap; it is not zero-shot. "
             "Training and test draws are independent draws from the same finite input universe.", ""]
    for profile, result in summary["profiles"].items():
        lines += [f"## {profile}", "", "| method | accuracy | mean IoU | max IoU | exact ideal | pair IoU |", "|---|---:|---:|---:|---:|---:|"]
        for method in EXPECTED_METHODS:
            metric = result["metrics"]["methods"][method]
            lines.append(f"| {method} | {metric['mean_accuracy']:.4f} | {metric['mean_iou_to_ideal']:.4f} | "
                         f"{metric['max_iou_to_ideal']:.4f} | {metric['exact_ideal_count']} | {metric['mean_pair_iou']:.4f} |")
        lines += ["", "Paired accuracy differences below average decoder and evaluation-seed replicas, then use a crossed bootstrap over the eight shared motif pairs and 64 common ordinals. The intervals are conditional on these fixed checkpoints and gaps; they are descriptive and were not used to select a lambda.", ""]
        for name, comparison in result["comparisons"].items():
            low, high = comparison["crossed_bootstrap_95"]
            lines.append(f"- {name}: {comparison['accuracy_delta']:+.5f} (conditional 95% CI [{low:+.5f}, {high:+.5f}])")
        lines += ["", "The gradient diagnostic compares the direct task gradient with λ times the pair-agreement gradient during search. It diagnoses objective scaling only; it is not a selection criterion.", ""]
    return "\n".join(lines) + "\n"


@torch.no_grad()
def run(root: Path, device: torch.device) -> dict[str, Any]:
    _require(device.type in {"cpu", "cuda"}, "audit device must be cpu or cuda")
    _require(device.type != "cuda" or torch.cuda.is_available(), "requested CUDA is unavailable")
    configure_compute_device(str(device))
    protocol_path = root / "protocol.json"
    _require(protocol_path.is_file(), f"missing protocol: {protocol_path}")
    protocol = json.loads(protocol_path.read_text())
    _require(protocol.get("settings", {}).get("methods") == EXPECTED_METHODS, "protocol method order differs")
    _require(protocol["settings"].get("evaluation_seeds") == [0, 1, 2], "protocol evaluation seeds differ")
    _require(protocol["settings"].get("n_starts") == N_ORDINALS, "protocol ordinal count differs")
    for source, digest in protocol["source_hashes"].items():
        _require(sha(source) == digest, f"prepared source changed: {source}")
    profile_results: dict[str, Any] = {}
    aggregate: dict[str, Any] = {"protocol_sha256": sha(protocol_path), "profiles": {}}
    for profile, spec in protocol["profiles"].items():
        for key in ("checkpoint", "checkpoint2", "prior", "split"):
            _require(sha(spec[key]) == spec[key + "_sha256"], f"{profile}: {key} provenance mismatch")
        split = json.loads(Path(spec["split"]).read_text())
        _require(all(task in split["test_tasks"] for gap in spec["heldout_gaps"] for task in _as_gap_tasks(spec, gap)),
                 f"{profile}: target tasks are not split test tasks")
        models = [_load_model(Path(spec[key]), device)[0] for key in ("checkpoint", "checkpoint2")]
        frozen = [_freeze(model) for model in models]
        gap_rows = []
        for gap in sorted(spec["heldout_gaps"]):
            shards = [_audit_shard(root, protocol, profile, gap, shard, models, frozen, device)
                      for shard in range(N_SHARDS)]
            _require(shards[0]["tasks"] == shards[1]["tasks"], f"{profile} gap {gap}: shard task order differs")
            hard = torch.cat([row["hard"] for row in shards], dim=2)
            iou = torch.cat([row["iou"] for row in shards], dim=2)
            pair_iou = torch.cat([row["pair_iou"] for row in shards], dim=1)
            _require(hard.shape == (8, 2, N_ORDINALS, 16, 16), f"{profile} gap {gap}: aggregate hard shape")
            evaluations = {metric: torch.stack([torch.cat([shard["evaluations"][seed][metric]
                                                             for shard in shards], dim=3)
                                                 for seed in range(3)])
                           for metric in ("acc", "bce")}
            for metric, value in evaluations.items():
                _require(value.shape == (3, 8, 2, 8, N_ORDINALS),
                         f"{profile} gap {gap}: {metric} aggregate shape {value.shape}")
            gap_rows.append({"gap": gap, "tasks": shards[0]["tasks"], "hard": hard, "iou": iou,
                             "pair_iou": pair_iou, "acc": evaluations["acc"], "bce": evaluations["bce"],
                             "histories": [shard["history"] for shard in shards],
                             "shards": [{"search_sha256": shard["search_sha256"], "hard_masks_sha256": shard["hard_masks_sha256"]}
                                        for shard in shards]})
        tasks = [task for row in gap_rows for task in row["tasks"]]
        # The search mask is shared by the eight target tasks at a gap, so
        # retain its actual [gap, ordinal] axes rather than duplicating it
        # across tasks.  Evaluation below retains the required task axis.
        hard = torch.stack([row["hard"] for row in gap_rows], dim=2)
        iou = torch.stack([row["iou"] for row in gap_rows], dim=2)  # [method, decoder, gap, ordinal]
        pair_iou = torch.stack([row["pair_iou"] for row in gap_rows], dim=1)  # [method, gap, ordinal]
        acc = torch.cat([row["acc"] for row in gap_rows], dim=3)
        bce = torch.cat([row["bce"] for row in gap_rows], dim=3)
        _require(acc.shape == (3, 8, 2, 16, N_ORDINALS), f"{profile}: evaluation [seed,method,decoder,task,ordinal] shape")
        metrics = _metrics(acc, bce, iou, pair_iou, tasks)
        result = {"gaps": [row["gap"] for row in gap_rows], "tasks": tasks, "metrics": metrics,
                  "comparisons": _profile_comparisons(acc, tasks, profile),
                  "gradient_diagnostic": _gradient_diagnostic([history for row in gap_rows for history in row["histories"]]),
                  "audit": {"redecoded_saved_latents": len(gap_rows) * N_SHARDS * 7 * 2 * 32,
                            "completed_shards": len(gap_rows) * N_SHARDS,
                            "evaluation_files_checked": len(gap_rows) * N_SHARDS * 3},
                  "artifact_shapes": {"hard_masks_method_decoder_gap_ordinal": list(hard.shape),
                                      "evaluation_seed_method_decoder_task_ordinal": list(acc.shape)},
                  "shards": {str(row["gap"]): row["shards"] for row in gap_rows}}
        profile_results[profile] = result
        aggregate["profiles"][profile] = {"tasks": tasks, "hard_masks": hard,
                                           "iou_to_ideal": iou, "pair_iou": pair_iou,
                                           "evaluation_acc": acc, "evaluation_bce": bce}
        for index, model in enumerate(models):
            _assert_frozen(model, frozen[index], f"{profile}/decoder{index}")
    # A single atomic tensor artifact makes the audited axes available for
    # downstream inspection without silently replacing search artifacts.
    aggregate_path = root / "audited_aggregate.pt"
    temporary = root / "audited_aggregate.pt.tmp"
    torch.save(aggregate, temporary)
    os.replace(temporary, aggregate_path)
    summary = {"audited": False, "protocol_sha256": sha(protocol_path), "aggregate_sha256": sha(aggregate_path),
               "scope": "supervised adaptation on target tasks; train/test are separate draws from the same finite input universe",
               "uncertainty": "conditional crossed bootstrap only; fixed source checkpoints and profile gaps; no hypothesis-test claim and no lambda selection",
               "profiles": profile_results}
    plot_path = _plot(root, profile_results)
    report_path = root / "REPORT.md"
    report_path.write_text(_markdown({**summary, "audited": True}))
    summary["figure"] = {"path": str(plot_path), "sha256": sha(plot_path)}
    summary["report"] = {"path": str(report_path), "sha256": sha(report_path)}
    # The flag flips only after every shard, witness decode, aggregate, plot,
    # and readable report has been written successfully.
    summary["audited"] = True
    write_json_atomic(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda", help="cpu or CUDA device used only for witness re-decoding")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    result = run(args.out, torch.device(args.device))
    print(json.dumps({"audited": result["audited"], "aggregate_sha256": result["aggregate_sha256"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
