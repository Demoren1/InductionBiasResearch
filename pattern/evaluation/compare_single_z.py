"""Matched-checkpoint comparison of task-adapted z and decoder agreement.

Single-z search has target-label access; agreement is label-free. They share
VAE42, initial latents, mask density, radius and final fresh-MLP evaluation,
but NOT their computational budgets or optimization objective.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from evaluation.run_decoder_agreement import (
    DEFAULT_ROOT, describe, dump_json, eval_one_task, file_hash, load_models, structure,
)


def search(args):
    from evaluation.single_z_task import optimize_task_z
    root = args.parent / "single_z_comparison"
    protocol = json.loads((root / "protocol.json").read_text())
    if args.pattern not in protocol["patterns"]:
        raise ValueError("Pattern outside predeclared held-out tasks")
    output = root / f"search_{args.pattern}.pt"
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads((args.parent / "search_provenance.json").read_text())
    if file_hash(args.parent / "masks.pt") != previous["mask_sha256"]:
        raise ValueError("Original mask hash mismatch")
    models, provenance = load_models(args.parent, torch.device(args.device))
    if provenance[0]["seed"] != protocol["model_seed"]:
        raise ValueError("Wrong decoder for single-z comparison")
    if provenance[0]["sha256"] != previous["models"][0]["sha256"]:
        raise ValueError("VAE42 checkpoint changed since agreement experiment")
    old = torch.load(args.parent / "optimization.pt", weights_only=True, map_location="cpu")
    initial = old["initial_z1"].to(args.device)
    assert len(initial) == protocol["n_starts"]
    result = optimize_task_z(models[0], initial, args.pattern, torch.device(args.device), **protocol["search"])
    if not torch.equal(result["initial_z"], old["initial_z1"]):
        raise AssertionError("Single-z starts differ from agreement starts")
    if not torch.equal(result["initial_masks"], old["initial_masks1"]):
        raise AssertionError("Initial masks differ from the agreement baseline")
    result["provenance"] = {"checkpoint": provenance[0],
                            "original_optimization_sha256": file_hash(args.parent / "optimization.pt"),
                            "original_masks_sha256": file_hash(args.parent / "masks.pt"),
                            "protocol_sha256": file_hash(root / "protocol.json"),
                            "source_sha256": {name: file_hash(Path(__file__).with_name(name))
                                              for name in ("single_z_task.py", "compare_single_z.py")},
                            "pattern": args.pattern, "uses_target_labels": True,
                            "uses_gold": False, "uses_final_test": False, "device": args.device}
    torch.save(result, output)
    print(f"[single-z] frozen mask artifacts -> {output}", flush=True)


def agreement_control(args):
    from evaluation.decoder_agreement import optimize_agreement
    root = args.parent / "single_z_comparison"
    protocol = json.loads((root / "protocol.json").read_text())
    output = root / "agreement_30.pt"
    if output.exists():
        raise FileExistsError(output)
    models, provenance = load_models(args.parent, torch.device(args.device))
    old_provenance = json.loads((args.parent / "search_provenance.json").read_text())
    assert [p["sha256"] for p in provenance] == [p["sha256"] for p in old_provenance["models"]]
    result = optimize_agreement(*models, device=args.device, **protocol["agreement_short_control"])
    original = torch.load(args.parent / "optimization.pt", weights_only=True, map_location="cpu")
    for name in ("initial_z1", "initial_z2", "initial_masks1", "initial_masks2"):
        assert torch.equal(result[name], original[name])
    result["provenance"] = {"models": provenance, "protocol_sha256": file_hash(root / "protocol.json"),
                            "original_optimization_sha256": file_hash(args.parent / "optimization.pt")}
    torch.save(result, output)


def evaluate(args):
    from data.generate import ideal_mask
    from models.mlp import generate_fixed_sparsity_masks
    root = args.parent / "single_z_comparison"
    protocol = json.loads((root / "protocol.json").read_text())
    source = root / f"search_{args.pattern}.pt"
    result = torch.load(source, weights_only=True, map_location="cpu")
    assert result["provenance"]["protocol_sha256"] == file_hash(root / "protocol.json")
    assert result["provenance"]["original_masks_sha256"] == file_hash(args.parent / "masks.pt")
    original = torch.load(args.parent / "masks.pt", weights_only=True, map_location="cpu")
    masks = {name: original[name] for name in ("initial_vae1", "optimized_vae1", "random_search_vae1")}
    control = torch.load(root / "agreement_30.pt", weights_only=True, map_location="cpu")
    assert control["provenance"]["protocol_sha256"] == file_hash(root / "protocol.json")
    masks["agreement_30"] = control["final_masks1"]
    masks["single_z_final"] = result["final_masks"]
    masks["single_z_best_val"] = result["best_val_masks"]
    masks["random_exact32"] = generate_fixed_sparsity_masks(64, 8, 8, 32, 20260908)
    masks["ideal"] = ideal_mask().float()[None].repeat(64, 1, 1)
    for value in masks.values():
        assert value.shape == (64, 8, 8) and (value.sum((1, 2)) == 32).all()
        assert ((value == 0) | (value == 1)).all()
    output = root / f"eval_{args.pattern}.json"
    if output.exists():
        raise FileExistsError(output)
    metrics = eval_one_task(masks, args.pattern, protocol["evaluation"], torch.device(args.device))
    dump_json(output, {"pattern": args.pattern, "search_sha256": file_hash(source),
                       "protocol_sha256": file_hash(root / "protocol.json"),
                       "methods": metrics, "structure": {k: structure(v) for k, v in masks.items()}})
    print(json.dumps({k: {"accuracy": v["accuracy"]["mean"]} for k, v in metrics.items()}, indent=2), flush=True)


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = args.parent / "single_z_comparison"
    protocol = json.loads((root / "protocol.json").read_text())
    tasks = {}
    for pattern in protocol["patterns"]:
        task = json.loads((root / f"eval_{pattern}.json").read_text())
        assert task["search_sha256"] == file_hash(root / f"search_{pattern}.pt")
        assert task["protocol_sha256"] == file_hash(root / "protocol.json")
        tasks[pattern] = task
    names = list(next(iter(tasks.values()))["methods"])
    macro = {name: {"accuracy": describe([v["methods"][name]["accuracy"]["mean"] for v in tasks.values()]),
                    "bce": describe([v["methods"][name]["bce"]["mean"] for v in tasks.values()]),
                    "gold_iou": describe([v["structure"][name]["iou"]["mean"] for v in tasks.values()])}
             for name in names}
    comparisons = {}
    for variant in ("single_z_final", "single_z_best_val"):
        comparisons[f"agreement_minus_{variant}"] = {
            "accuracy": describe([v["methods"]["optimized_vae1"]["accuracy"]["mean"] - v["methods"][variant]["accuracy"]["mean"] for v in tasks.values()]),
            "gold_iou": describe([v["structure"]["optimized_vae1"]["iou"]["mean"] - v["structure"][variant]["iou"]["mean"] for v in tasks.values()])}
    dump_json(root / "summary.json", {"protocol": protocol, "tasks": tasks, "macro": macro, "comparisons": comparisons})
    labels = {"initial_vae1": "VAE42: prior", "optimized_vae1": "VAE42: agreement with VAE43",
              "agreement_30": "VAE42: agreement, 30 updates",
              "random_search_vae1": "VAE42: random pair search", "single_z_final": "VAE42: task-z final",
              "single_z_best_val": "VAE42: task-z best search-val", "random_exact32": "Random exact-32", "ideal": "Ideal support"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for axis, key, title in ((axes[0], "gold_iou", "Ideal-support IoU (post-hoc)"), (axes[1], "accuracy", "Fresh MLP test accuracy")):
        axis.barh(range(len(names)), [macro[n][key]["mean"] for n in names])
        axis.set_yticks(range(len(names)), [labels[n] for n in names])
        axis.invert_yaxis()
        axis.set(xlim=(0, 1), title=title)
        axis.grid(axis="x", alpha=.2)
    fig.suptitle("Same frozen VAE42, same initial z and final evaluation; different search objectives/budgets")
    fig.savefig(root / "comparison.png", dpi=160)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    lines = ["# Один VAE: task-loss поиск z против согласия двух decoder", "", "Дата: 2026-09-06.", "",
             "Используется один и тот же замороженный checkpoint VAE seed 42. Для согласия берутся "
             "только его маски из ранее выполненной пары VAE42/VAE43, без усреднения двух decoder. "
             "64 начальных z совпадают побитно. Все методы оцениваются свежими MLP при одинаковых "
             "начальных весах, minibatches и финальных данных на четырёх held-out задачах.", "",
             "## Протокол одиночного поиска", "",
             "Для каждой задачи: 30 внешних обновлений z; перед каждым — свежие 64 MLP, 300 шагов "
             "обучения на detached soft mask и 100 шагов с накоплением прямых градиентов по z; "
             "затем добавляется прямой градиент search-validation BCE. Внутренние обновления Adam "
             "не дифференцируются: это прежняя direct-gradient эвристика, не полный bilevel hypergradient. "
             "Маска — soft top-32, температура 0.5, радиус z=8, z learning rate 0.05.", "",
             "Отдельный search-validation набор: 1024 примера, seed=80000+pattern. Финальный test "
             "имеет seed=1000+pattern и не участвует в поиске/отборе. Синтетические входы могут "
             "повторяться; независимы потоки генерации, а не множество возможных битовых строк. "
             "Gold используется только для post-hoc IoU.", "",
             "Сохраняются последний z и отдельный вариант с наименьшим search-validation BCE. "
             "Сравнение не уравнивает вычислительный бюджет: single-z использует 12000 шагов MLP "
             "на задачу и target labels; согласие — 1000 decoder-only обновлений без меток. "
             "Отдельно выполнен контроль согласия с 30 внешними обновлениями, чтобы сравнить "
             "одинаковое число шагов z (но не вычислительную стоимость или доступ к меткам). "
             "В best-search-val отбираются pre-update состояния; последний post-update z "
             "оценивается отдельно как single-z final.", "",
             "## Результаты", "", "| Метод | Gold IoU | Test accuracy | BCE |", "|---|---:|---:|---:|"]
    for name in names:
        d = macro[name]
        lines.append(f"| {labels[name]} | {d['gold_iou']['mean']:.4f} | {d['accuracy']['mean']:.4f} | {d['bce']['mean']:.4f} |")
    lines += ["", "| Held-out задача | Agreement accuracy | Single-z final | Single-z best search-val |", "|---|---:|---:|---:|"]
    for pattern, task in tasks.items():
        d = task["methods"]
        lines.append(f"| {pattern} | {d['optimized_vae1']['accuracy']['mean']:.4f} | {d['single_z_final']['accuracy']['mean']:.4f} | {d['single_z_best_val']['accuracy']['mean']:.4f} |")
    lines += ["", "Один checkpoint и одно task split; разброс по четырём задачам не заменяет "
              "повторение с другими VAE seeds. Результат характеризует данную процедуру одиночного "
              "task-loss поиска, а не все возможные методы оптимизации latent.", "",
              "Артефакты: `protocol.json`, `search_<pattern>.pt`, `eval_<pattern>.json`, "
              "`summary.json`, `comparison.png/pdf`. Search artifacts сохранены до итоговой оценки "
              "и содержат checkpoint/source/data-protocol hashes.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({k: {m: v["mean"] for m, v in d.items()} for k, d in macro.items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--pattern", choices=["0100", "1011", "0000", "0011"])
    p.add_argument("--stage", choices=["search", "evaluate", "all", "report", "agreement_control"], default="all")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.stage not in ("report", "agreement_control") and args.pattern is None:
        p.error("--pattern is required for search/evaluation")
    if args.stage != "report" and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required; run outside sandbox")
    if args.stage in ("search", "all"):
        search(args)
    if args.stage in ("evaluate", "all"):
        evaluate(args)
    if args.stage == "report":
        report(args)
    if args.stage == "agreement_control":
        agreement_control(args)


if __name__ == "__main__":
    main()
