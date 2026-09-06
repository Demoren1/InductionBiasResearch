"""Oracle reachability test: optimize latent codes toward the known ideal mask.

This deliberately uses gold support in the objective and iterate selection.
It is a diagnostic of the SAME frozen decoders, not a label-free transfer
result. Failure to find an exact mask does not prove it is unreachable.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from data.generate import ideal_mask
from evaluation.decoder_agreement import align_columns, hard_topk, soft_topk
from evaluation.run_decoder_agreement import DEFAULT_ROOT, load_models


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_(z, radius):
    with torch.no_grad():
        z.mul_((radius / z.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1))


def optimize_ideal(model, initial_z, target, *, steps=1000, lr=.03,
                   temperature=.5, radius=8.):
    """Track best soft objective and best actual hard-mask witness separately."""
    if steps < 0 or lr <= 0 or radius <= 0:
        raise ValueError("Invalid optimization budget, learning rate, or radius")
    if target.ndim != 2 or not ((target == 0) | (target == 1)).all():
        raise ValueError("Target must be a binary matrix")
    k = int(target.sum())
    n = len(initial_z)
    model.eval().requires_grad_(False)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    z = torch.nn.Parameter(initial_z.detach().clone())
    project_(z, radius)
    start_z = z.detach().clone()
    target = target.to(z.device).float()[None].expand(n, -1, -1)
    optimizer = torch.optim.Adam([z], lr=lr)
    best_loss = z.new_full((n,), float("inf"))
    best_iou = z.new_full((n,), -1.)
    hard_tie_loss = best_loss.clone()
    best_soft_z, best_hard_z = z.detach().clone(), z.detach().clone()
    best_soft_steps = torch.zeros(n, dtype=torch.long, device=z.device)
    best_hard_steps = best_soft_steps.clone()
    history = []
    initial_loss = initial_iou = None

    def evaluate(codes):
        logits = model.decode(codes, codes.new_empty(len(codes), 0))
        soft = soft_topk(logits, k, temperature).reshape_as(target)
        aligned = align_columns(target, soft)
        loss = (aligned - target).square().mean((1, 2))
        with torch.no_grad():
            hard = hard_topk(logits.detach(), k).reshape_as(target)
            hard_aligned = align_columns(target, hard)
            intersection = (target * hard_aligned).sum((1, 2))
            iou = intersection / (2 * k - intersection)
        return loss, iou, soft, hard, hard_aligned

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, iou, _, _, _ = evaluate(z)
        if not torch.isfinite(loss).all():
            raise RuntimeError("Non-finite oracle loss")
        with torch.no_grad():
            if step == 0:
                initial_loss, initial_iou = loss.clone(), iou.clone()
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_soft_z[improved] = z[improved]
            best_soft_steps[improved] = step
            improved_hard = (iou > best_iou) | ((iou == best_iou) & (loss < hard_tie_loss))
            best_iou = torch.where(improved_hard, iou, best_iou)
            hard_tie_loss = torch.where(improved_hard, loss, hard_tie_loss)
            best_hard_z[improved_hard] = z[improved_hard]
            best_hard_steps[improved_hard] = step
            history.append({"step": step, "soft_loss_mean": float(loss.mean()),
                            "hard_iou_mean": float(iou.mean()),
                            "exact_now": int((iou == 1).sum()),
                            "exact_ever": int((best_iou == 1).sum())})
        if step % 100 == 0 or step == steps:
            print(f"[oracle radius={radius:g}] step={step}/{steps} "
                  f"loss={float(loss.mean()):.6f} IoU={float(iou.mean()):.4f} "
                  f"exact-ever={int((best_iou == 1).sum())}/{n}", flush=True)
        if step < steps:
            loss.sum().backward()
            optimizer.step()
            project_(z, radius)

    def pack(codes):
        with torch.no_grad():
            loss, iou, soft, hard, aligned = evaluate(codes)
        return {"z": codes.detach().cpu(), "loss": loss.cpu(), "iou": iou.cpu(),
                "soft": soft.cpu(), "hard": hard.cpu(), "aligned_hard": aligned.cpu()}

    for name, value in model.state_dict().items():
        if not torch.equal(before[name], value):
            raise AssertionError(f"Frozen decoder state changed: {name}")
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError("Frozen decoder received gradients")
    return {"initial": pack(start_z), "best_soft": pack(best_soft_z),
            "best_hard": pack(best_hard_z), "best_soft_steps": best_soft_steps.cpu(),
            "best_hard_steps": best_hard_steps.cpu(), "history": history,
            "decoder_unchanged": True,
            "settings": {"steps": steps, "lr": lr, "temperature": temperature, "radius": radius}}


def run(args):
    root = args.parent / "oracle_ideal"
    protocol = json.loads((root / "protocol.json").read_text())
    if args.radius not in protocol["radii"]:
        raise ValueError("Radius absent from the saved protocol")
    provenance = json.loads((args.parent / "search_provenance.json").read_text())
    if sha(args.parent / "masks.pt") != provenance["mask_sha256"]:
        raise ValueError("Original experiment mask provenance mismatch")
    models, meta = load_models(args.parent, torch.device(args.device))
    index = [record["seed"] for record in meta].index(args.model_seed)
    for actual, expected in zip(meta, provenance["models"]):
        if actual["sha256"] != expected["sha256"]:
            raise ValueError("VAE differs from the completed agreement experiment")
    original = torch.load(args.parent / "optimization.pt", weights_only=True, map_location="cpu")
    output = root / f"vae_{args.model_seed}_radius_{args.radius:g}.pt"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    initial_z = original[f"initial_z{index+1}"].to(args.device)
    if len(initial_z) != protocol["n_starts"]:
        raise ValueError("Wrong number of initial latent vectors")
    result = optimize_ideal(models[index], initial_z, ideal_mask(),
                            steps=protocol["steps"], lr=protocol["lr"],
                            temperature=protocol["temperature"], radius=args.radius)
    result["provenance"] = {"oracle": True, "checkpoint": meta[index],
                            "protocol_sha256": sha(root / "protocol.json"),
                            "initial_z_source_sha256": sha(args.parent / "optimization.pt"),
                            "original_masks_sha256": sha(args.parent / "masks.pt"),
                            "source_sha256": {"oracle_ideal.py": sha(__file__),
                                              "decoder_agreement.py": sha(Path(__file__).with_name("decoder_agreement.py"))},
                            "device": str(args.device), "torch_version": str(torch.__version__)}
    torch.save(result, output)
    print(f"Saved {output}", flush=True)


def stats(record, radius):
    iou = record["iou"]
    norms = record["z"].norm(dim=1)
    return {"mean_iou": float(iou.mean()), "max_iou": float(iou.max()),
            "min_iou": float(iou.min()), "exact_count": int((iou == 1).sum()),
            "mean_loss": float(record["loss"].mean()), "mean_z_norm": float(norms.mean()),
            "at_radius": int((norms >= radius - 1e-5).sum()), "per_start_iou": iou.tolist()}


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = args.parent / "oracle_ideal"
    protocol = json.loads((root / "protocol.json").read_text())
    previous = json.loads((args.parent / "summary.json").read_text())
    results, records = {}, {}
    for seed in protocol["model_seeds"]:
        for radius in protocol["radii"]:
            path = root / f"vae_{seed}_radius_{radius:g}.pt"
            record = torch.load(path, weights_only=True, map_location="cpu")
            assert record["decoder_unchanged"]
            assert record["provenance"]["protocol_sha256"] == sha(root / "protocol.json")
            key = f"vae_{seed}_radius_{radius:g}"
            records[key] = record
            results[key] = {stage: stats(record[stage], radius) for stage in ("initial", "best_soft", "best_hard")}
    payload = {"protocol": protocol, "results": results, "previous_agreement": {
        str(seed): previous["structure"][f"optimized_vae{i+1}"]["iou"]
        for i, seed in enumerate(protocol["model_seeds"])}}
    (root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for key, record in records.items():
        axes[0].plot([r["soft_loss_mean"] for r in record["history"]], label=key)
        axes[1].plot([r["hard_iou_mean"] for r in record["history"]], label=key)
    axes[0].set(xlabel="Adam updates", ylabel="Oracle soft MSE", yscale="log")
    axes[1].set(xlabel="Adam updates", ylabel="Mean hard-mask IoU vs ideal", ylim=(.5, 1.01))
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    fig.suptitle("Oracle optimization: same frozen decoders, gold explicitly used")
    fig.savefig(root / "convergence.png", dpi=160)
    fig.savefig(root / "convergence.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9), layout="constrained")
    for row, (key, record) in enumerate(records.items()):
        # First three starts, not best examples. All are aligned to gold for display.
        for col in range(3):
            axes[row, col].imshow(record["best_soft"]["aligned_hard"][col], cmap="Greys", vmin=0, vmax=1)
            axes[row, col].set_title(f"Start {col}: IoU {float(record['best_soft']['iou'][col]):.3f}", fontsize=9)
        axes[row, 3].imshow(ideal_mask(), cmap="Greys", vmin=0, vmax=1)
        axes[row, 3].set_title("Ideal target", fontsize=9)
        axes[row, 0].set_ylabel(key, fontsize=9)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Oracle results: first three starts, columns matched to gold")
    fig.savefig(root / "masks.png", dpi=160)
    plt.close(fig)
    lines = ["# Oracle: достижимость ideal через замороженные VAE", "", "Дата: 2026-09-06.", "",
             "Оптимизируются только latent-векторы тех же VAE seeds 42/43. Gold явно используется "
             "в loss и выборе итераций: это диагностика достижимости, не результат zero-shot переноса. "
             "Начальные z совпадают с исходными 64 стартами эксперимента на согласие.", "",
             "1000 шагов Adam, lr=0.03, soft top-32 с температурой 0.5, Hungarian по колонкам. "
             "Радиусы 8 и 16 запускаются независимо с одинаковых начальных z. "
             "Сохраняются два варианта: минимум soft loss и лучший бинарный IoU за всю траекторию.", "",
             "| VAE | Радиус | IoU до поиска | IoU после согласия | Oracle: средний IoU по best-soft | Oracle: максимальный IoU по best-soft | Точных по best-soft | Точных хотя бы раз |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, seed in enumerate(protocol["model_seeds"]):
        for radius in protocol["radii"]:
            result = results[f"vae_{seed}_radius_{radius:g}"]
            soft, hard = result["best_soft"], result["best_hard"]
            lines.append(f"| {seed} | {radius:g} | {result['initial']['mean_iou']:.4f} | "
                         f"{previous['structure'][f'optimized_vae{i+1}']['iou']['mean']:.4f} | "
                         f"{soft['mean_iou']:.4f} | {soft['max_iou']:.4f} | {soft['exact_count']}/64 | {hard['exact_count']}/64 |")
    lines += ["", "## Норма latent и soft loss", ""]
    for key, result in results.items():
        soft = result["best_soft"]
        lines.append(f"- {key}: soft MSE {soft['mean_loss']:.6f}; норма z {soft['mean_z_norm']:.3f}; на границе {soft['at_radius']}/64.")
    lines += ["", "Точный бинарный результат является конструктивным свидетельством достижимости ideal "
              "с точностью до перестановки колонок. Отсутствие точного результата за этот бюджет "
              "не доказывает недостижимость. Радиус 16 расширяет область поиска, но найденные там "
              "latents могут быть нетипичными для стандартного Gaussian prior.", "",
              "Checkpoint-файлы и тензоры decoder не изменены. Обучение новых VAE и downstream MLP "
              "в этой диагностике не выполняется.", "",
              "Артефакты: `protocol.json`, `vae_<seed>_radius_<R>.pt`, `summary.json`, "
              "`convergence.png/pdf`, `masks.png`. В `.pt` сохранены z, бинарные и мягкие маски, "
              "история, номера лучших итераций, хеши исходных checkpoint и протокола.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({k: {stage: {kk: vv for kk, vv in d.items() if kk != "per_start_iou"} for stage, d in v.items()} for k, v in results.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", choices=["run", "report"], default="run")
    parser.add_argument("--model_seed", type=int, choices=[42, 43], default=42)
    parser.add_argument("--radius", type=float, default=8.)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.stage == "run":
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Run with GPU access outside sandbox")
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
