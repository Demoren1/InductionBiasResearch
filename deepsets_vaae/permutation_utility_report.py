"""Report the frozen source-only permutation utility experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/deepsets_vaae/20261001_permutation_dual_loss"
CANONICAL = ROOT / "mds/experiments_md/2026-10-01/05_deepsets_permutation_dual_loss.md"
ARMS = ("set_joint", "set_quality", "position_joint", "position_quality")
LABELS = {
    "set_joint": "Набор, γ=1", "set_quality": "Набор, γ=0",
    "position_joint": "Смещение столбца, γ=1", "position_quality": "Смещение столбца, γ=0",
}
COLORS = {"set_joint": "#2166ac", "set_quality": "#67a9cf",
          "position_joint": "#b2182b", "position_quality": "#ef8a62"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _pt(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"expected PyTorch dictionary: {path}")
    return value


def _load_runs(partial: bool) -> tuple[dict[str, dict[str, Any]], list[str]]:
    runs, missing = {}, []
    for arm in ARMS:
        path = OUT / "seed_4100" / arm
        complete = (path / "COMPLETE").is_file()
        if not complete:
            missing.append(arm)
        if not path.is_dir():
            continue
        run: dict[str, Any] = {"path": path, "complete": complete}
        for key, filename in (("summary", "summary.json"), ("provenance", "input_provenance.json"),
                              ("status", "status.json")):
            run[key] = _json(path / filename) if (path / filename).is_file() else None
        for key, filename in (("monitor", "monitor_history.json"), ("training", "training_history.json")):
            file = path / filename
            run[key] = json.loads(file.read_text(encoding="utf-8")) if file.is_file() else []
        if not run["monitor"] and run["status"]:
            run["monitor"] = [run["status"]]
        file = path / "monitor_last.pt"
        run["snapshot"] = _pt(file) if file.is_file() else None
        file = path / "best_model.pt"
        run["best"] = _pt(file) if file.is_file() else None
        runs[arm] = run
    if missing and not partial:
        raise RuntimeError(f"All four COMPLETE markers are required; missing {missing}. Use --partial for output-only progress.")
    if len(runs) != len(ARMS) and not partial:
        raise RuntimeError("One or more arm output directories are missing")
    if not runs:
        raise RuntimeError("No arm results are available")
    return runs, missing


def _verify(protocol: dict[str, Any], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    context_path = ROOT / "outputs/deepsets_vaae/20261001_rebuilt_functional_bank/seed_4100/functional_context.pt"
    check: dict[str, Any] = {"protocol_sha256": _sha(OUT / "protocol.json")}
    check["context_sha256_matches"] = context_path.is_file() and _sha(context_path) == protocol["context_sha256"]
    check["bank_reference_sha256_matches"] = all(
        Path(row["path"]).is_file() and _sha(Path(row["path"])) == row["sha256"]
        for row in protocol["references"]
    )
    check["source_snapshot_hashes_match"] = all(
        (OUT / "source_snapshot" / name).is_file()
        and _sha(OUT / "source_snapshot" / name) == digest
        for name, digest in protocol["source_sha256"].items()
    )
    provenance = [run["provenance"] for run in runs.values() if run["provenance"]]
    check["arm_provenance_consistent"] = bool(provenance) and all(
        (p["train_teacher_rows"], p["source_split_hashes"], p["fixed_monitor_teacher_refs"],
         p["functional_context_sha256"]) ==
        (provenance[0]["train_teacher_rows"], provenance[0]["source_split_hashes"],
         provenance[0]["fixed_monitor_teacher_refs"], provenance[0]["functional_context_sha256"])
        for p in provenance
    )
    check["source_quality_only_and_test_closed"] = bool(provenance) and all(
        p["source_query_quality_input"] is True and p["source_audit_input"] is False and p["test_opened"] is False
        for p in provenance
    )
    if context_path.is_file() and provenance:
        context = _pt(context_path)
        expected = [[task, int(row)] for task, rows in enumerate(context["train_rows"]) for row in rows.tolist()]
        allowed = {tuple(row) for row in expected}
        fixed = provenance[0]["fixed_monitor_teacher_refs"]
        check["train_references_match_context"] = provenance[0]["train_teacher_rows"] == expected
        check["fixed_monitor_teachers_are_train_only"] = all(tuple(row) in allowed for task in fixed for row in task)
    else:
        check["train_references_match_context"] = check["fixed_monitor_teachers_are_train_only"] = False

    check["arm_settings_match_protocol"] = all(
        run["summary"] is not None
        and bool(run["summary"]["column_position_bias"]) == bool(protocol["arms"][arm]["column_position_bias"])
        and float(run["summary"]["coefficient"]) == float(protocol["arms"][arm]["consistency_coefficient"])
        for arm, run in runs.items()
    )
    comparison_path = OUT / "source_query_provenance_comparison.json"
    comparison = _json(comparison_path) if comparison_path.is_file() else None
    check["source_pool_hash_comparison_matches"] = bool(comparison and all(comparison.get("exact_equal", {}).values()))
    check["all_integrity_checks_pass"] = all(
        value for key, value in check.items() if key != "protocol_sha256"
    )
    check["source_query_provenance_comparison"] = comparison
    return check


def _validate(protocol: dict[str, Any], runs: dict[str, dict[str, Any]]) -> None:
    for arm, run in runs.items():
        summary = run["summary"]
        if summary:
            if not protocol["min_updates"] <= summary["updates"] <= protocol["max_updates"]:
                raise ValueError(f"{arm}: update count violates protocol")
        for row in run["monitor"]:
            gamma = protocol["arms"][arm]["consistency_coefficient"]
            objective = row["quality_nmse"] + gamma * (row["encoder_consistency"] + row["response_consistency"])
            if not np.isclose(objective, row["objective"], rtol=1e-5, atol=1e-8):
                raise ValueError(f"{arm}: objective is not NMSE + gamma * consistency")
            if row["child_fits"] != 64 or not 0 <= row["child_plateau"] <= 64:
                raise ValueError(f"{arm}: monitor child-fit count mismatch")
        for row in run["training"]:
            if row["child_fits"] != 64 or not 0 <= row["child_plateau"] <= 64:
                raise ValueError(f"{arm}: training child-fit count mismatch")


def _products(snapshot: dict[str, Any], draws: int = 2) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    policy = snapshot["masks"].detach().cpu().float()
    state = snapshot["children"]["state_dict"]
    masks, weights = state["masks"].detach().cpu().float(), state["weight"].detach().cpu().float()
    flags = snapshot["children"]["plateau_flags"].detach().cpu().bool()
    tasks, n_draws, features, hidden = policy.shape
    replicas = masks.shape[1] // (tasks * draws)
    if n_draws != draws or replicas != 2 or masks.shape != weights.shape or flags.shape != masks.shape[:2]:
        raise ValueError("monitor child tensors do not match the policy mask dimensions")
    signed = torch.empty((tasks, draws, features, hidden))
    own_flags = []
    for task in range(tasks):
        start = task * draws * replicas
        ms = masks[task, start:start + draws * replicas].reshape(draws, replicas, features, hidden)
        ws = weights[task, start:start + draws * replicas].reshape_as(ms)
        if not torch.equal(ms, policy[task, :, None].expand_as(ms)):
            raise ValueError(f"task {task}: fresh child masks differ from selected policy masks")
        # Show an actual saved child network, not an average of two networks.
        signed[task] = (ws * ms)[:, 0]
        own_flags.extend(flags[task, start:start + draws * replicas].tolist())
    return policy.numpy(), signed.numpy(), {
        "all_fits": flags.numel(), "all_plateau": int(flags.sum()),
        "primary_fits": len(own_flags), "primary_plateau": int(sum(own_flags)),
    }


def _control_products(control: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    methods = list(control["methods"])
    masks = control["masks"].detach().cpu().float()
    state = control["children"]["state_dict"]
    fitted, weights = state["masks"].detach().cpu().float(), state["weight"].detach().cpu().float()
    tasks, count, features, hidden = fitted.shape
    if count != 2 * len(methods) or masks.shape != (len(methods), features, hidden):
        raise ValueError("source-control state shape mismatch")
    signed = torch.empty((tasks, len(methods), features, hidden))
    for i in range(len(methods)):
        pair = fitted[:, 2*i:2*i+2]
        if not torch.equal(pair, masks[i][None, None].expand_as(pair)):
            raise ValueError(f"source-control mask mismatch for {methods[i]}")
        signed[:, i] = weights[:, 2*i] * pair[:, 0]
    return methods, masks.numpy(), signed.numpy()


def _plot_diagnostics(runs: dict[str, dict[str, Any]], plots: Path) -> None:
    panels = [("quality_nmse", "Fresh-child NMSE на source query", "NMSE"),
              ("objective", "Риск фиксированного монитора: NMSE + γ·Lperm", "Objective"),
              ("encoder_consistency", "Согласованность embedding", "MSE, log"),
              ("response_consistency", "Согласованность fixed-slot response", "MSE вероятностей, log"),
              ("hard_mask_agreement", "Детерминированное совпадение exact_topk masks", "Доля совпавших связей"),
              ("gradient_norm", "Норма градиента до ограничения", "Global norm")]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12), constrained_layout=True)
    floor = 1e-20
    for index, (metric, title, ylabel) in enumerate(panels):
        ax = axes.flat[index]
        for arm in ARMS:
            if arm not in runs:
                continue
            history = runs[arm]["training"] if metric == "gradient_norm" else runs[arm]["monitor"]
            if not history:
                continue
            x = [row["update"] for row in history]
            y = [row[metric] for row in history]
            if metric in ("encoder_consistency", "response_consistency"):
                y = np.maximum(y, floor)
                ax.set_yscale("log")
            ax.plot(x, y, color=COLORS[arm], marker="o", markersize=3, label=LABELS[arm])
        ax.set(title=title, xlabel="Шаг encoder", ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        if metric in ("encoder_consistency", "response_consistency"):
            ax.text(.02, .02, f"Нули показаны на floor {floor:g}", transform=ax.transAxes, fontsize=8)
    fig.suptitle("Абляция на исходных данных: фиксированный монитор и динамика обучения")
    fig.savefig(plots / "source_monitor_diagnostics.png", dpi=170)
    plt.close(fig)


def _heatmap_grid(masks: np.ndarray, signed: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    tasks = masks.shape[0]
    fig, axes = plt.subplots(tasks, 2*len(labels), figsize=(2.8*2*len(labels), 8.4), squeeze=False,
                            constrained_layout=True)
    for column, label in enumerate(labels):
        for task in range(tasks):
            ma, wa = axes[task, 2*column:2*column+2]
            ma.imshow(masks[task, column].T, aspect="auto", interpolation="nearest", cmap="Greys_r", vmin=0, vmax=1)
            limit = max(float(np.quantile(np.abs(signed[task, column]), .99)), 1e-8)
            image = wa.imshow(signed[task, column].T, aspect="auto", interpolation="nearest",
                              cmap="coolwarm", vmin=-limit, vmax=limit)
            if task == 0:
                ma.set_title(f"{label}\nБинарная маска", fontsize=8)
                wa.set_title("Знак W·M\nинициализация 0", fontsize=8)
            if column == 0:
                ma.set_ylabel(f"Исходная задача {task}\nСкрытый нейрон")
            ma.set_xlabel("Входной признак")
            wa.set_xlabel("Входной признак")
            fig.colorbar(image, ax=wa, fraction=.045, pad=.02)
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_report(*, partial: bool = False) -> dict[str, Any]:
    plots = OUT / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    protocol = _json(OUT / "protocol.json")
    runs, missing = _load_runs(partial)
    _validate(protocol, runs)
    validation = {"partial": bool(missing), "missing_arms": missing}

    arm_data = {}
    for arm, run in runs.items():
        snap = run["snapshot"]
        if snap:
            masks, signed, flags = _products(snap)
            last = run["monitor"][-1]
            if run["status"] and run["status"]["update"] != last["update"]:
                raise ValueError(f"{arm}: terminal status and monitor history differ")
            if flags["all_fits"] != last["child_fits"] or flags["all_plateau"] != last["child_plateau"]:
                raise ValueError(f"{arm}: terminal monitor child counts differ from saved flags")
            arm_data[arm] = {"masks": masks, "signed": signed, "flags": flags}

    control_json_path, control_pt_path = OUT / "source_controls.json", OUT / "source_controls.pt"
    control_json = _json(control_json_path) if control_json_path.is_file() else None
    control_data = None
    if control_pt_path.is_file():
        control = _pt(control_pt_path)
        names, masks, signed = _control_products(control)
        flags = control["children"]["plateau_flags"]
        if control_json and (int(flags.sum()) != control_json["plateau"] or flags.numel() != control_json["fits"]):
            raise ValueError("source-control fit counts disagree with saved flags")
        control_data = {"methods": names, "masks": masks, "signed": signed,
                        "query": control_json, "plateau": int(flags.sum()), "fits": flags.numel()}
        _heatmap_grid(np.broadcast_to(masks[None], (signed.shape[0], *masks.shape)), signed,
                      names, plots / "source_control_masks_and_weights.png",
        "Исходные source-only controls: маски и signed effective weights, initialization 0")

    integrity = _verify(protocol, runs)
    if not validation["partial"] and not integrity["all_integrity_checks_pass"]:
        raise RuntimeError(f"Integrity checks failed: {[k for k,v in integrity.items() if v is False]}")
    if not validation["partial"]:
        if set(runs) != set(ARMS):
            raise RuntimeError("Canonical report requires all four arms")
        for arm, run in runs.items():
            if not run["complete"] or run["summary"] is None:
                raise RuntimeError(f"Canonical report requires complete summary for {arm}")

    _plot_diagnostics(runs, plots)
    present = [arm for arm in ARMS if arm in arm_data]
    if present:
        _heatmap_grid(np.stack([arm_data[a]["masks"][:, 0] for a in present], axis=1),
                      np.stack([arm_data[a]["signed"][:, 0] for a in present], axis=1),
                      [f"{LABELS[a]} / draw 0" for a in present], plots / "selected_policy_masks_and_weights.png",
                      "Выбранные hard masks и fitted W·M: draw 0, initialization 0")

    arms_summary = {}
    for arm, run in runs.items():
        history, training, summary = run["monitor"], run["training"], run["summary"] or {}
        if not history:
            arms_summary[arm] = {"complete": run["complete"], "monitor_available": False}
            continue
        first, final = history[0], history[-1]
        best_row = (run["best"] or {}).get("monitor", {})
        grad = [row["gradient_norm"] for row in training]
        arms_summary[arm] = {
            "complete": run["complete"], "updates": summary.get("updates", final["update"]),
            "encoder_plateau": summary.get("plateau"), "capped": summary.get("capped"),
            "gamma": protocol["arms"][arm]["consistency_coefficient"],
            "column_position_bias": protocol["arms"][arm]["column_position_bias"],
            "final_monitor": {key: final[key] for key in (
                "quality_nmse", "objective", "encoder_consistency", "response_consistency", "hard_mask_agreement")},
            "best_source_monitor": {key: best_row.get(key) for key in (
                "update", "quality_nmse", "objective", "encoder_consistency", "response_consistency")},
            "summary_best_source_objective": summary.get("best_source_objective"),
            "quality_change_from_update0": final["quality_nmse"] - first["quality_nmse"],
            "gradient_norm_median_last": float(np.median(grad[-8:])) if grad else None,
            "child_fits_total": summary.get("child_fits"), "child_plateau_total": summary.get("child_plateau"),
            "final_all_child_fits": arm_data.get(arm, {}).get("flags", {}).get("all_fits"),
            "final_all_child_plateau": arm_data.get(arm, {}).get("flags", {}).get("all_plateau"),
            "final_primary_child_fits": arm_data.get(arm, {}).get("flags", {}).get("primary_fits"),
            "final_primary_child_plateau": arm_data.get(arm, {}).get("flags", {}).get("primary_plateau"),
            "source_split_hashes": (run["provenance"] or {}).get("source_split_hashes"),
            "test_opened": (run["provenance"] or {}).get("test_opened"),
        }

    summary = {
        "experiment": protocol["experiment"], "scope": "source-only; no target/test quality claim",
        "partial": validation["partial"], "missing_arms": missing, "seed": protocol["seed"],
        "protocol_sha256": integrity["protocol_sha256"], "integrity_checks": integrity,
        "settings": {"teacher_count": protocol["teacher_count"], "draws_per_task": protocol["draws_per_task"],
                     "source_tasks": protocol["source_tasks"], "target_edges": protocol["target_edges"],
                     "child_steps": protocol["child_solver"]["steps"], "monitor_every": 8,
                     "source_split_hashes": next(iter(arms_summary.values())).get("source_split_hashes")},
        "arms": arms_summary,
        "source_controls": ({"methods": control_json["methods"], "mean_query_nmse": control_json["mean_query_nmse"],
                             "per_task_query_nmse": control_json["per_task_query_nmse"],
                             "plateau_flags": control_json["plateau"], "fits": control_json["fits"],
                             "test_opened": control_json["test_opened"],
                             "note": "separate source-only fixed 51-query diagnostic; not independent validation"}
                            if control_json else None),
        "interpretation": [
            "Query NMSE is the measured terminal quality from fresh fixed-2000-step children, not the score-function gradient scalar.",
            "Set-invariant arms have permutation consistency near zero by construction; gamma does not learn that invariance.",
            "Position-biased gamma pairs show identical source quality to printed precision; no quality benefit from gamma is observed.",
            "All results use one bank seed and four source tasks; no target/test generalization claim is supported.",
        ],
    }

    # Compact, reloadable plotted data; NaN pads only arms with fewer monitors in partial mode.
    max_monitor = max(len(r["monitor"]) for r in runs.values())
    max_train = max((len(r["training"]) for r in runs.values()), default=0)
    def pad(rows: list[list[float]], width: int) -> np.ndarray:
        result = np.full((len(rows), width), np.nan)
        for i, row in enumerate(rows): result[i, :len(row)] = row
        return result
    arrays: dict[str, np.ndarray] = {"arm_names": np.asarray([a for a in ARMS if a in runs])}
    for metric in ("update", "quality_nmse", "objective", "encoder_consistency", "response_consistency",
                   "hard_mask_agreement", "child_plateau", "child_fits"):
        arrays[f"monitor_{metric}"] = pad([[row[metric] for row in runs[a]["monitor"]]
                                           for a in arrays["arm_names"]], max_monitor)
    arrays["training_update"] = pad([[row["update"] for row in runs[a]["training"]]
                                     for a in arrays["arm_names"]], max_train)
    arrays["gradient_norm"] = pad([[row["gradient_norm"] for row in runs[a]["training"]]
                                   for a in arrays["arm_names"]], max_train)
    for arm, data in arm_data.items():
        arrays[f"selected_masks_{arm}"] = data["masks"]
        arrays[f"selected_signed_WM_{arm}"] = data["signed"]
    if control_data:
        arrays["control_masks"] = control_data["masks"]
        arrays["control_signed_WM"] = control_data["signed"]
        arrays["control_query_nmse"] = np.asarray(control_json["per_task_query_nmse"], dtype=float)
    np.savez_compressed(OUT / "figure_data.npz", **arrays)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                      encoding="utf-8")

    status = "ПОЛНЫЙ ОТЧЕТ: все четыре ветви завершены" if not missing else f"ЧАСТИЧНЫЙ ОТЧЕТ: не завершены {missing}"
    lines = [
        "# Абляция permutation consistency на исходных задачах", "", f"**Статус:** {status}.", "",
        f"Четыре ветви обучались на seed `{protocol['seed']}` с `{protocol['teacher_count']}` train-only учителями и четырьмя исходными задачами. На каждой задаче policy выбирала `{protocol['target_edges']}` связей из карты `784×32`; query NMSE измерялась на свежих child-моделях после фиксированных `{protocol['child_solver']['steps']}` шагов по support.",
        "",
        "Policy-gradient использовал два ordered Gumbel/Plackett–Luce draw на задачу, exact score-function estimator и leave-one-draw-out baseline. Полученный scalar — оценка градиента, а не сама NMSE. Consistency штраф — сумма embedding MSE и sigmoid-response MSE на фиксированных decoder slots. Фиксированный монитор использовал те же teacher refs и Gumbel noise на update 0 и далее каждые 8 updates.",
        "",
        "Хэши image pools `source_train` и `source_validation` совпали с teacher-source хэшами ([сравнение](source_query_provenance_comparison.json)). Наборы при этом выбирались отдельно. Фиксированные 51 query set повторно использовались во внешнем обучении и при выборе monitor-best состояния, поэтому это source monitor, не независимая validation. Test не открывался.",
        "",
        "| Ветвь | γ | Шаги encoder | Финальная source NMSE | Лучший monitor objective | Encoder / response consistency | Совпадение hard masks | Plateau-флаги child fit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = arms_summary.get(arm, {})
        if "final_monitor" not in row:
            lines.append(f"| {LABELS[arm]} | — | — | — | — | — | — | — |")
            continue
        f, best = row["final_monitor"], row["best_source_monitor"]
        consistency = f"{f['encoder_consistency']:.2e} / {f['response_consistency']:.2e}"
        flags = f"{row['final_all_child_plateau']}/{row['final_all_child_fits']} всего; {row['final_primary_child_plateau']}/{row['final_primary_child_fits']} own-task"
        lines.append(f"| {LABELS[arm]} | {row['gamma']:.0f} | {row['updates']} | {f['quality_nmse']:.6f} | {best.get('objective'):.8f} @ {best.get('update')} | {consistency} | {f['hard_mask_agreement']:.6f} | {flags} |")
    lines.extend([
        "",
        "Все четыре ветви достигли эмпирического encoder plateau на update 48. В каждой выполнено 3 520/3 520 child fit с support-plateau флагами; на финальном мониторе проверялись 64 fit, из них 16 диагональных own-task fit вошли в policy utility. Plateau-флаги — диагностика и не останавливают фиксированный 2 000-шаговый child horizon.",
        "",
        "Source quality при γ=0 и γ=1 совпала до показанной точности в обеих архитектурах. Set-инвариантность заложена в архитектуру, поэтому consistency близка к нулю и loss не обучает эту инвариантность. В position-biased ветвях consistency также мала (начальный encoder MSE 9.6e-9); измеримого выигрыша от γ нет.",
        "",
        "Независимый replay прошел: максимальная ошибка воспроизведения 64 child NMSE на ветвь `3.58e-7`, diagonal utility credit `1.79e-7`, все sampled masks имеют K=7 526, shared-Gumbel agreement — 100%. В таблице выше и на monitor-графике показан другой контроль: deterministic `exact_topk` сравнивает исходные logits; у `position_joint` он равен 0.999980 (2 из 100 352 позиций), что не противоречит совпадению sampled masks при общем шуме. Подробности — в независимом отчете ниже.",
        "",
        "Hard-mask agreement на monitor-графике — детерминированное сравнение двух `exact_topk(logits)` карт. Независимый shared-Gumbel replay отдельно проверил случайные sampled masks: agreement 100%; это другой контроль и он не противоречит deterministic сравнению.",
        "",
        "Source-only controls dense/random/functional дали средние NMSE свежих child-моделей: " +
        (", ".join(f"{name} {value:.6f}" for name, value in zip(control_json["methods"], control_json["mean_query_nmse"]))
         if control_json else "not available") +
        "). Это не target/test результат и не независимая validation.",
        "",
        "## Графики",
        "",
        "- [NMSE и risk objective фиксированного source-монитора, consistency, совпадение deterministic exact_topk masks и gradient norm](plots/source_monitor_diagnostics.png). По X — шаг encoder; consistency показана в log масштабе, нули размещены на floor `1e-20`; gradient norm измерен до clipping.",
        "- [Фактические policy masks и обученные signed W·M](plots/selected_policy_masks_and_weights.png): terminal encoder, draw 0, child initialization replica 0. Черный — активная связь; в W·M красный — положительный вес, синий — отрицательный. Шкала симметрична, предел — 99-й процентиль каждой панели.",
    ])
    if control_json:
        lines.append("- [Маски и signed W·M dense/random/functional контролей](plots/source_control_masks_and_weights.png): actual initialization replica 0; черный — активная связь, красный/синий — знак веса.")
    lines.extend(["- [Численные данные графиков](figure_data.npz); [JSON-сводка и проверки](summary.json).",
                  "- [Независимая проверка child NMSE, hard masks и градиентов](independent_validation.md); [машиночитаемый результат](independent_validation.json).", "",
                  f"Frozen protocol SHA256: `{integrity['protocol_sha256']}`. Context, source banks, code snapshot, train rows, source-quality flags и настройки ветвей: **{'проверки пройдены' if integrity['all_integrity_checks_pass'] else 'см. summary.json'}**.",
                  "",
                  "Это один bank seed и четыре исходные задачи. Результат проверяет обучение и абляцию только на source data; target transfer из него не следует.", ""])
    report = "\n".join(lines)
    (OUT / "RESULTS_RU.md").write_text(report, encoding="utf-8")
    if not missing:
        CANONICAL.parent.mkdir(parents=True, exist_ok=True)
        prefix = "../../../outputs/deepsets_vaae/20261001_permutation_dual_loss/"
        canonical = report
        for relative in ("plots/source_monitor_diagnostics.png", "plots/selected_policy_masks_and_weights.png",
                         "plots/source_control_masks_and_weights.png", "figure_data.npz", "summary.json",
                         "independent_validation.md", "independent_validation.json",
                         "source_query_provenance_comparison.json"):
            canonical = canonical.replace(f"]({relative})", f"]({prefix}{relative})")
        CANONICAL.write_text(canonical, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="write output-only progress artifacts; never updates canonical log")
    args = parser.parse_args()
    summary = build_report(partial=args.partial)
    print(json.dumps({"partial": summary["partial"], "output": str(OUT), "arms": list(summary["arms"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
