"""Score only frozen outputs and preserve a compact, explicit comparison."""

import json
import shutil
from pathlib import Path

import numpy as np

from experiments.paper_structure_benchmark_v1.reporting.evaluate import (
    adjusted_rand,
    align_index,
    write_csv,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    save_json,
)


def report(base):
    frozen = json.loads((base / "frozen.json").read_text())
    for name, expected in frozen["files"].items():
        if digest(base / name) != expected:
            raise RuntimeError(f"Frozen output changed: {name}")
    project = Path(__file__).resolve().parents[2]
    output = project / "artifacts" / "anchor_start_2026-11-10"
    output.mkdir(exist_ok=False)
    mapping = json.loads((base / "target_map.json").read_text())
    rows, target_rows, arrays = [], [], {}
    for item in mapping.values():
        root = base / item["cohort"]
        case, condition, method = [
            item[k] for k in ("case", "condition", "method")
        ]
        number = int(case.rsplit("_", 1)[1])
        family = (
            ("pattern" if number < 3 else "deepsets")
            if item["cohort"] == "standard"
            else ("general_local", "general_quartet_mix", "general_random")[
                number // 2
            ]
        )
        key = f"{case}_c{condition}"
        sizes = json.loads((root / "public" / f"{case}.json").read_text())
        codes = sizes["codes"]
        folder = root / "methods" / method / "fits" / key
        chosen = dict(np.load(folder / "joint_selected.npz"))
        index = chosen["joint_functional_index"]
        truth = np.load(root / "private" / f"{case}.npz")["index"]
        aligned, recovery = align_index(index, truth, codes)
        overlap = (truth != 0) & (aligned != 0)
        initial = np.load(folder / "fitted.npz")["indices"]
        proposals = np.load(folder / "joint_bank.npz")["indices"]
        truth_key = canonical_index(truth, codes).tobytes()
        metadata = json.loads((folder / "joint_bank.json").read_text())
        selected_metadata = json.loads(
            (folder / "joint_selected.json").read_text()
        )
        selected_meta = selected_metadata["methods"]["joint_functional"][
            "meta_mse"
        ]
        predictions = np.load(
            root / "target_cache" / item["cache"] / "predictions.npz"
        )
        private_targets = np.load(root / "private" / f"{case}_targets.npz")
        target = np.stack([private_targets[f"query_y_{t}"] for t in range(8)])
        losses = ((predictions["prediction"] - target) ** 2).mean((1, 2))
        adam_losses = ((predictions["adam_prediction"] - target) ** 2).mean(
            (1, 2)
        )
        entry = {
            "family": family,
            "case": case,
            "replicate": number % (3 if item["cohort"] == "standard" else 2),
            "source_noise": condition
            * (0.1 if item["cohort"] == "standard" else 0.2),
            "method": method,
            **recovery,
            "sharing_ari_on_overlap": adjusted_rand(
                np.abs(aligned[overlap]), np.abs(truth[overlap])
            ),
            "selected_in_initial_bank": any(
                np.array_equal(index, a) for a in initial
            ),
            "true_in_initial_bank": any(
                a.tobytes() == truth_key for a in initial
            ),
            "true_in_projection_bank": any(
                a.tobytes() == truth_key for a in proposals
            ),
            "projection_calls": metadata["projection_calls"],
            "projection_candidates": len(proposals),
            "source_meta_mse": selected_meta,
            "target_mse_1024": float(losses.mean()),
            "target_adam_mse_1024": float(adam_losses.mean()),
            "numerical_parameters": codes
            + sizes["hidden"] * (1 + sizes["outputs"])
            + sizes["outputs"],
        }
        rows.append(entry)
        for task, (mse, adam) in enumerate(zip(losses, adam_losses)):
            target_rows.append(
                {
                    "family": family,
                    "case": case,
                    "condition": condition,
                    "method": method,
                    "task": task,
                    "mse": float(mse),
                    "adam_mse": float(adam),
                }
            )
        prefix = f"{item['cohort']}_{key}_{method}"
        arrays[prefix + "_index"] = index
        arrays[prefix + "_aligned"] = aligned
        arrays[prefix + "_truth"] = truth
        arrays[prefix + "_source_parameters"] = chosen[
            "joint_functional_parameters"
        ]
        for name in (
            "joint_selected.json",
            "joint_bank.json",
            "joint_fitted.json",
        ):
            destination = (
                output / "records" / item["cohort"] / key / method / name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / name, destination)
    rows.sort(
        key=lambda r: (r["family"], r["case"], r["source_noise"], r["method"])
    )
    write_csv(output / "recovery.csv", rows)
    write_csv(output / "transfer_per_task.csv", target_rows)
    np.savez_compressed(output / "selected_structures.npz", **arrays)
    text = [
        "# Один запуск проекции из каждого якоря",
        "",
        "Свежая парная проверка; численные гиперпараметры не подбирались по результатам.",
        "16 source и 8 target-задач на структуру; все target-числа обучены заново.",
        "",
        "| Семейство | Source noise | Метод | Exact I | Exact mask | Target MSE, N=1024 | Выбран из B0 | Вызовов проекции |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for family in sorted({r["family"] for r in rows}):
        for noise in sorted(
            {r["source_noise"] for r in rows if r["family"] == family}
        ):
            for method in ("full", "anchor_fixed", "anchor_free"):
                group = [
                    r
                    for r in rows
                    if (r["family"], r["source_noise"], r["method"])
                    == (family, noise, method)
                ]
                total = len(group)
                text.append(
                    f"| {family} | {noise:g} | {method} | "
                    f"{sum(r['exact'] for r in group)}/{total} | "
                    f"{sum(r['mask_exact'] for r in group)}/{total} | "
                    f"{np.mean([r['target_mse_1024'] for r in group]):.6g} | "
                    f"{sum(r['selected_in_initial_bank'] for r in group)}/{total} | "
                    f"{np.mean([r['projection_calls'] for r in group]):g} |"
                )
    text.extend(
        [
            "",
            "`full` — исходный G34; `anchor_fixed` — один вызов на anchor с его support;",
            "`anchor_free` — один вызов на anchor с разрешённым нулём на всех позициях.",
            "Во всех вариантах сохранены B0 и исходная обработка знаковых ориентаций.",
            "Уменьшаются и число стартов, и набор support proposals; их отдельный причинный вклад не разделён.",
            "Если выбран исходный B0, успех нельзя приписывать совместной проекции.",
            "Wall time измерен при чужой нагрузке на GPU и не является выделенным hardware benchmark.",
            "Torch FLOPs в этой быстрой проверке не измерялись; CPU-геометрия также не включена в FLOPs.",
            "Source/model checkpoints и данные лежат во временной директории, старые артефакты не изменялись.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(text) + "\n")
    for name in (
        "registration.json",
        "PROTOCOL.md",
        "frozen.json",
        "state.json",
        "target_map.json",
    ):
        shutil.copy2(base / name, output / name)
    shutil.copytree(base / "code_snapshot", output / "code_snapshot")
    save_json(output / "summary.json", {"rows": rows, "run_root": str(base)})
    print(json.dumps({"report": str(output / "REPORT.md"), "rows": len(rows)}))
