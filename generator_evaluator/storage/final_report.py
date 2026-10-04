"""Markdown report of the frozen mask and its measured controls."""
from pathlib import Path

import numpy as np
import torch

from generator_evaluator.storage.artifacts import _atomic_write


def write_final_report(out, summary, methods, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    folder = out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    functional = "functional_mean" if "functional_mean" in methods else "functional_consensus"
    names = ["common", "dense", "random", functional]
    labels = ["Финальная маска", "Dense", "Random", "Functional mean"]
    metric = "BCE" if summary["domain"] == "pattern" else "NMSE"
    tests = summary["test_task_ids"]
    selection = [task for task in summary["final"] if task not in tests]
    lines = ["# Финальный отчёт", "",
             f"Реальная query-ошибка ({metric}); меньше — лучше. Финальная маска выбрана до открытия test.", "",
             "Functional mean — одна top-K маска по средней функциональной карте обучающих задач: "
             "карты нормированы по столбцам, банки имеют одинаковый вес.", "",
             "## Качество", "",
             "Столбцы показывают среднюю ошибку; отрезки — стандартное отклонение по инициализациям на фиксированной задаче.", ""]
    figures = []
    for split, tasks, title in [("selection", selection, "Selection"), ("test", tests, "Test")]:
        if not tasks:
            continue
        fig, axis = plt.subplots(figsize=(max(8, len(tasks) * .85), 4.8))
        x = np.arange(len(tasks))
        for index, (name, label) in enumerate(zip(names, labels)):
            rows = [summary["final"][task][name] for task in tasks]
            means = [row["query_error"] for row in rows]
            std = [np.std(row["replica_losses"]) for row in rows]
            axis.bar(x + (index - 1.5) * .2, means, .2, yerr=std,
                     capsize=2, label=label)
        axis.set_xticks(x, [task.removeprefix("pattern:").removesuffix(":selection").removesuffix(":test") for task in tasks], rotation=35, ha="right")
        axis.set(xlabel="Задача", ylabel=f"Query {metric} (меньше — лучше)", title=title)
        axis.legend(fontsize=8)
        fig.tight_layout()
        relative = f"figures/final_quality_{split}.png"
        fig.savefig(out / relative, dpi=160)
        plt.close(fig)
        figures.append(relative)
        lines.extend([f"### {title}", "", f"![Качество на {title}]({relative})", "",
                      "| Задача | Финальная маска | Dense | Random | Functional mean |",
                      "|---|---:|---:|---:|---:|"])
        for task in tasks:
            values = [summary["final"][task][name]["query_error"] for name in names]
            lines.append(f"| {task} | " + " | ".join(f"{value:.6g}" for value in values) + " |")
        lines.append("")

    by_method = {example["method"]: example for example in examples}
    if tests and all(name in by_method and by_method[name].get("history") for name in names):
        fig, axis = plt.subplots(figsize=(9, 4.8))
        for name, label in zip(names, labels):
            history = by_method[name]["history"]
            if isinstance(history, list):
                steps = [row["step"] for row in history[0]]
                values = np.asarray([[row["query_bce"] for row in replica] for replica in history]).mean(0)
            else:
                steps = torch.as_tensor(history["steps"]).cpu().numpy()
                values = torch.as_tensor(history["queryNMSE"]).detach().cpu().reshape(len(steps), -1).mean(1).numpy()
            axis.plot(steps, values, label=label)
        axis.set(xlabel="Шаг обучения дочерней сети", ylabel=f"Query {metric} (меньше — лучше)", title=tests[0])
        axis.legend()
        fig.tight_layout()
        relative = "figures/final_query_curves.png"
        fig.savefig(out / relative, dpi=160)
        plt.close(fig)
        figures.append(relative)
        lines.extend(["### Кривые качества", "", f"Первая test-задача: `{tests[0]}`. Среднее по инициализациям; query используется только для диагностики.", "",
                      f"![Кривые качества]({relative})", ""])

    fig, axes = plt.subplots(1, 4, figsize=(14, 6))
    for axis, name, label in zip(axes, names, labels):
        mask = torch.as_tensor(methods[name]).detach().cpu().numpy()
        axis.imshow(mask, cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        axis.set(title=f"{label}\nK={int(mask.sum())}", xlabel="Скрытый нейрон", ylabel="Входной признак")
    fig.tight_layout()
    relative = "figures/final_masks.png"
    fig.savefig(out / relative, dpi=160)
    plt.close(fig)
    figures.append(relative)
    lines.extend(["## Хитмапы масок", "", "Строки — входные признаки, столбцы — скрытые нейроны. Чёрный: активная связь (1), белый: отсутствующая (0).", "",
                  f"![Маски]({relative})", ""])
    report = out / "final_report.md"
    contents = "\n".join(lines)
    _atomic_write(report, lambda stream: stream.write(contents), binary=False)
    return {"markdown": "final_report.md", "figures": figures}
