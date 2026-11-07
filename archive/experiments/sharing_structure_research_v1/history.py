"""Preserve an indexed appendix of existing experiment reports and their tables."""

import argparse
import json
import subprocess
from pathlib import Path

from experiments.darts_structured_tasks_v1.data import digest, save_json

REPOSITORY = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
ROOTS = (
    "experiments/darts_indexator_v1",
    "experiments/darts_structured_tasks_v1",
    "experiments/mlp_structure_discovery_v1",
    "experiments/random_structure_search_v1",
    "experiments/sharing_benchmark_v1",
    "experiments/sharing_diagnostics_v1",
    "experiments/sharing_gradient_v1",
    "archive/2026-11-04",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "history")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    roots = [str(REPOSITORY / name) for name in ROOTS if (REPOSITORY / name).exists()]
    result = subprocess.run(
        ["rg", "--files", *roots, "-g", "*.md"],
        check=True,
        text=True,
        capture_output=True,
    )
    records = []
    appendix = [
        "# Исторические методы: указатель и исходные таблицы",
        "",
        "Это индекс сохранившихся артефактов, а не единый сопоставимый benchmark. Таблицы относятся к разным функциям, бюджетам и способам выбора. Полный контекст и ограничения остаются по ссылке на каждый исходный отчёт. Неполные или удалённые raw-запуски не воспроизводились и не восстанавливались.",
        "",
    ]
    for name in sorted(result.stdout.splitlines()):
        path = Path(name)
        if any(
            part
            in (
                "registered_code",
                "confirmation_code",
                "code_snapshot",
                "analysis_snapshot",
                "source_snapshot",
            )
            for part in path.parts
        ):
            continue
        if path.name not in (
            "REPORT.md",
            "FINAL_COMPARISON.md",
            "SHARING_ABLATIONS.md",
            "README.md",
        ):
            continue
        text = path.read_text()
        lines = text.splitlines()
        tables, block = [], []
        for line in lines:
            if line.startswith("|"):
                block.append(line)
            elif block:
                tables.append("\n".join(block))
                block = []
        if block:
            tables.append("\n".join(block))
        title = next(
            (line.removeprefix("# ") for line in lines if line.startswith("# ")),
            path.name,
        )
        relative = str(path.relative_to(REPOSITORY))
        records.append(
            {
                "path": relative,
                "title": title,
                "sha256": digest(path),
                "tables": len(tables),
                "bytes": path.stat().st_size,
            }
        )
        appendix += [f"## {title}", "", f"Источник: [{relative}]({path}).", ""]
        intro = []
        for line in lines[1:]:
            if line.startswith(("##", "|", "![")):
                break
            if line.strip():
                intro.append(line)
            if len(" ".join(intro)) > 1200:
                break
        appendix += intro
        for table in tables:
            appendix += ["", table]
        if not tables:
            appendix.append(
                "Таблицы в этом документе отсутствуют; метод/состояние описаны в исходном тексте."
            )
        appendix.append("")
    save_json(
        args.output / "manifest.json",
        {
            "scope": "Historical artifact inventory, not comparable pooled results",
            "reports": records,
        },
    )
    (args.output / "APPENDIX.md").write_text("\n".join(appendix) + "\n")
    print(
        json.dumps(
            {
                "reports": len(records),
                "tables": sum(r["tables"] for r in records),
                "appendix": str(args.output / "APPENDIX.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
