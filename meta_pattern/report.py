"""Summarize a completed suite, retaining individual lengths and seeds."""
import argparse
import json
from pathlib import Path

from .common import write_json


def report(suite):
    suite = Path(suite)
    manifest = json.loads((suite / "suite.json").read_text())
    rows = []
    for method, seed in manifest["jobs"]:
        path = suite / f"{method}_seed{seed}" / "evaluation.json"
        data = json.loads(path.read_text())
        for row in data["aggregates"]:
            rows.append({"method": method, "seed": seed, "checkpoint_step": data["checkpoint_step"], **row})
    write_json(suite / "summary.json", {"preset": manifest["preset"], "rows": rows})
    budgets = sorted({r["steps"] for r in rows})
    lines = ["# Full-U pattern experiment", "", f"Preset: `{manifest['preset']}`.", "",
             "Each cell averages held-out patterns and repeats within each length, then weights lengths equally.",
             "Seeds are reported separately; this small suite does not provide a confidence interval.", "",
             "| Method | Seed | Regime | Adaptation steps | Balanced accuracy | Natural accuracy |", "|---|---:|---|---:|---:|---:|"]
    for method, seed in manifest["jobs"]:
        regimes = sorted({r["regime"] for r in rows if r["method"] == method and r["seed"] == seed})
        for regime in regimes:
            for budget in budgets:
                values = []
                for kind in ("balanced", "natural"):
                    group = [r["accuracy"] for r in rows if r["method"] == method and r["seed"] == seed
                             and r["steps"] == budget and r["distribution"] == kind and r["regime"] == regime]
                    values.append(f"{100 * sum(group) / len(group):.2f}%" if group else "n/a")
                lines.append(f"| {method} | {seed} | {regime} | {budget} | {values[0]} | {values[1]} |")
    lines += ["", "`ideal` means analytical U with a freshly trained v, not analytical solution weights.",
              "The analytical detector is tested independently for exact correctness.",
              "Natural accuracy can be dominated by prevalence, especially for short patterns.",
              "Table has no prediction for unseen lengths; inspect per-length JSON before comparing methods.", ""]
    (suite / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    report(parser.parse_args().suite)
