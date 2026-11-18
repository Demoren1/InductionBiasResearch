"""Score the retained synthetic case and render the matched result tables."""

import argparse

import numpy as np

from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .source import BUDGETS, OUTPUT, ROOT, SYNTHETIC, read_json
from .timings import EXTENSION
from .transfer import METHODS
from .wsr import TARGET


NAMES = {
    "dense": "Dense",
    "g34": "G34",
    "asrr": "aSRR",
    "lutq": "LUT-Q",
    "yeh_map": "Yeh (MAP)",
    "yeh_soft": "Yeh (soft)",
    "msr": "MSR-FC",
    "wsr": "WSR",
}


def score_synthetic():
    paths = {
        "asrr": SYNTHETIC / TARGET / "asrr/frozen_predictions.npz",
        "wsr": ROOT / "synthetic" / TARGET / "wsr/frozen_predictions.npz",
        "legacy": EXTENSION / "synthetic_scoring.npz",
    }
    for method in ("asrr", "wsr"):
        receipt = read_json(paths[method].parent / "registered.json")
        for name in ("support.npz", "selection.npz", "query_x.npz"):
            if receipt["input_sha256"][name] != digest(
                SYNTHETIC / TARGET / name
            ):
                raise ValueError(
                    f"Synthetic {method} uses different target examples"
                )
    save_json(
        OUTPUT / "synthetic/predictions_frozen.json",
        {m: digest(p) for m, p in paths.items()},
    )
    with np.load(SYNTHETIC / "private/answers.npz") as data:
        truth = data["y"]
    with np.load(paths["legacy"]) as data:
        if not np.array_equal(truth, data["case4_truth"]):
            raise ValueError(
                "Synthetic scoring archive is from different functions"
            )
        predictions = {
            method: data[f"case4_{method}"]
            for method in METHODS
            if method not in ("asrr", "wsr")
        }
    for method in ("asrr", "wsr"):
        with np.load(paths[method]) as data:
            predictions[method] = data["prediction_256"]
    audit = read_json(EXTENSION / "synthetic_structure_audit.json")["4"]
    rows = {}
    for method in METHODS:
        prediction = predictions[method]
        if (
            prediction.shape != truth.shape
            or not np.isfinite(prediction).all()
        ):
            raise ValueError(f"Synthetic {method} predictions are invalid")
        mse = ((prediction - truth) ** 2).mean((1, 2))
        exact = (
            audit[method]["signed_exact"]
            if method in audit
            else False
            if method in ("asrr", "wsr")
            else None
        )
        rows[method] = {
            "mse": float(mse.mean()),
            "per_function_mse": mse.tolist(),
            "signed_exact": exact,
            "codes": 256 if method == "dense" else BUDGETS["synthetic"],
        }
    save_json(
        OUTPUT / "synthetic/results.json",
        {"methods": rows, "geometry": 4, "target_functions": len(truth)},
    )
    print(rows)


def render_tables():
    timings = read_json(OUTPUT / "source_times.json")
    for setting, label, caption in (
        (
            "synthetic",
            "recovery-synthetic",
            "Synthetic transfer on four new numerical functions with the same generating structure. Exact recovery compares signed groups and support with the teacher. Dashes denote quantities outside the timing or recovery comparison.",
        ),
        (
            "omniglot10",
            "tab:omniglot-wsr",
            "Five-way Omniglot on 1024 common episodes. Accuracy gives 95\\% intervals across episodes conditional on the held-out characters.",
        ),
        (
            "har3",
            "tab:recovery-har",
            "UCI HAR on 180 common episodes. Accuracy gives 95\\% intervals over the nine subject means.",
        ),
    ):
        result = read_json(OUTPUT / setting / "results.json")["methods"]
        synthetic = setting == "synthetic"
        metrics = (
            r"Query MSE & Exact recovery"
            if synthetic
            else r"Accuracy (\%) & CE"
        )
        lines = [
            r"\begin{table}[htbp]",
            r"\centering\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label if label.startswith('tab:') else 'tab:' + label}}}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            rf"Method & Budget & {metrics} & Source time (s) \\",
            r"\midrule",
        ]
        for method in METHODS:
            row = result[method]
            if synthetic:
                mse = row["mse"]
                coefficient, exponent = f"{mse:.2e}".split("e")
                value = (
                    rf"${coefficient}\times10^{{{int(exponent)}}}$"
                    if mse < 0.001
                    else f"{mse:.3g}"
                )
                exact = row["signed_exact"]
                second = (
                    r"\textemdash"
                    if exact is None
                    else ("Yes" if exact else "No")
                )
            else:
                value = rf"${100 * row['accuracy']:.2f}\pm{100 * row['accuracy_ci95_halfwidth']:.2f}$"
                second = f"{row['ce']:.4f}"
            elapsed = timings[setting].get(method, {}).get("seconds")
            seconds = (
                r"\textemdash"
                if elapsed is None
                else f"{elapsed:.2f}"
                if elapsed < 100
                else f"{elapsed:,.1f}"
            )
            lines.append(
                rf"{NAMES[method]} & {row['codes']:,} & {value} & {second} & {seconds} \\"
            )
        lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
        OUTPUT.joinpath(f"{setting}_table.tex").write_text(
            "\n".join(lines) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("synthetic", "tables"))
    {"synthetic": score_synthetic, "tables": render_tables}[
        parser.parse_args().action
    ]()
