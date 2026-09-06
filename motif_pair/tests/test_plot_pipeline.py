from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.baselines import gold_mask, interpolate_train_gap_mean
from evaluation.eval_generated_masks import _task_at_gap, _wrong_gap, _wrong_gaps
from evaluation.plot_pipeline import _circular_diagonal_profile, _circular_toeplitz_score, build_plots


def test_gap_ood_runner_rejects_one_hot_before_numerical_stages():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "10_gap_ood.sh")],
        cwd=ROOT,
        env={**os.environ, "CONDITION_ENCODING": "one_hot"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires CONDITION_ENCODING=scalar" in result.stderr


def test_plot_pipeline_writes_available_stages_and_skips_later_ones(tmp_path: Path):
    run = tmp_path / "run"
    ckpt = run / "checkpoints" / "task_A000_B001_G03"
    ckpt.mkdir(parents=True)
    split = {
        "split_seed": 1,
        "pair_disjoint": True,
        "train_tasks": ["A000_B001_G03"],
        "test_tasks": ["A001_B010_G04"],
    }
    (run / "split.json").write_text(json.dumps(split))
    (ckpt / "best10pct_summary.txt").write_text(
        "task=A000_B001_G03\nval_loss (BCE) min=0.1 p50=0.2 max=0.3\n"
    )
    data = tmp_path / "data"
    data.mkdir()
    torch.save({
        "x": torch.tensor([[-1., 1.] * 8, [1., -1.] * 8,
                           [-1., -1., 1., 1.] * 4, [1., 1., -1., -1.] * 4]),
        "y": torch.tensor([0., 1., 0., 1.]),
        "ones_count": torch.tensor([7, 7, 8, 8]),
        "a_start": torch.tensor([0, 1, 2, 3]),
        "b_start": torch.tensor([4, 4, 7, 6]),
        "delta": torch.tensor([4, 3, 5, 3]),
    }, data / "val_A000_B001_G03.pt")
    torch.save({"y": torch.tensor([0., 1., 0., 1.]),
                "ones_count": torch.tensor([7, 7, 8, 8])},
               data / "val_A001_B010_G04.pt")

    manifest = build_plots(run, data_dir=data, include_generator=False, device="cpu")
    plots = run / "plots"
    persisted = json.loads((plots / "manifest.json").read_text())
    assert manifest == persisted
    assert persisted["stages"]["split"]["status"] == "generated"
    assert persisted["stages"]["data"]["status"] == "generated"
    assert persisted["stages"]["candidate_selection"]["status"] == "generated"
    assert persisted["stages"]["continuous_importance"]["status"] == "missing"
    assert persisted["stages"]["final_evaluation"]["status"] == "missing"
    assert (plots / "01_split.png").is_file()
    assert (plots / "02_data.png").is_file()
    assert (plots / "02_data_examples.png").is_file()
    assert (plots / "03_candidates.png").is_file()


def test_circular_toeplitz_diagnostic_is_exact_for_gold_support():
    ideal = gold_mask("A000_B001_G07").reshape(16, 16)
    profile = _circular_diagonal_profile(ideal)

    assert _circular_toeplitz_score(ideal).item() == pytest.approx(1.0)
    assert int((profile == 1).sum()) == 6
    assert int((profile == 0).sum()) == 10


def test_train_only_gap_mean_interpolates_and_clamps_without_target_maps():
    lower = torch.zeros(4)
    upper = torch.full((4,), 10.0)
    interpolated, provenance = interpolate_train_gap_mean({4: lower, 6: upper}, 5)
    assert torch.equal(interpolated, torch.full((4,), 5.0))
    assert provenance == {"kind": "linear_interpolation", "source_gaps": [4, 6], "weight_upper": .5}

    clamped, provenance = interpolate_train_gap_mean({4: lower, 6: upper}, 3)
    assert torch.equal(clamped, lower)
    assert provenance["kind"] == "boundary_clamp_low"
    assert provenance["extrapolation_policy"] == "nearest_train_gap_clamp"


def test_wrong_gap_changes_task_semantics_not_condition_tensor_layout():
    task = "A000_B001_G03"
    # Extrapolation: both g=3 and g=4 must use the nearest *seen* regime g=5,
    # never another held-out condition.
    assert _wrong_gap(task, [5, 6, 7, 8, 9, 10]) == 5
    assert _wrong_gap("A000_B001_G04", [5, 6, 7, 8, 9, 10]) == 5
    # Production interpolation retains both equally-near seen conditions.
    assert _wrong_gaps("A000_B001_G05", [3, 4, 6, 7, 9, 10]) == (4, 6)
    assert _wrong_gaps("A000_B001_G08", [3, 4, 6, 7, 9, 10]) == (7, 9)
    # g=10 is permutation-equivalent to g=6 and is not a structural negative.
    assert _wrong_gaps("A000_B001_G06", [5, 10]) == (5,)
    # The compatibility helper remains deterministic for direct callers.
    assert _wrong_gap("A000_B001_G05", [3, 4, 6, 7, 9, 10]) == 4
    assert _task_at_gap(task, _wrong_gap(task, [5, 6, 7, 8, 9, 10])) == "A000_B001_G05"
    # Direct legacy callers retain the adjacent semantic-gap fallback.
    assert _wrong_gap("A000_B001_G10") == 9


def test_gap_ood_generalization_panel_always_exports_png_and_pdf(tmp_path: Path):
    run = tmp_path / "gap_ood"
    run.mkdir()
    split = {
        "split_kind": "gap_heldout",
        "heldout_gaps": [5, 8],
        "train_gaps": [3, 4, 6, 7, 9, 10],
        "train_tasks": ["A000_B001_G03"],
        "test_tasks": ["A000_B001_G05", "A000_B001_G08"],
    }
    (run / "split.json").write_text(json.dumps(split))
    eval_dir = run / "eval"
    eval_dir.mkdir()
    methods = ("cvae", "conditional_mean", "vae", "random_exact96", "cvae_wrong_gap")
    task_rows = {}
    for task, gap in (("A000_B001_G05", 5), ("A000_B001_G08", 8)):
        task_rows[task] = {
            method: {"mean_acc": .6 + .01 * index + gap * .001,
                     "mean_best_permutation_iou": .3 + .01 * index}
            for index, method in enumerate(methods)
        }
    (eval_dir / "eval_results.json").write_text(json.dumps({
        "provenance": {"condition_encoding": "scalar"},
        "tasks": task_rows,
    }))
    (eval_dir / "summary.json").write_text(json.dumps({
        "methods": {method: {"mean_acc": .6, "mean_best_permutation_iou": .3} for method in methods}
    }))

    manifest = build_plots(run, include_generator=False, device="cpu", pdf=False)
    assert manifest["stages"]["gap_ood_generalization"]["status"] == "generated"
    assert (run / "plots" / "11_gap_ood_generalization.png").is_file()
    assert (run / "plots" / "11_gap_ood_generalization.pdf").is_file()


def test_legacy_one_hot_final_evaluation_still_plots(tmp_path: Path):
    run = tmp_path / "legacy_one_hot"
    eval_dir = run / "eval"
    eval_dir.mkdir(parents=True)
    (run / "split.json").write_text(json.dumps({
        "train_tasks": ["A000_B001_G03"],
        "test_tasks": ["A001_B010_G04"],
    }))
    methods = {
        "cvae": {"mean_acc": .7, "mean_best_permutation_iou": .5},
        "random_exact96": {"mean_acc": .6, "mean_best_permutation_iou": .3},
    }
    (eval_dir / "summary.json").write_text(json.dumps({"methods": methods}))
    (eval_dir / "eval_results.json").write_text(json.dumps({
        "provenance": {"condition_encoding": "one_hot"},
        "tasks": {"A001_B010_G04": methods},
    }))

    manifest = build_plots(run, include_generator=False, device="cpu")
    assert manifest["stages"]["final_evaluation"]["status"] == "generated"
    assert "gap_ood_generalization" not in manifest["stages"]


def test_gap_ood_publication_panel_rejects_one_hot_provenance(tmp_path: Path):
    run = tmp_path / "invalid_gap_one_hot"
    eval_dir = run / "eval"
    eval_dir.mkdir(parents=True)
    task = "A000_B001_G05"
    split = {
        "split_kind": "gap_heldout",
        "heldout_gaps": [5],
        "train_tasks": ["A000_B001_G03"],
        "test_tasks": [task],
    }
    (run / "split.json").write_text(json.dumps(split))
    row = {"cvae": {"mean_acc": .7, "mean_best_permutation_iou": .5}}
    (eval_dir / "summary.json").write_text(json.dumps({"methods": row}))
    (eval_dir / "eval_results.json").write_text(json.dumps({
        "provenance": {"condition_encoding": "one_hot"},
        "tasks": {task: row},
    }))

    manifest = build_plots(run, include_generator=False, device="cpu")
    stage = manifest["stages"]["gap_ood_generalization"]
    assert stage["status"] == "failed"
    assert "requires scalar-conditioned" in stage["reason"]
    assert not (run / "plots" / "11_gap_ood_generalization.png").exists()
