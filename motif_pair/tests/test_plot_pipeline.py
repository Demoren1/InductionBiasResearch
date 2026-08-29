from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.baselines import gold_mask
from evaluation.plot_pipeline import _circular_diagonal_profile, _circular_toeplitz_score, build_plots


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
