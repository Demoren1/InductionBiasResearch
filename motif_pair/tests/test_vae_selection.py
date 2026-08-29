from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.cvae import (load_top_importance, posterior_diagnostics,
                         read_split_provenance, verify_generator_config)
from models.sweep_beta import select_beta_run
from models.train_cvae import collapse_guard, tail_stability


def _metrics(*, kl: float, active: int, gap: float) -> dict:
    return {
        "total": 10.0,
        "recon": 9.0,
        "kl": kl,
        "posterior_mu_std": .2,
        "posterior_std_mean": .9,
        "posterior_vs_z0_recon_gap": gap,
        "active_mu_variance_dims": active,
        "active_mu_variance_threshold": 1e-2,
    }


def _guard(metrics: dict) -> dict:
    return collapse_guard(metrics, min_active_dims=2, min_val_kl=1.0,
                          active_mu_variance_threshold=1e-2,
                          min_posterior_z0_recon_gap=0.0)


def test_posterior_diagnostics_reports_mu_variance_activity():
    mu = torch.zeros(8, 3)
    mu[:, 0] = torch.arange(8, dtype=torch.float32)
    mu[:, 1] = torch.arange(8, dtype=torch.float32) * .2
    logvar = torch.zeros_like(mu)
    stats = posterior_diagnostics(mu, logvar, active_mu_variance_threshold=.01)
    assert stats["active_mu_variance_dims"] == 2
    assert stats["active_latent_dims"] == 2
    assert len(stats["posterior_mu_variance_per_dim"]) == 3


def test_collapse_guard_requires_kl_mu_activity_and_positive_z0_gap():
    assert _guard(_metrics(kl=1.2, active=2, gap=.1))["passed"]
    failure = _guard(_metrics(kl=.9, active=1, gap=0.0))
    assert not failure["passed"]
    assert "kl<1" in failure["failures"]
    assert "active_mu_variance_dims<2" in failure["failures"]
    assert "posterior_vs_z0_recon_gap<=0" in failure["failures"]


def test_beta_selector_prefers_largest_noncollapsed_beta_then_val_recon():
    def row(beta, seed, recon, passed=True):
        metrics = _metrics(kl=1.1, active=2, gap=.2)
        guard = _guard(metrics)
        if not passed:
            guard = {**guard, "passed": False, "failures": ["forced"]}
        return {"status": "completed", "variant": "cvae", "beta": beta, "seed": seed,
                "run_dir": f"/tmp/{beta}_{seed}", "checkpoint": f"/tmp/{beta}_{seed}/best.pt",
                "summary": {"selection_status": "eligible" if passed else "tail_unstable_not_promoted",
                            "selected_val": {**metrics, "recon": recon}, "collapse_guard": guard,
                            "tail_stability": {"stable": passed}}}

    selected = select_beta_run([row(1.0, 42, 8.0, passed=False), row(.1, 2, 7.5),
                                row(.1, 1, 7.0), row(.03, 42, 6.0)])
    assert selected["selection_status"] == "selected"
    assert selected["selected"]["beta"] == .1
    assert selected["selected"]["seed"] == 1


def test_beta_selector_requires_every_declared_seed_to_be_stable():
    def row(beta, seed, stable=True, recon=7.0):
        metrics = _metrics(kl=1.1, active=2, gap=.2)
        return {"status": "completed" if stable else "failed", "variant": "cvae",
                "beta": beta, "seed": seed, "run_dir": f"/tmp/{beta}_{seed}",
                "checkpoint": f"/tmp/{beta}_{seed}/best.pt",
                "summary": {"selection_status": "eligible",
                            "selected_val": {**metrics, "recon": recon},
                            "collapse_guard": _guard(metrics),
                            "tail_stability": {"stable": stable}}}

    selected = select_beta_run([
        row(.3, 1), row(.3, 2, stable=False),  # strongest beta is not reproducible
        row(.1, 1, recon=7.2), row(.1, 2, recon=7.0),
    ], expected_seeds=[1, 2])
    assert selected["selection_status"] == "selected"
    assert selected["selected"]["beta"] == .1
    assert selected["selected"]["seed"] == 2


def test_transient_epoch_one_guard_pass_does_not_make_beta_eligible():
    passed = _guard(_metrics(kl=2.0, active=2, gap=.2))
    failed = _guard(_metrics(kl=.2, active=0, gap=0.0))
    history = [
        {"epoch": 1, "val": _metrics(kl=2.0, active=2, gap=.2), "collapse_guard": passed},
        {"epoch": 2, "val": _metrics(kl=.2, active=0, gap=0.0), "collapse_guard": failed},
        {"epoch": 3, "val": _metrics(kl=.2, active=0, gap=0.0), "collapse_guard": failed},
        {"epoch": 4, "val": _metrics(kl=.2, active=0, gap=0.0), "collapse_guard": failed},
        {"epoch": 5, "val": _metrics(kl=.2, active=0, gap=0.0), "collapse_guard": failed},
    ]
    tail = tail_stability(history, tail_fraction=.1, tail_min_epochs=5)
    assert not tail["stable"]
    transient = {
        "status": "completed", "variant": "cvae", "beta": .3, "seed": 42,
        "run_dir": "/tmp/transient", "checkpoint": "/tmp/transient/best.pt",
        # This emulates the old bad selection: its early checkpoint passed,
        # but its final tail did not.  The new selector must reject it.
        "summary": {"selection_status": "tail_unstable_not_promoted",
                    "selected_val": history[0]["val"], "collapse_guard": passed,
                    "tail_stability": tail},
    }
    stable = {
        "status": "completed", "variant": "cvae", "beta": .1, "seed": 42,
        "run_dir": "/tmp/stable", "checkpoint": "/tmp/stable/best.pt",
        "summary": {"selection_status": "eligible", "selected_val": _metrics(kl=1.1, active=2, gap=.2),
                    "collapse_guard": passed,
                    "tail_stability": {"stable": True}},
    }
    selected = select_beta_run([transient, stable])
    assert selected["selected"]["beta"] == .1


def test_load_top_importance_default_keeps_continuous_values(tmp_path):
    task = "A000_B001_G03"
    directory = tmp_path / f"task_{task}"
    directory.mkdir()
    maps = torch.stack([torch.full((16, 16), .05 * (i + 1)) for i in range(10)])
    torch.save({"importance": maps, "val_loss": torch.arange(10, dtype=torch.float32)},
               directory / "importance.pt")
    x, c, provenance = load_top_importance([task], ckpt_root=tmp_path, top_frac=.1)
    assert x.shape == (1, 256)
    assert c.shape == (1, 8)
    assert torch.allclose(x, torch.full_like(x, .05))
    assert provenance[0]["source"].endswith("importance.pt")


def test_load_top_importance_rejects_stale_split_even_with_matching_task_ids(tmp_path):
    task = "A000_B001_G03"
    split = tmp_path / "split.json"
    split.write_text('{"train_tasks": ["A000_B001_G03"], "test_tasks": ["A001_B010_G04"]}\n')
    expected = read_split_provenance(split)
    directory = tmp_path / f"task_{task}"
    directory.mkdir()
    torch.save({"task": task, "importance": torch.ones(2, 16, 16),
                "val_loss": torch.tensor([0., 1.]),
                "split_sha256": "stale", "split_train_tasks": [task]},
               directory / "importance.pt")
    try:
        load_top_importance([task], ckpt_root=tmp_path, split_path=split)
    except ValueError as error:
        assert "SHA256 mismatch" in str(error)
    else:
        raise AssertionError("stale SHA256 must be rejected")


def test_load_top_importance_accepts_exact_current_split(tmp_path):
    task = "A000_B001_G03"
    split = tmp_path / "split.json"
    split.write_text('{"train_tasks": ["A000_B001_G03"], "test_tasks": ["A001_B010_G04"]}\n')
    expected = read_split_provenance(split)
    directory = tmp_path / f"task_{task}"
    directory.mkdir()
    torch.save({"task": task, "importance": torch.ones(2, 16, 16),
                "val_loss": torch.tensor([0., 1.]), **expected}, directory / "importance.pt")
    x, _, provenance = load_top_importance([task], ckpt_root=tmp_path, split_path=split)
    assert x.shape == (1, 256)
    assert provenance[0]["split_sha256"] == expected["split_sha256"]


def test_generator_config_rejects_different_importance_preprocessing():
    payload = {"importance_name": "importance.pt", "top_frac": .1}
    try:
        verify_generator_config(payload, importance_name="best10pct.pt", top_frac=.1,
                                label="CVAE checkpoint")
    except ValueError as error:
        assert "importance_name mismatch" in str(error)
    else:
        raise AssertionError("checkpoint preprocessing mismatch must be rejected")
    try:
        verify_generator_config(payload, importance_name="importance.pt", top_frac=.2,
                                label="CVAE checkpoint")
    except ValueError as error:
        assert "top_frac mismatch" in str(error)
    else:
        raise AssertionError("checkpoint top fraction mismatch must be rejected")
