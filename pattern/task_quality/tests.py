"""Targeted CPU tests for the task-quality experiment protocol."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from meta_pattern.data import build_task_splits, partition_ids
from .core import (
    build_experiment_data,
    build_task_support_query,
    build_test_pool,
    common_probe,
)
from .generator import FEATURE_DIM, Generator, exact_topk_ste, generate, permute_hidden_columns
from .meta import (
    MetaConfig,
    _batched_logits,
    _init_child_batch,
    _inner_adapt,
    child_logits_batch,
    fit_child_batch,
    train_meta,
)


class TaskQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_exact_topk_forward_and_surrogate_gradient(self) -> None:
        logits = torch.randn(4, 11, 8, requires_grad=True)
        mask = exact_topk_ste(logits)
        self.assertTrue(set(torch.unique(mask.detach()).tolist()).issubset({0.0, 1.0}))
        self.assertTrue(torch.equal(mask.detach().sum((-2, -1)), torch.full((4,), 32.0)))
        weights = torch.arange(88, dtype=mask.dtype).reshape(1, 11, 8)
        (mask * weights).sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_generator_is_invariant_to_hidden_and_map_order(self) -> None:
        torch.manual_seed(101)
        model = Generator(feature_dim=FEATURE_DIM, mode="transformer_mask").eval()
        bank = torch.randn(9, 8, FEATURE_DIM)
        x = torch.randint(0, 2, (16, 11)).float().mul(2).sub(1)
        y = torch.tensor([0.0, 1.0] * 8)
        hidden_permutations = torch.stack([torch.randperm(8) for _ in range(bank.size(0))])
        map_permutation = torch.randperm(bank.size(0))
        relabeled_bank = permute_hidden_columns(bank, hidden_permutations).index_select(
            0, map_permutation
        )
        with torch.no_grad():
            mask, scores = generate(model, bank, x, y)
            relabeled_mask, relabeled_scores = generate(model, relabeled_bank, x, y)
        self.assertTrue(torch.equal(mask, relabeled_mask))
        self.assertTrue(torch.allclose(scores, relabeled_scores, atol=1e-6, rtol=1e-6))

    def test_hypergradient_reaches_bank_and_support_encoders(self) -> None:
        torch.manual_seed(77)
        model = Generator(feature_dim=FEATURE_DIM, mode="transformer_mask")
        bank = torch.randn(5, 8, FEATURE_DIM)
        support_x = torch.randint(0, 2, (2, 16, 11)).float().mul(2).sub(1)
        support_y = torch.tensor([[0.0, 1.0] * 8, [1.0, 0.0] * 8])
        query_x = torch.randint(0, 2, (2, 12, 11)).float().mul(2).sub(1)
        query_y = torch.tensor([[0.0, 1.0] * 6, [1.0, 0.0] * 6])
        mask, _ = generate(model, bank, support_x, support_y)
        rng = torch.Generator(device="cpu").manual_seed(12)
        params = _init_child_batch(2, 2, rng, torch.device("cpu"))
        params, _ = _inner_adapt(
            support_x, support_y, mask, params, steps=2,
            learning_rate=0.1, momentum=0.9, create_graph=True,
        )
        logits = _batched_logits(query_x, mask, params)
        loss = F.binary_cross_entropy_with_logits(
            logits, query_y[:, None, :].expand_as(logits)
        )
        loss.backward()
        names = dict(model.named_parameters())
        for name in (
            "token_mlp.0.weight",
            "map_encoder.encoder.layers.0.self_attn.in_proj_weight",
            "bank_set_attention.in_proj_weight",
            "support_mlp.0.weight",
            "support_encoder.encoder.layers.0.self_attn.in_proj_weight",
            "context_mlp.0.weight",
            "edge_decoder.2.weight",
        ):
            gradient = names[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_task_orbits_and_global_input_partitions_are_disjoint(self) -> None:
        splits = build_task_splits([4], seed=42)
        self.assertEqual((len(splits["train"]), len(splits["val"]), len(splits["test"])),
                         (10, 2, 4))

        def orbit(pattern: str) -> frozenset[str]:
            reverse = pattern[::-1]
            complement = "".join("1" if bit == "0" else "0" for bit in pattern)
            reverse_complement = complement[::-1]
            return frozenset((pattern, reverse, complement, reverse_complement))

        split_orbits = [
            {orbit(task.pattern) for task in splits[name]}
            for name in ("train", "val", "test")
        ]
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertTrue(split_orbits[left].isdisjoint(split_orbits[right]))

        ids = torch.arange(1 << 11)
        partition = partition_ids(ids, split_seed=1729)
        self.assertEqual(tuple(int((partition == code).sum()) for code in range(3)),
                         (1226, 408, 414))
        probe = common_probe(n_probe=128, seed=8100, split_seed=1729)
        self.assertTrue((partition[probe["ids"]] == 0).all())
        pools = build_task_support_query(splits["test"][0], probe["ids"], split_seed=1729)
        self.assertEqual(set(pools), {"support", "query"})
        test_pool = build_test_pool(splits["test"][0], split_seed=1729)
        support_ids = set(pools["support"]["ids"].tolist())
        query_ids = set(pools["query"]["ids"].tolist())
        test_ids = set(test_pool["ids"].tolist())
        self.assertEqual(len(support_ids), 1226 - 128)
        self.assertEqual(len(query_ids), 408)
        self.assertEqual(len(test_ids), 414)
        self.assertFalse(support_ids & query_ids)
        self.assertFalse(support_ids & test_ids)
        self.assertFalse(query_ids & test_ids)
        self.assertFalse(support_ids & set(probe["ids"].tolist()))

    def test_child_adam_checkpoint_resume_is_exact(self) -> None:
        torch.manual_seed(303)
        x = torch.randint(0, 2, (40, 11)).float().mul(2).sub(1)
        y = torch.tensor([0.0, 1.0] * 20)
        masks = torch.zeros(2, 11, 8)
        masks.reshape(2, -1)[:, :32] = 1.0
        options = dict(
            x_query=x, y_query=y, seeds=[7, 8],
            learning_rates=torch.tensor([0.001, 0.003]),
            max_steps=4, min_steps=0, batch_size=8, eval_every=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            full = fit_child_batch(masks, x, y, **options)
            checkpoint = Path(temp_dir) / "child.pt"
            first = fit_child_batch(masks, x, y, checkpoint_path=checkpoint,
                                    max_updates=2, **options)
            self.assertEqual(first["total_steps"], 2)
            resumed = fit_child_batch(
                masks, x, y, checkpoint_path=checkpoint, resume=True,
                max_updates=2, **options,
            )
        self.assertEqual(resumed["total_steps"], 4)
        for key in ("w", "b", "a", "c"):
            self.assertTrue(torch.equal(full["best_params"][key], resumed["best_params"][key]), key)
            self.assertTrue(torch.equal(full["last_params"][key], resumed["last_params"][key]), key)
        self.assertTrue(torch.equal(full["best_steps"], resumed["best_steps"]))
        self.assertEqual(len(full["history"]), len(resumed["history"]))
        self.assertEqual(child_logits_batch(x, masks, resumed["best_params"]).shape, (2, 40))
        self.assertTrue(all(row["selection_complete"] for row in resumed["fit_status"]))

    def test_meta_chunk_resume_restores_optimizer_rng_and_curves(self) -> None:
        torch.set_num_threads(1)
        config = MetaConfig(
            method="transformer_mask", meta_batch_tasks=2, replicas=1,
            validation_replicas=1, support_size=8, query_size=8, inner_steps=1,
            min_steps=1, max_steps=2, resume_max_steps=3, eval_every=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bank_path = root / "bank.pt"
            torch.save({"protocol": "cpu-smoke", "feature": torch.randn(3, 8, FEATURE_DIM)},
                       bank_path)
            full_dir = root / "full"
            split_dir = root / "split"
            train_meta(bank_path, full_dir, 8100, "transformer_mask", "cpu",
                       config=config, resume=False)
            train_meta(bank_path, split_dir, 8100, "transformer_mask", "cpu",
                       config=config, resume=False, max_updates=1)
            checkpoint = torch.load(split_dir / "meta" / "checkpoint.pt",
                                    map_location="cpu", weights_only=False)
            self.assertIsInstance(checkpoint["curves"]["step"], torch.Tensor)
            train_meta(bank_path, split_dir, 8100, "transformer_mask", "cpu",
                       config=config, resume=True, max_updates=1)
            full = torch.load(full_dir / "meta" / "last.pt", map_location="cpu",
                              weights_only=False)
            resumed = torch.load(split_dir / "meta" / "last.pt", map_location="cpu",
                                 weights_only=False)
        self.assertEqual(full["step"], resumed["step"])
        self.assertEqual(full["step"], 2)
        self.assertEqual(full["bank_sha256"], resumed["bank_sha256"])
        for key, value in full["model_state"].items():
            self.assertTrue(torch.equal(value, resumed["model_state"][key]), key)
        for key in ("step", "train_monitor_query_bce", "val_query_bce"):
            self.assertTrue(torch.equal(full["curves"][key], resumed["curves"][key]), key)


if __name__ == "__main__":
    unittest.main()
