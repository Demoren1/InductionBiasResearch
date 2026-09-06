"""Tests for the clean variable-length pattern data sampler."""

from __future__ import annotations

import unittest

import torch

from meta_pattern.data import PatternTask, build_task_splits, contains_pattern, partition_ids, sample_dataset


def _reference_contains(bits: list[int], pattern: list[int]) -> bool:
    return any(bits[start:start + len(pattern)] == pattern for start in range(len(bits) - len(pattern) + 1))


class PatternDataTests(unittest.TestCase):
    def test_contains_pattern_handles_multiple_occurrences_and_near_matches(self) -> None:
        x = torch.tensor([
            [0, 1, 0, 1, 0, 1, 0],  # 010 occurs three times
            [0, 1, 1, 0, 1, 1, 0],  # all near-matches to 010
            [1, 0, 1, 0, 1, 0, 1],  # 010 occurs twice
        ])
        pattern = torch.tensor([0, 1, 0])
        self.assertTrue(torch.equal(contains_pattern(x, pattern), torch.tensor([True, False, True])))

    def test_contains_pattern_agrees_with_scalar_reference_for_lengths_3_to_8(self) -> None:
        sequences = torch.tensor([
            [int(bit) for bit in f"{value:032b}"]
            for value in (0, 1, 0x12345678, 0xAAAAAAAA, 0xFFFFFFFF)
        ])
        for length in range(3, 9):
            pattern = [int(bit) for bit in format((1 << (length - 1)) + 1, f"0{length}b")]
            expected = torch.tensor([_reference_contains(row.tolist(), pattern) for row in sequences])
            self.assertTrue(torch.equal(contains_pattern(sequences, torch.tensor(pattern)), expected))

    def test_task_splits_are_reproducible_and_orbit_disjoint(self) -> None:
        first = build_task_splits(seed=91)
        second = build_task_splits(seed=91)
        self.assertEqual(first, second)
        all_tasks = [task for tasks in first.values() for task in tasks]
        self.assertEqual(len(all_tasks), sum(1 << length for length in range(3, 9)))
        self.assertEqual(len({task.task_id for task in all_tasks}), len(all_tasks))
        memberships = {task.pattern: name for name, tasks in first.items() for task in tasks}
        for task in all_tasks:
            orbit = {task.pattern, task.pattern[::-1]}
            complement = "".join("1" if bit == "0" else "0" for bit in task.pattern)
            orbit.update({complement, complement[::-1]})
            self.assertEqual({memberships[member] for member in orbit}, {memberships[task.pattern]})
        # Length 3 has only three reversal/complement orbits, but each split
        # still receives one of them.
        for split in first.values():
            self.assertTrue(any(task.length == 3 for task in split))

    def test_samples_are_reproducible_balanced_and_labelled(self) -> None:
        task = PatternTask("010")
        first = sample_dataset(task, n_samples=257, seed=12, split="support")
        second = sample_dataset(task, n_samples=257, seed=12, split="support")
        for key in ("x", "y", "ids"):
            self.assertTrue(torch.equal(first[key], second[key]))
        self.assertEqual(int(first["y"].sum().item()), 128)
        self.assertEqual(first["ids"].unique().numel(), 257)
        x01 = ((first["x"] + 1.0) / 2.0).to(torch.int64)
        expected = contains_pattern(x01, torch.tensor([0, 1, 0])).to(torch.float32)
        self.assertTrue(torch.equal(first["y"], expected))

    def test_sampler_labels_match_independent_reference_for_every_length(self) -> None:
        for length in range(3, 9):
            pattern_string = "0" * (length - 1) + "1"
            pattern = [int(bit) for bit in pattern_string]
            data = sample_dataset(PatternTask(pattern_string), n_samples=65, seed=length, split="test")
            expected = []
            for identifier in data["ids"].tolist():
                bits = [int(bit) for bit in f"{identifier:032b}"]
                expected.append(float(_reference_contains(bits, pattern)))
            self.assertTrue(torch.equal(data["y"], torch.tensor(expected)))

    def test_global_partitions_are_disjoint_even_across_tasks_and_seeds(self) -> None:
        task_a = PatternTask("010")
        task_b = PatternTask("111001")
        support = sample_dataset(task_a, n_samples=301, seed=1, split="support")
        query = sample_dataset(task_b, n_samples=301, seed=2, split="query")
        test = sample_dataset(task_a, n_samples=301, seed=3, split="test")
        sets = [set(data["ids"].tolist()) for data in (support, query, test)]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        for data, expected_code in zip((support, query, test), (0, 1, 2)):
            codes = partition_ids(data["ids"])
            self.assertTrue(torch.equal(codes, torch.full_like(codes, expected_code)))

    def test_natural_sampler_keeps_natural_rate_without_injection(self) -> None:
        # A rare length-8 pattern should remain far from 50% when balancing is
        # disabled; any insertion-based generator would violate this check.
        data = sample_dataset(PatternTask("01011010"), n_samples=2_000, seed=4, split="query", balanced=False)
        fraction = float(data["y"].mean())
        self.assertGreater(fraction, 0.01)
        self.assertLess(fraction, 0.25)


if __name__ == "__main__":
    unittest.main()
