"""Protocol-level tests for the multi-initialization agreement study."""

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run_agreement_replicates import (
    DEFAULT_PAIRS, parse_pair, pair_protocol, search_is_complete,
)


class AgreementReplicateTests(unittest.TestCase):
    def test_parse_pair(self):
        self.assertEqual(parse_pair("42:43"), (42, 43))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_pair("42:42")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_pair("42")

    def test_protocol_varies_only_model_seeds(self):
        left = pair_protocol((42, 43), 11, 7, 5, 3)
        right = pair_protocol((44, 45), 11, 7, 5, 3)
        self.assertEqual(left["model_seeds"], [42, 43])
        self.assertEqual(right["model_seeds"], [44, 45])
        left.pop("model_seeds")
        right.pop("model_seeds")
        self.assertEqual(left, right)
        self.assertEqual(left["replicate_design"]["independence_unit"], "VAE pair, not latent start")

    def test_default_has_32_disjoint_confirmatory_pairs(self):
        self.assertEqual(len(DEFAULT_PAIRS), 32)
        seeds = [seed for pair in DEFAULT_PAIRS for seed in pair]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual((min(seeds), max(seeds)), (186, 249))

    def test_tuned_search_defaults(self):
        protocol = pair_protocol((186, 187), 11, 64, 2000, 160)
        self.assertEqual(protocol["vae"]["epochs"], 160)
        self.assertEqual(protocol["search"]["steps"], 2000)
        self.assertEqual(protocol["search"]["lr"], 0.03)
        self.assertEqual(protocol["search"]["temperature"], 0.5)
        self.assertEqual(protocol["search"]["latent_radius"], 12.0)

    def test_completed_search_requires_matching_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, content in (("protocol.json", b"{}\n"), ("masks.pt", b"masks"),
                                  ("optimization.pt", b"optimization"),
                                  ("random_search.pt", b"random"), ("checkpoint.pt", b"model")):
                (root / name).write_bytes(content)
            digest = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            provenance = {
                "protocol_sha256": digest("protocol.json"),
                "mask_sha256": digest("masks.pt"),
                "models": [{"checkpoint": str(root / "checkpoint.pt"),
                            "sha256": digest("checkpoint.pt")}],
            }
            (root / "search_provenance.json").write_text(json.dumps(provenance))
            self.assertTrue(search_is_complete(root))
            (root / "masks.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                search_is_complete(root)


if __name__ == "__main__":
    unittest.main()
