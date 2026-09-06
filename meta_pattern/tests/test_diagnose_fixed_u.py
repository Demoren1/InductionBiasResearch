import unittest

from meta_pattern.diagnose_fixed_u import choose_by_support


def _row(config_index, restart, support_bce, query_bce):
    return {
        "config_index": config_index,
        "restart": restart,
        "support": {"bce": support_bce, "accuracy": 0.5, "nonfinite": False},
        "query": {"bce": query_bce, "accuracy": 0.5, "nonfinite": False},
    }


class SupportSelectionTest(unittest.TestCase):
    def test_selection_ignores_query_metric(self):
        rows = [_row(0, 0, 0.4, 0.9), _row(1, 3, 0.2, 0.1)]
        chosen = choose_by_support(rows)
        self.assertEqual((chosen["config_index"], chosen["restart"]), (1, 3))
        # Reverse query preference; selected trajectory must be identical.
        rows[0]["query"]["bce"], rows[1]["query"]["bce"] = 0.0, 99.0
        chosen = choose_by_support(rows)
        self.assertEqual((chosen["config_index"], chosen["restart"]), (1, 3))

    def test_nonfinite_support_is_not_selectable(self):
        failed, healthy = _row(0, 0, 0.0, 0.0), _row(1, 0, 0.5, 0.5)
        failed["support"] = {"bce": None, "accuracy": None, "nonfinite": True}
        self.assertEqual(choose_by_support([failed, healthy]), healthy)


if __name__ == "__main__":
    unittest.main()
