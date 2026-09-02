import json
import unittest
from pathlib import Path

import numpy as np

from data.fetch_shift_benchmarks import canonical_text, deterministic_subset
from experiments.run_shift_recalibration import cache_entry_is_current


ROOT = Path(__file__).resolve().parents[1]


class ShiftBenchmarkProvenanceTests(unittest.TestCase):
    def test_deterministic_subset_is_order_invariant_and_balanced(self):
        rows = [{"text": f"row {i}", "label": i % 2} for i in range(20)]
        forward = deterministic_subset(rows, 3, 1729)
        reverse = deterministic_subset(list(reversed(rows)), 3, 1729)
        self.assertEqual(forward, reverse)
        self.assertEqual([row["label"] for row in forward].count(0), 3)
        self.assertEqual([row["label"] for row in forward].count(1), 3)

    def test_generated_targets_have_no_normalized_cross_source_overlap(self):
        seen = set()
        manifest = json.loads((ROOT / "data" / "shift_benchmarks_manifest.json").read_text())
        for dataset in manifest:
            path = ROOT / dataset["cached_path"]
            keys = {
                canonical_text(json.loads(line)["text"])
                for line in path.read_text().splitlines() if line.strip()
            }
            self.assertEqual(len(keys), dataset["n"])
            self.assertTrue(seen.isdisjoint(keys))
            seen.update(keys)

    def test_score_cache_is_bound_to_content_and_length(self):
        cache = {
            "det__target": np.arange(3),
            "data_sha256__det__target": np.asarray("abc"),
            "model_sha256__det": np.asarray("model-a"),
        }
        self.assertTrue(cache_entry_is_current(cache, "det", "target", "abc", 3, "model-a"))
        self.assertFalse(cache_entry_is_current(cache, "det", "target", "changed", 3, "model-a"))
        self.assertFalse(cache_entry_is_current(cache, "det", "target", "abc", 2, "model-a"))
        self.assertFalse(cache_entry_is_current(cache, "det", "target", "abc", 3, "model-b"))

    def test_cache_fingerprint_cannot_validate_another_detector(self):
        cache = {
            "detector_a__target": np.arange(3),
            "detector_b__target": np.arange(3),
            "data_sha256__detector_a__target": np.asarray("new"),
            "data_sha256__detector_b__target": np.asarray("old"),
            "model_sha256__detector_a": np.asarray("model-a"),
            "model_sha256__detector_b": np.asarray("model-b"),
        }
        self.assertTrue(cache_entry_is_current(
            cache, "detector_a", "target", "new", 3, "model-a"
        ))
        self.assertFalse(cache_entry_is_current(
            cache, "detector_b", "target", "new", 3, "model-b"
        ))


if __name__ == "__main__":
    unittest.main()
