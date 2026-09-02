import unittest

from experiments.run_benchmark import (
    grouped_split,
    maximum_full_coverage_matches,
    stage1_reference_holdout_split,
)


class BenchmarkMetricTests(unittest.TestCase):
    def test_stage1_omitted_families_are_test_only(self):
        examples = [
            {"id": f"b{i}", "label": "benign"} for i in range(300)
        ] + [
            {"id": f"r{i}", "label": "adversarial", "in_reference_set": True}
            for i in range(225)
        ] + [
            {"id": f"o{i}", "label": "adversarial", "in_reference_set": False}
            for i in range(75)
        ]
        dev, test = stage1_reference_holdout_split(examples)
        self.assertEqual((len(dev), len(test)), (240, 360))
        self.assertFalse(any(
            e["label"] == "adversarial" and not e["in_reference_set"] for e in dev
        ))
        self.assertEqual(sum(
            e["label"] == "adversarial" and not e["in_reference_set"] for e in test
        ), 75)

    def test_grouped_split_never_leaks_a_group(self):
        rows = [
            {"group": group, "row": row}
            for group in range(10) for row in range(2)
        ]
        dev, test = grouped_split(rows, lambda value: value["group"], seed=7)
        self.assertTrue({row["group"] for row in dev}.isdisjoint(
            {row["group"] for row in test}
        ))

    def test_one_broad_prediction_cannot_match_two_true_entities(self):
        truth = [
            {"start": 0, "end": 4, "type": "name"},
            {"start": 6, "end": 10, "type": "name"},
        ]
        predicted = [(0, 10, "PER")]
        self.assertEqual(len(maximum_full_coverage_matches(truth, predicted)), 1)

    def test_matching_maximizes_number_of_covered_entities(self):
        truth = [
            {"start": 0, "end": 4, "type": "name"},
            {"start": 6, "end": 10, "type": "email"},
        ]
        predicted = [(0, 10, "PER"), (0, 4, "PER")]
        self.assertEqual(len(maximum_full_coverage_matches(truth, predicted)), 2)


if __name__ == "__main__":
    unittest.main()
