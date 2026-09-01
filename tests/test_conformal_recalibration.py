import unittest

import numpy as np

from guardrail.conformal_recalibration import (
    conformal_predict,
    detect_score_shift,
    empirical_fpr_threshold,
    negative_conformal_pvalues,
    stratified_three_way_indices,
)


class ConformalRecalibrationTests(unittest.TestCase):
    def test_pvalues_use_conservative_upper_tail_ties(self):
        got = negative_conformal_pvalues([0.1, 0.5, 0.9], [0.1, 0.5, 0.7])
        np.testing.assert_allclose(got, [1.0, 0.75, 0.25])

    def test_small_calibration_bank_abstains_at_five_percent(self):
        # With n=9, the smallest attainable p-value is 0.1.
        pred = conformal_predict([100.0], np.arange(9), alpha=0.05)
        self.assertEqual(pred.tolist(), [0])

    def test_empirical_threshold_respects_observed_budget(self):
        scores = [0.1, 0.2, 0.7, 0.8]
        labels = [0, 0, 1, 1]
        threshold = empirical_fpr_threshold(scores, labels, target_fpr=0.0)
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.7)

    def test_empirical_threshold_can_step_above_tied_benign_score(self):
        scores = [0.2, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        threshold = empirical_fpr_threshold(scores, labels, target_fpr=0.05)
        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)

    def test_shift_trigger(self):
        decision = detect_score_shift(np.linspace(0, 0.2, 100), np.linspace(0.8, 1, 100))
        self.assertTrue(decision.triggered)

    def test_three_way_split_is_disjoint_and_complete(self):
        y = np.array([0] * 50 + [1] * 50)
        monitor, calibration, test = stratified_three_way_indices(y, seed=7)
        self.assertFalse(set(monitor) & set(calibration))
        self.assertFalse(set(monitor) & set(test))
        self.assertFalse(set(calibration) & set(test))
        self.assertEqual(set(np.r_[monitor, calibration, test]), set(range(100)))
        for idx in (monitor, calibration, test):
            self.assertEqual(set(y[idx]), {0, 1})


if __name__ == "__main__":
    unittest.main()
