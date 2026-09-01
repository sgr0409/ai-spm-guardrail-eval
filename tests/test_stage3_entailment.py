"""Unit tests for guardrail.stage3_entailment.

The real DeBERTa-v3 cross-encoder is never loaded: EntailmentAuditor is
constructed via __new__ and self.model.predict is stubbed with fixed
logits, so these tests check the softmax-over-logits and label-index
lookup, which is where a silent bug (e.g. reading the wrong class index)
would actually live.
"""
import math
import unittest

import numpy as np

from guardrail.stage3_entailment import EntailmentAuditor, LABEL_ORDER


class FakeCrossEncoder:
    def __init__(self, logits_row):
        self._logits_row = logits_row

    def predict(self, pairs, convert_to_numpy=True):
        return np.array([self._logits_row])


def make_auditor(logits_row):
    auditor = EntailmentAuditor.__new__(EntailmentAuditor)
    auditor.model = FakeCrossEncoder(logits_row)
    auditor.entailment_idx = LABEL_ORDER.index("entailment")
    return auditor


class EntailmentProbTests(unittest.TestCase):
    def test_uniform_logits_give_exactly_one_third(self):
        auditor = make_auditor([0.0, 0.0, 0.0])
        prob = auditor.entailment_prob("response", "context")
        self.assertAlmostEqual(prob, 1.0 / 3.0, places=6)

    def test_dominant_entailment_logit_gives_high_probability(self):
        auditor = make_auditor([0.0, 10.0, 0.0])
        prob = auditor.entailment_prob("response", "context")
        self.assertGreater(prob, 0.99)

    def test_dominant_contradiction_logit_gives_low_entailment_probability(self):
        # entailment_idx must point at index 1, not just "the argmax":
        # this fails if the index lookup is wrong even though the softmax
        # itself is correct.
        auditor = make_auditor([10.0, 0.0, 0.0])
        prob = auditor.entailment_prob("response", "context")
        self.assertLess(prob, 0.01)

    def test_matches_manual_softmax_computation(self):
        logits = [2.0, 5.0, 1.0]
        auditor = make_auditor(logits)
        prob = auditor.entailment_prob("response", "context")

        exps = [math.exp(x - max(logits)) for x in logits]
        expected = exps[LABEL_ORDER.index("entailment")] / sum(exps)
        self.assertAlmostEqual(prob, expected, places=9)

    def test_label_order_has_entailment_at_index_one(self):
        # Locks in the id2label mapping the module docstring says was
        # confirmed against the model config, not assumed.
        self.assertEqual(LABEL_ORDER, ["contradiction", "entailment", "neutral"])


if __name__ == "__main__":
    unittest.main()
