"""Unit tests for guardrail.stage1_shield.

These tests avoid loading the real SentenceTransformer/NER models: instances
are constructed via __new__ (bypassing __init__) and the model-dependent
attributes (embed, ner) are stubbed directly. This exercises the actual
similarity, overlap-resolution, and regex logic without any network or
GPU/CPU model-inference cost.
"""
import unittest

import numpy as np

from guardrail.stage1_shield import SemanticShield, STRUCTURED_PII_PATTERNS


def make_shield():
    return SemanticShield.__new__(SemanticShield)


class L2NormalizeTests(unittest.TestCase):
    def test_normalizes_rows_to_unit_length(self):
        mat = np.array([[3.0, 4.0], [1.0, 0.0]])
        out = SemanticShield._l2_normalize(mat)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0])
        np.testing.assert_allclose(out[0], [0.6, 0.8])

    def test_zero_vector_does_not_divide_by_zero(self):
        mat = np.array([[0.0, 0.0]])
        out = SemanticShield._l2_normalize(mat)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_promotes_1d_input_to_2d(self):
        out = SemanticShield._l2_normalize(np.array([3.0, 4.0]))
        self.assertEqual(out.shape, (1, 2))


class InjectionScoreTests(unittest.TestCase):
    def test_returns_max_cosine_similarity_to_reference_bank(self):
        shield = make_shield()
        shield.reference_matrix = SemanticShield._l2_normalize(
            np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        shield.embed = lambda texts: np.array([[0.0, 5.0]])
        score = shield.injection_score("anything")
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_orthogonal_query_scores_near_zero(self):
        shield = make_shield()
        shield.reference_matrix = SemanticShield._l2_normalize(np.array([[1.0, 0.0]]))
        shield.embed = lambda texts: np.array([[0.0, 1.0]])
        score = shield.injection_score("anything")
        self.assertAlmostEqual(score, 0.0, places=6)


class MaskEntitiesTests(unittest.TestCase):
    def test_overlapping_ner_spans_keep_the_longer_one(self):
        shield = make_shield()
        text = "abcdefghij"
        # Two overlapping spans starting at the same offset: the algorithm
        # must prefer the longer one (0,6) over the shorter one (0,3),
        # per the documented "prefer the longest span" rule.
        shield.ner = lambda t: [
            {"entity_group": "PER", "start": 0, "end": 3},
            {"entity_group": "ORG", "start": 0, "end": 6},
        ]
        masked, kept = shield.mask_entities(text)
        self.assertEqual(kept, [(0, 6, "ORG")])
        self.assertEqual(masked, "[REDACTED:ORG]ghij")

    def test_longer_later_start_wins_global_overlap_resolution(self):
        shield = make_shield()
        text = "abcdefghij"
        shield.ner = lambda t: [
            {"entity_group": "PER", "start": 0, "end": 4},
            {"entity_group": "ORG", "start": 2, "end": 9},
        ]
        masked, kept = shield.mask_entities(text)
        self.assertEqual(kept, [(2, 9, "ORG")])
        self.assertEqual(masked, "ab[REDACTED:ORG]j")

    def test_non_overlapping_spans_are_both_kept(self):
        shield = make_shield()
        text = "Contact John Smith at john@example.com for details."
        start = text.index("John Smith")
        end = start + len("John Smith")
        shield.ner = lambda t: [
            {"entity_group": "PER", "start": start, "end": end},
        ]
        masked, kept = shield.mask_entities(text)
        self.assertIn("[REDACTED:PER]", masked)
        self.assertIn("[REDACTED:EMAIL]", masked)
        self.assertEqual(len(kept), 2)

    def test_ner_spans_outside_named_entity_labels_are_dropped(self):
        shield = make_shield()
        text = "abcdef"
        shield.ner = lambda t: [{"entity_group": "DATE", "start": 0, "end": 3}]
        masked, kept = shield.mask_entities(text)
        self.assertEqual(kept, [])
        self.assertEqual(masked, text)


class StructuredPiiPatternTests(unittest.TestCase):
    def test_email_pattern_matches(self):
        m = STRUCTURED_PII_PATTERNS["EMAIL"].search("reach me at a.b+c@sub.example.com now")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(), "a.b+c@sub.example.com")

    def test_ssn_pattern_matches(self):
        m = STRUCTURED_PII_PATTERNS["SSN"].search("SSN 123-45-6789 on file")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(), "123-45-6789")

    def test_credit_card_pattern_matches_16_digits(self):
        m = STRUCTURED_PII_PATTERNS["CREDIT_CARD"].search("card 4111111111111111 charged")
        self.assertIsNotNone(m)

    def test_phone_pattern_includes_leading_parenthesis(self):
        # Regression test for the documented bug: a leading "(" is a
        # non-word character, so \b previously failed to match right after
        # a preceding space and silently dropped the opening paren.
        m = STRUCTURED_PII_PATTERNS["PHONE"].search("Call (206) 555-1234 now")
        self.assertIsNotNone(m)
        self.assertTrue(m.group().startswith("("))
        self.assertEqual(m.group(), "(206) 555-1234")

    def test_phone_pattern_matches_without_parens_too(self):
        m = STRUCTURED_PII_PATTERNS["PHONE"].search("call 206-555-1234 now")
        self.assertIsNotNone(m)


if __name__ == "__main__":
    unittest.main()
