"""Unit tests for guardrail.pipeline.GuardrailPipeline.

Exercises the three-stage control flow (early exit at Stage 1 or Stage 2,
decision labeling, trace fields) against small fake shield/rag_control/
auditor objects, independent of any real model.
"""
import unittest

from guardrail.pipeline import GuardrailPipeline


class FakeShield:
    def __init__(self, injection_score, masked=None):
        self._injection_score = injection_score
        self._masked = masked if masked is not None else ("masked", [])
        self.mask_calls = 0

    def injection_score(self, prompt):
        return self._injection_score

    def mask_entities(self, prompt):
        self.mask_calls += 1
        return self._masked


class FakeRagControl:
    def __init__(self, poison_score):
        self._poison_score = poison_score
        self.calls = 0

    def poison_score(self, chunk):
        self.calls += 1
        return self._poison_score


class FakeAuditor:
    def __init__(self, entailment_prob):
        self._entailment_prob = entailment_prob
        self.calls = 0

    def entailment_prob(self, response, context):
        self.calls += 1
        return self._entailment_prob


def make_pipeline(injection_score, poison_score, entailment_prob,
                   tau_injection=0.5, tau_context=0.5, tau_entailment=0.5):
    shield = FakeShield(injection_score)
    rag_control = FakeRagControl(poison_score)
    auditor = FakeAuditor(entailment_prob)
    pipeline = GuardrailPipeline(
        shield, rag_control, auditor,
        tau_injection=tau_injection, tau_context=tau_context, tau_entailment=tau_entailment,
    )
    return pipeline, shield, rag_control, auditor


class GuardrailPipelineTests(unittest.TestCase):
    def test_blocks_at_stage1_and_never_reaches_later_stages(self):
        pipeline, shield, rag_control, auditor = make_pipeline(
            injection_score=0.9, poison_score=0.9, entailment_prob=0.9,
        )
        trace = pipeline.run("prompt", "chunk", "response")
        self.assertEqual(trace["decision"], "blocked_at_stage1")
        self.assertTrue(trace["stage1_blocked"])
        self.assertNotIn("stage2_blocked", trace)
        self.assertNotIn("stage3_approved", trace)
        self.assertEqual(shield.mask_calls, 0)
        self.assertEqual(rag_control.calls, 0)
        self.assertEqual(auditor.calls, 0)

    def test_stage1_boundary_is_inclusive(self):
        # inj_score == tau_injection must block (>=, not >).
        pipeline, _, _, _ = make_pipeline(
            injection_score=0.5, poison_score=0.0, entailment_prob=1.0,
            tau_injection=0.5,
        )
        trace = pipeline.run("prompt", "chunk", "response")
        self.assertEqual(trace["decision"], "blocked_at_stage1")

    def test_passes_stage1_masks_entities_then_blocks_at_stage2(self):
        pipeline, shield, rag_control, auditor = make_pipeline(
            injection_score=0.1, poison_score=0.9, entailment_prob=0.9,
        )
        trace = pipeline.run("prompt", "chunk", "response")
        self.assertEqual(trace["decision"], "blocked_at_stage2")
        self.assertFalse(trace["stage1_blocked"])
        self.assertTrue(trace["stage2_blocked"])
        self.assertNotIn("stage3_approved", trace)
        self.assertEqual(shield.mask_calls, 1)
        self.assertEqual(rag_control.calls, 1)
        self.assertEqual(auditor.calls, 0)

    def test_low_entailment_is_blocked_at_stage3(self):
        pipeline, shield, rag_control, auditor = make_pipeline(
            injection_score=0.1, poison_score=0.1, entailment_prob=0.2,
        )
        trace = pipeline.run("prompt", "chunk", "response")
        self.assertEqual(trace["decision"], "blocked_at_stage3")
        self.assertFalse(trace["stage3_approved"])
        self.assertEqual(auditor.calls, 1)

    def test_high_entailment_is_approved(self):
        pipeline, shield, rag_control, auditor = make_pipeline(
            injection_score=0.1, poison_score=0.1, entailment_prob=0.9,
        )
        trace = pipeline.run("prompt", "chunk", "response")
        self.assertEqual(trace["decision"], "approved")
        self.assertTrue(trace["stage3_approved"])

    def test_trace_includes_latency_fields_for_every_stage_reached(self):
        pipeline, _, _, _ = make_pipeline(
            injection_score=0.1, poison_score=0.1, entailment_prob=0.9,
        )
        trace = pipeline.run("prompt", "chunk", "response")
        for key in (
            "stage1_latency_s", "stage1_mask_latency_s",
            "stage2_latency_s", "stage3_latency_s", "total_latency_s",
        ):
            self.assertIn(key, trace)
            self.assertGreaterEqual(trace[key], 0.0)


if __name__ == "__main__":
    unittest.main()
