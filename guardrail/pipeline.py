"""End-to-end orchestrator: F(x) -> y as a sequential 3-stage pipeline."""
import time

from guardrail.stage1_shield import SemanticShield
from guardrail.stage2_rag_control import RagContextControl
from guardrail.stage3_entailment import EntailmentAuditor


class GuardrailPipeline:
    def __init__(self, shield: SemanticShield, rag_control: RagContextControl, auditor: EntailmentAuditor,
                 tau_injection, tau_context, tau_entailment):
        self.shield = shield
        self.rag_control = rag_control
        self.auditor = auditor
        self.tau_injection = tau_injection
        self.tau_context = tau_context
        self.tau_entailment = tau_entailment

    def run(self, prompt, context_chunk, response, grounding_context=None):
        """Runs all three stages sequentially and returns a decision trace with
        per-stage wall-clock latency. Sequential timing is a conservative
        (worst-case) measurement; Sec. VI discusses the async deployment model
        separately from what is actually benchmarked here. ``context_chunk``
        is screened by Stage 2; ``grounding_context`` may contain the fuller
        context supporting the response and defaults to the same chunk."""
        if grounding_context is None:
            grounding_context = context_chunk
        t_start = time.perf_counter()
        trace = {}

        t0 = time.perf_counter()
        inj_score = self.shield.injection_score(prompt)
        trace["stage1_injection_score"] = inj_score
        trace["stage1_blocked"] = inj_score >= self.tau_injection
        trace["stage1_latency_s"] = time.perf_counter() - t0
        if trace["stage1_blocked"]:
            trace["decision"] = "blocked_at_stage1"
            trace["total_latency_s"] = time.perf_counter() - t_start
            return trace

        t0 = time.perf_counter()
        masked_prompt, entities = self.shield.mask_entities(prompt)
        trace["masked_prompt"] = masked_prompt
        trace["entities"] = entities
        trace["stage1_mask_latency_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        poison_score = self.rag_control.poison_score_windowed(context_chunk)
        trace["stage2_poison_score"] = poison_score
        trace["stage2_blocked"] = poison_score >= self.tau_context
        trace["stage2_latency_s"] = time.perf_counter() - t0
        if trace["stage2_blocked"]:
            trace["decision"] = "blocked_at_stage2"
            trace["total_latency_s"] = time.perf_counter() - t_start
            return trace

        t0 = time.perf_counter()
        entail_prob = self.auditor.entailment_prob(response, grounding_context)
        trace["stage3_entailment_prob"] = entail_prob
        trace["stage3_approved"] = entail_prob >= self.tau_entailment
        trace["stage3_latency_s"] = time.perf_counter() - t0

        trace["decision"] = "approved" if trace["stage3_approved"] else "blocked_at_stage3"
        trace["total_latency_s"] = time.perf_counter() - t_start
        return trace
