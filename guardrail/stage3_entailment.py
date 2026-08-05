"""Stage 3: Post-execution entailment auditing.

Scores P(entailment | R, C) with a DeBERTa-v3 NLI cross-encoder (Eq. 2 in the
paper). A response is approved for consumer routing iff the entailment
probability clears tau_entailment; otherwise it is treated as an unsupported /
hallucinated claim relative to the retrieved context.
"""
import numpy as np
from sentence_transformers import CrossEncoder

# confirmed via the model's config.json id2label, not assumed:
# {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL_ORDER = ["contradiction", "entailment", "neutral"]


class EntailmentAuditor:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-small", device=None):
        self.model = CrossEncoder(model_name, device=device)
        self.entailment_idx = LABEL_ORDER.index("entailment")

    def entailment_prob(self, response, context):
        logits = self.model.predict([(context, response)], convert_to_numpy=True)[0]  # shape (3,)
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        return float(probs[self.entailment_idx])
