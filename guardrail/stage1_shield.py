"""Stage 1: Semantic shielding.

Two independent functions, both timed separately in the benchmark:
  1. injection_score(text) -- cosine similarity of the input embedding against a
     reference bank D of known adversarial-prompt vectors (Eq. 1 in the paper).
  2. mask_entities(text) -- named-entity + structured-PII redaction, applied to
     text that passes the injection check, before it is forwarded downstream.
"""
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

STRUCTURED_PII_PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    # (?<!\d)/(?!\d) instead of \b: a leading "(" is a non-word char, so \b
    # fails to match right after a preceding space, silently dropping the
    # opening paren from phone numbers like "(206) 555-1234".
    "PHONE": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}

NAMED_ENTITY_LABELS = {"PER", "ORG", "LOC", "MISC"}
EMBEDDER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
NER_REVISION = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"


class SemanticShield:
    def __init__(
        self,
        reference_texts,
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        ner_model_name="dslim/bert-base-NER",
        embed_revision=EMBEDDER_REVISION,
        ner_revision=NER_REVISION,
        device=None,
    ):
        self.embedder = SentenceTransformer(embed_model_name, revision=embed_revision, device=device)
        self.ner = pipeline(
            "ner",
            model=ner_model_name,
            revision=ner_revision,
            aggregation_strategy="simple",
            device=-1 if device in (None, "cpu") else 0,
        )
        self.reference_texts = list(reference_texts)
        self.reference_matrix = self._l2_normalize(
            self.embedder.encode(self.reference_texts, convert_to_numpy=True, show_progress_bar=False)
        )

    @staticmethod
    def _l2_normalize(mat):
        mat = np.atleast_2d(mat)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return mat / norms

    def embed(self, texts):
        return self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def injection_score(self, text):
        """Sim(x, D) = max_i cos(v_x, v_i), Eq. (1)."""
        v = self._l2_normalize(self.embed([text]))[0]
        sims = self.reference_matrix @ v
        return float(np.max(sims))

    def mask_entities(self, text):
        """Redact named entities and structured PII. Returns (masked_text, spans)."""
        spans = []
        for ent in self.ner(text):
            label = ent["entity_group"]
            if label in NAMED_ENTITY_LABELS:
                spans.append((int(ent["start"]), int(ent["end"]), label))
        for label, pattern in STRUCTURED_PII_PATTERNS.items():
            for m in pattern.finditer(text):
                spans.append((m.start(), m.end(), label))

        # Resolve overlaps globally by preferring the longest span (not merely
        # the longest span among candidates with the same start), then replace
        # the retained spans right-to-left.
        spans.sort(key=lambda s: (-(s[1] - s[0]), s[0], s[1], s[2]))
        kept = []
        for start, end, label in spans:
            if any(not (end <= k[0] or start >= k[1]) for k in kept):
                continue
            kept.append((start, end, label))

        out = text
        for start, end, label in sorted(kept, key=lambda s: -s[0]):
            out = out[:start] + f"[REDACTED:{label}]" + out[end:]
        return out, kept
