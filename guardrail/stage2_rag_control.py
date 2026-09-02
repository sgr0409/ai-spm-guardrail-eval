"""Stage 2: RAG context control.

Retrieved-document chunks can carry indirect prompt injections -- instructions
embedded in third-party text that a naive RAG loop would feed straight into the
model's context window. Stage 2 reuses the Stage-1 embedding space and the same
reference bank (the two stages differ in the text being screened and the
deployed threshold, not in scoring mechanism) and flags a chunk as poisoned when its similarity to the
injection reference bank exceeds tau_context, independent of how topically
relevant it is to the query.
"""
from guardrail.stage1_shield import SemanticShield

DEFAULT_WINDOW_SIZE = 20
DEFAULT_STRIDE = 10


def _word_windows(text, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE):
    words = text.split()
    if len(words) <= window_size:
        return [text]
    windows = []
    for start in range(0, len(words), stride):
        w = words[start:start + window_size]
        if not w:
            break
        windows.append(" ".join(w))
        if start + window_size >= len(words):
            break
    return windows


class RagContextControl:
    def __init__(self, shield: SemanticShield):
        # Deliberately shares the Stage-1 embedder + reference bank rather than
        # instantiating a second model: the paper's efficiency claim rests on
        # not paying for a second heavyweight model per stage.
        self.shield = shield

    def relevance_score(self, query, chunk):
        qv = self.shield._l2_normalize(self.shield.embed([query]))[0]
        cv = self.shield._l2_normalize(self.shield.embed([chunk]))[0]
        return float(qv @ cv)

    def poison_score(self, chunk):
        """Whole-chunk embedding similarity to the injection reference bank.
        This is the original Stage 2 design; Sec. VI-B shows it under-recalls
        because an instruction diluted across several sentences of legitimate
        content moves the whole-chunk vector only partway toward D."""
        return self.shield.injection_score(chunk)

    def poison_score_windowed(
        self, chunk, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE
    ):
        """Max injection-similarity over overlapping word windows instead of
        the whole chunk at once, to localize a short injected instruction
        buried in longer legitimate text. O(k) embedding calls per chunk of
        k windows instead of 1 -- the direct cost of the higher recall."""
        windows = _word_windows(chunk, window_size, stride)
        return max(self.shield.injection_score(w) for w in windows)

    def evaluate(self, query, chunk):
        return {
            "relevance": self.relevance_score(query, chunk),
            "poison_score": self.poison_score(chunk),
            "poison_score_windowed": self.poison_score_windowed(chunk),
        }
