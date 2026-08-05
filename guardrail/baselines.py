"""Comparison systems for the benchmark table.

LegacyRegexFilter    -- deterministic keyword/pattern blocklist. Fast, low recall
                         against paraphrase and obfuscation.
SecondaryTransformerVerifier -- a heavyweight zero-shot classification pass
                         (facebook/bart-large-mnli) run per prompt, standing in
                         for "call a second, general-purpose model to judge every
                         input" (the approach used by NeMo-Guardrails-style
                         secondary-LLM verification). This is the realistic
                         accuracy/latency ceiling the proposed pipeline is
                         benchmarked against.
OpenSourceInjectionClassifier -- deepset/deberta-v3-base-injection, a
                         DeBERTa-v3-base model fine-tuned specifically for
                         prompt-injection classification (not a general-purpose
                         NLI model pressed into service). A real, widely-used
                         open-source guardrail component, included as a fourth
                         comparison system. IMPORTANT: this model was
                         fine-tuned on deepset's prompt-injections dataset,
                         which is also this project's public external
                         benchmark (Section VI-B) -- so it is only a fair,
                         out-of-domain comparison on OUR OWN synthetic Stage 1
                         benchmark. Evaluating it on the deepset benchmark
                         itself would be train/test contamination, not a
                         genuine baseline, and run_benchmark.py deliberately
                         does not do that.
"""
import re
from transformers import pipeline

REGEX_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all|any|the)?\s*(previous|prior|above)\s*(instructions?|prompts?|rules?)",
        r"disregard (all|any|the)?\s*(previous|prior|above)",
        r"you are (now )?dan\b",
        r"developer mode",
        r"no (restrictions|filters|limitations)",
        r"reveal (your|the) (system|hidden) prompt",
        r"pretend (you|to) (are|be)",
        r"act as (if )?(you (are|were)|an?)\b.*\b(unrestricted|uncensored|jailbroken)",
        r"do anything now",
        r"bypass (your|all)? ?(safety|content)? ?(guidelines|filters|policy)",
        r"jailbreak",
        r"system prompt:?",
        r"\\u00|\\x[0-9a-f]{2}",  # crude unicode/hex-escape smuggling marker
    ]
]


class LegacyRegexFilter:
    def score(self, text):
        return 1.0 if any(p.search(text) for p in REGEX_PATTERNS) else 0.0

    def blocked(self, text, threshold=0.5):
        return self.score(text) >= threshold


class SecondaryTransformerVerifier:
    def __init__(self, model_name="facebook/bart-large-mnli", device=None):
        self.clf = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1 if device in (None, "cpu") else 0,
        )
        self.candidate_labels = ["prompt injection attack", "safe enterprise query"]

    def score(self, text):
        out = self.clf(text, candidate_labels=self.candidate_labels, multi_label=False)
        idx = out["labels"].index("prompt injection attack")
        return float(out["scores"][idx])

    def blocked(self, text, threshold=0.5):
        return self.score(text) >= threshold


class OpenSourceInjectionClassifier:
    def __init__(self, model_name="deepset/deberta-v3-base-injection", device=None):
        self.clf = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=-1 if device in (None, "cpu") else 0,
        )

    def score(self, text):
        out = self.clf(text[:2000])[0]  # list of {"label", "score"} for both classes
        for entry in out:
            if entry["label"] == "INJECTION":
                return float(entry["score"])
        raise ValueError(f"unexpected label set: {out}")

    def blocked(self, text, threshold=0.5):
        return self.score(text) >= threshold
