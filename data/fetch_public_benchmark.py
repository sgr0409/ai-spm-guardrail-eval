"""Fetches deepset/prompt-injections (public, HF-hosted) and caches it locally
as data/public_benchmark_eval.jsonl so the benchmark is reproducible offline.

Important caveat, disclosed in the paper rather than hidden: this dataset's
"malicious" label was built by deepset around a task-oriented chatbot
out-of-scope-detection use case, not a pure security-attack definition -- it
labels some generic out-of-scope requests (e.g., "generate SQL code," casual
roleplay framings) as positive alongside classic prompt-injection attempts.
It is a genuinely external, independently-labeled data source (which is the
point: does the shield's tuned operating point transfer at all to data it had
no hand in constructing), but its label semantics are not identical to our
synthetic benchmark's security-attack construct, and recall on it should be
read with that in mind.
"""
import json
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "public_benchmark_eval.jsonl"


def main():
    ds = load_dataset("deepset/prompt-injections")
    examples = []
    for split in ("train", "test"):
        for row in ds[split]:
            examples.append({
                "text": row["text"],
                "label": "adversarial" if row["label"] == 1 else "benign",
                "source": "deepset/prompt-injections",
                "source_split": split,
            })
    OUT.write_text("\n".join(json.dumps(e) for e in examples))
    n_adv = sum(1 for e in examples if e["label"] == "adversarial")
    print(f"Wrote {OUT}: {len(examples)} examples ({n_adv} adversarial / {len(examples) - n_adv} benign)")


if __name__ == "__main__":
    main()
