"""Fetch public prompt-injection shift benchmarks into a common JSONL form.

The generated files are cached evaluation inputs, not new datasets.  Their
original identifiers and splits remain in ``shift_benchmarks_metadata.json``.
Users must comply with each upstream dataset's terms.
"""

import json
from pathlib import Path

from datasets import load_dataset


OUT = Path(__file__).resolve().parent / "external_shift"
OUT.mkdir(exist_ok=True)

SPECS = [
    {
        "slug": "deepset",
        "dataset": "deepset/prompt-injections",
        "config": None,
        "splits": ["train", "test"],
        "text": "text",
        "label": "label",
        "license": "Apache-2.0",
        "construct_note": "Positive labels include injection and task-out-of-scope prompts.",
    },
    {
        "slug": "xtram",
        "dataset": "xTRam1/safe-guard-prompt-injection",
        "config": None,
        "splits": ["test"],
        "text": "text",
        "label": "label",
        "license": "Not specified on the Hub card at retrieval time; fetched, not redistributed.",
        "construct_note": "Mixed benign and prompt-injection classification benchmark.",
    },
    {
        "slug": "jasper",
        "dataset": "JasperLS/prompt-injections",
        "config": None,
        "splits": ["test"],
        "text": "text",
        "label": "label",
        "license": "Not specified on the Hub card at retrieval time; fetched, not redistributed.",
        "construct_note": "Small external test split; estimates have correspondingly wide uncertainty.",
    },
    {
        "slug": "neuralchemy",
        "dataset": "neuralchemy/Prompt-injection-dataset",
        "config": "core",
        "splits": ["test"],
        "text": "text",
        "label": "label",
        "license": "Apache-2.0",
        "construct_note": "Core test split with benign hard negatives and categorized attacks.",
    },
]


def main():
    metadata = []
    for spec in SPECS:
        dataset = load_dataset(spec["dataset"], spec["config"])
        rows = []
        seen = set()
        for split in spec["splits"]:
            for row in dataset[split]:
                text = str(row[spec["text"]]).strip()
                if not text or text in seen:
                    continue
                label = int(row[spec["label"]])
                if label not in (0, 1):
                    raise ValueError(f"Unexpected label {label} in {spec['dataset']}")
                seen.add(text)
                rows.append({"text": text, "label": label, "source": spec["dataset"]})
        path = OUT / f"{spec['slug']}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        metadata.append({
            **{k: v for k, v in spec.items() if k not in ("text", "label")},
            "n": len(rows),
            "n_benign": sum(row["label"] == 0 for row in rows),
            "n_injection": sum(row["label"] == 1 for row in rows),
            "cached_path": str(path.relative_to(Path(__file__).resolve().parents[1])),
        })
        print(f"{spec['slug']}: {len(rows)} rows -> {path}")
    (OUT / "shift_benchmarks_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
