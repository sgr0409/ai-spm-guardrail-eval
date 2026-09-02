"""Fetch pinned public shift benchmarks into non-redistributed local JSONL files.

The tracked manifest records immutable upstream revisions, licenses, selection
rules, counts, and output checksums. Generated prompt text stays local and is
ignored by Git. Targets are de-duplicated within and across listed sources.
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download


DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "external_shift"
MANIFEST = DATA_DIR / "shift_benchmarks_manifest.json"
OUT.mkdir(exist_ok=True)

SPECS = [
    {
        "slug": "deepset", "dataset": "deepset/prompt-injections",
        "revision": "4f61ecb038e9c3fb77e21034b22511b523772cdd",
        "config": None, "splits": ["train", "test"], "text": "text", "label": "label",
        "license": "Apache-2.0",
        "construct_note": "Positive labels include injection and task-out-of-scope prompts.",
    },
    {
        "slug": "xtram", "dataset": "xTRam1/safe-guard-prompt-injection",
        "revision": "a3a877d608f37b7d20d9945671902df895ecdb46",
        "config": None, "splits": ["test"], "text": "text", "label": "label",
        "license": "Not specified on the Hub card at retrieval time; fetched, not redistributed.",
        "construct_note": "Mixed benign and prompt-injection classification benchmark.",
    },
    {
        "slug": "neuralchemy", "dataset": "neuralchemy/Prompt-injection-dataset",
        "revision": "7d70432dfcf47a821612cbf9d34e9d9e3ad20e75",
        "config": "core", "splits": ["test"], "text": "text", "label": "label",
        "license": "Apache-2.0",
        "construct_note": "Core test split with benign hard negatives and categorized attacks.",
    },
    {
        "slug": "safeguard", "dataset": "jcanode/safeguard-prompt-injection",
        "revision": "61fbe3588450fa9b47ac1176ca7b5d2cc932344c",
        "arrow_file": "data/test/data-00000-of-00001.arrow",
        "splits": ["test"], "text": "text", "label": "label",
        "license": "Apache-2.0", "max_per_label": 1000, "selection_seed": 1729,
        "construct_note": "Deterministic near-balanced subset capped at 1,000 rows per class from the held-out test split; attacks are independently generated and benign prompts come from public instruction corpora.",
    },
]


def canonical_text(value):
    """Normalize text for exact cross-source duplicate detection."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def deterministic_subset(rows, max_per_label, seed):
    """Select up to ``max_per_label`` rows per class without source-order bias."""
    selected = []
    for label in (0, 1):
        group = [row for row in rows if row["label"] == label]
        group.sort(key=lambda row: hashlib.sha256(
            f"{seed}\0{label}\0{canonical_text(row['text'])}".encode("utf-8")
        ).hexdigest())
        selected.extend(group[:max_per_label])
    return selected


def load_spec(spec):
    if "arrow_file" in spec:
        path = hf_hub_download(
            spec["dataset"], spec["arrow_file"], repo_type="dataset", revision=spec["revision"]
        )
        return {"test": Dataset.from_file(path)}
    return load_dataset(spec["dataset"], spec.get("config"), revision=spec["revision"])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metadata = []
    global_seen = set()
    for spec in SPECS:
        dataset = load_spec(spec)
        candidates, within_seen = [], set()
        within_removed = 0
        for split in spec["splits"]:
            for source_row in dataset[split]:
                text = str(source_row[spec["text"]]).strip()
                key = canonical_text(text)
                if not key or key in within_seen:
                    within_removed += 1
                    continue
                label = int(source_row[spec["label"]])
                if label not in (0, 1):
                    raise ValueError(f"Unexpected label {label} in {spec['dataset']}")
                within_seen.add(key)
                candidates.append({"text": text, "label": label, "source": spec["dataset"]})

        if spec.get("max_per_label"):
            candidates = deterministic_subset(candidates, spec["max_per_label"], spec["selection_seed"])

        rows, cross_removed = [], 0
        for row in candidates:
            key = canonical_text(row["text"])
            if key in global_seen:
                cross_removed += 1
                continue
            global_seen.add(key)
            rows.append(row)

        path = OUT / f"{spec['slug']}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        metadata.append({
            **{k: v for k, v in spec.items() if k not in ("text", "label")},
            "n": len(rows),
            "n_benign": sum(row["label"] == 0 for row in rows),
            "n_injection": sum(row["label"] == 1 for row in rows),
            "within_source_duplicates_removed": within_removed,
            "cross_source_duplicates_removed": cross_removed,
            "cached_path": str(path.relative_to(DATA_DIR.parent)), "sha256": sha256(path),
        })
        print(f"{spec['slug']}: {len(rows)} rows, {cross_removed} cross-source duplicates removed")

    MANIFEST.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
