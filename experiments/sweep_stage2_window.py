"""Small hyperparameter sweep for Stage 2's windowed-scoring window_size and
stride, evaluated on the DEV split only (never the test split, preserving
the paper's train/dev/test discipline). Answers a real reviewer question:
were 12/6 chosen empirically, or just asserted? Honest answer before this
script existed: they were reasonable defaults, not swept. This script makes
the choice empirical and reports it either way -- if 12/6 is not the best
combo found, the paper should say so and either switch or disclose the gap.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import roc_auc_score

from guardrail.stage1_shield import SemanticShield
from guardrail.stage2_rag_control import RagContextControl, _word_windows

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stratified_split(examples, label_fn, dev_frac=0.4, seed=42):
    import random
    rng = random.Random(seed)
    by_label = {}
    for ex in examples:
        by_label.setdefault(label_fn(ex), []).append(ex)
    dev, test = [], []
    for label, group in by_label.items():
        group = group[:]
        rng.shuffle(group)
        cut = int(len(group) * dev_frac)
        dev.extend(group[:cut])
        test.extend(group[cut:])
    rng.shuffle(dev)
    return dev, test


def main():
    reference_bank = json.loads((DATA_DIR / "reference_bank.json").read_text())
    shield = SemanticShield(reference_bank, device="cpu")
    rag_control = RagContextControl(shield)

    stage2 = load_jsonl(DATA_DIR / "stage2_eval.jsonl")
    dev2, _ = stratified_split(stage2, lambda e: e["label"], dev_frac=0.4)
    labels = [1 if e["label"] == "poisoned" else 0 for e in dev2]

    grid = [
        (8, 4), (8, 8),
        (12, 6), (12, 12),
        (16, 8), (16, 16),
        (20, 10),
    ]

    results = []
    for window_size, stride in grid:
        scores = []
        for ex in dev2:
            windows = _word_windows(ex["chunk"], window_size, stride)
            s = max(shield.injection_score(w) for w in windows)
            scores.append(s)
        auroc = roc_auc_score(labels, scores)
        avg_windows = np.mean([len(_word_windows(ex["chunk"], window_size, stride)) for ex in dev2])
        results.append({"window_size": window_size, "stride": stride,
                         "dev_auroc": round(float(auroc), 4),
                         "avg_windows_per_chunk": round(float(avg_windows), 2)})
        print(results[-1])

    out = {"note": "AUROC on Stage 2 DEV split only (n={}), never test.".format(len(dev2)),
           "grid": results}
    out_path = Path(__file__).resolve().parent / "results" / "stage2_window_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
