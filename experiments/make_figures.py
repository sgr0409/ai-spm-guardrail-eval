"""Generates every figure referenced in the paper from
experiments/results/raw_scores.npz and results.json. Run after
run_benchmark.py. Figures are written as PDF (vector, matches IEEEtran output)
into experiments/figures/ in this repo. The paper lives in a separate
repository; after regenerating, copy the contents of experiments/figures/
into that repo's figures/ directory to update the paper's copies.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

raw = np.load(RESULTS_DIR / "raw_scores.npz")
results = json.loads((RESULTS_DIR / "results.json").read_text())


def savefig(name):
    """Saves both a PDF (vector, for LaTeX) and a PNG (for quick preview
    without a PDF viewer, e.g. on GitHub) from the same figure."""
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


# --------------------------------------------------------------------- #
# Fig: Stage 1 ROC curves (shield vs. legacy vs. secondary)
# --------------------------------------------------------------------- #
plt.figure(figsize=(3.4, 2.8))
labels1 = raw["stage1_test_labels"]
STAGE1_SYSTEMS = [
    ("stage1_opensource_scores", "Open-source classifier", "-"),
    ("stage1_shield_scores", "Proposed shield", "-"),
    ("stage1_secondary_scores", "Secondary transformer", "--"),
    ("stage1_legacy_scores", "Legacy regex", ":"),
]
for key, name, style in STAGE1_SYSTEMS:
    fpr, tpr, _ = roc_curve(labels1, raw[key])
    plt.plot(fpr, tpr, style, label=name, linewidth=1.6)
plt.plot([0, 1], [0, 1], color="gray", linewidth=0.8, alpha=0.5)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Stage 1 injection detection ROC")
plt.legend(fontsize=6.5, loc="lower right")
savefig("stage1_roc.pdf")

# --------------------------------------------------------------------- #
# Fig: Stage 1 Precision-Recall curves
# --------------------------------------------------------------------- #
plt.figure(figsize=(3.4, 2.8))
for key, name, style in STAGE1_SYSTEMS:
    prec, rec, _ = precision_recall_curve(labels1, raw[key])
    plt.plot(rec, prec, style, label=name, linewidth=1.6)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Stage 1 injection detection PR curve")
plt.legend(fontsize=6.5, loc="lower left")
savefig("stage1_pr.pdf")

# --------------------------------------------------------------------- #
# Fig: Stage 2 whole-chunk vs windowed ROC
# --------------------------------------------------------------------- #
plt.figure(figsize=(3.4, 2.8))
labels2 = raw["stage2_test_labels"]
for key, name, style in [
    ("stage2_windowed_scores", "Windowed sub-chunk", "-"),
    ("stage2_whole_scores", "Whole-chunk (original)", "--"),
]:
    fpr, tpr, _ = roc_curve(labels2, raw[key])
    plt.plot(fpr, tpr, style, label=name, linewidth=1.6)
plt.plot([0, 1], [0, 1], color="gray", linewidth=0.8, alpha=0.5)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Stage 2 indirect-injection detection ROC")
plt.legend(fontsize=9, loc="lower right")
savefig("stage2_roc.pdf")

# --------------------------------------------------------------------- #
# Fig: Stage 3 easy vs hard ROC
# --------------------------------------------------------------------- #
plt.figure(figsize=(3.4, 2.8))
for lkey, skey, name, style in [
    ("stage3_test_labels", "stage3_risk_scores", "Easy set", "-"),
    ("stage3_hard_labels", "stage3_hard_risk_scores", "Hard set (subtle)", "--"),
]:
    fpr, tpr, _ = roc_curve(raw[lkey], raw[skey])
    plt.plot(fpr, tpr, style, label=name, linewidth=1.6)
plt.plot([0, 1], [0, 1], color="gray", linewidth=0.8, alpha=0.5)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Stage 3 hallucination ROC: easy vs. hard set")
plt.legend(fontsize=9, loc="lower right")
savefig("stage3_roc.pdf")

# --------------------------------------------------------------------- #
# Fig: Latency + throughput comparison (Stage 1 systems, log scale),
# combined into one 2-panel figure -- same two plots, one figure slot.
# --------------------------------------------------------------------- #
s1 = results["stage1_injection_detection"]
names = ["Legacy regex", "Proposed shield", "Open-source classifier", "Secondary transformer"]
colors = ["#888888", "#2b6cb0", "#2f855a", "#c05621"]
p99 = [
    s1["legacy_regex"]["latency"]["p99_ms"],
    s1["proposed_semantic_shield"]["latency"]["p99_ms"],
    s1["open_source_injection_classifier"]["latency"]["p99_ms"],
    s1["secondary_transformer_verifier"]["latency"]["p99_ms"],
]
qps = [
    s1["legacy_regex"]["latency"]["throughput_qps"],
    s1["proposed_semantic_shield"]["latency"]["throughput_qps"],
    s1["open_source_injection_classifier"]["latency"]["throughput_qps"],
    s1["secondary_transformer_verifier"]["latency"]["throughput_qps"],
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1))
bars = ax1.bar(names, p99, color=colors)
ax1.set_yscale("log")
ax1.set_ylabel("P99 latency (ms, log)", fontsize=10)
ax1.set_title("Latency", fontsize=11)
ax1.tick_params(axis="y", labelsize=9)
plt.setp(ax1.get_xticklabels(), rotation=20, ha="right", fontsize=9)
for b, v in zip(bars, p99):
    ax1.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:g}", ha="center", fontsize=8.5)

bars = ax2.bar(names, qps, color=colors)
ax2.set_yscale("log")
ax2.set_ylabel("Throughput (q/s, log)", fontsize=10)
ax2.set_title("Throughput", fontsize=11)
ax2.tick_params(axis="y", labelsize=9)
plt.setp(ax2.get_xticklabels(), rotation=20, ha="right", fontsize=9)
for b, v in zip(bars, qps):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:,.0f}" if v > 100 else f"{v:.1f}", ha="center", fontsize=8.5)
savefig("stage1_latency_throughput.pdf")

# --------------------------------------------------------------------- #
# Fig: Confusion matrices (shield stage1, stage2 windowed, stage3 hard)
# --------------------------------------------------------------------- #
def plot_confusion(ax, labels, scores, threshold, title, class_names):
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
plot_confusion(axes[0], labels1, raw["stage1_shield_scores"],
               results["stage1_injection_detection"]["proposed_semantic_shield"]["threshold"],
               "Stage 1 shield", ["benign", "adv."])
plot_confusion(axes[1], labels2, raw["stage2_windowed_scores"],
               results["stage2_rag_context_control"]["windowed_sub_chunk"]["threshold"],
               "Stage 2 windowed", ["clean", "poisoned"])
plot_confusion(axes[2], raw["stage3_hard_labels"], raw["stage3_hard_risk_scores"],
               results["stage3_hard_entailment_auditing"]["threshold"],
               "Stage 3 hard set", ["faithful", "halluc."])
savefig("confusion_matrices.pdf")

# --------------------------------------------------------------------- #
# Fig: Ablation -- incremental pipeline coverage by attack vector
# --------------------------------------------------------------------- #
abl = results.get("ablation_pipeline_coverage")
if abl:
    configs = ["stage1_only", "stage1_plus_stage2", "full_pipeline"]
    config_labels = ["Stage 1\nonly", "Stage 1\n+ Stage 2", "Full\npipeline"]
    vectors = ["injection", "poisoning", "hallucination"]
    vector_labels = ["Injection", "RAG poisoning", "Hallucination"]
    vector_colors = ["#2b6cb0", "#c05621", "#6b46c1"]

    x = np.arange(len(configs))
    width = 0.25
    plt.figure(figsize=(3.6, 2.8))
    for i, (vec, vlabel, color) in enumerate(zip(vectors, vector_labels, vector_colors)):
        vals = [abl["configurations"][c]["coverage_by_vector"][vec] for c in configs]
        plt.bar(x + (i - 1) * width, vals, width, label=vlabel, color=color)
    plt.xticks(x, config_labels)
    plt.ylabel("Coverage (recall)")
    plt.ylim(0, 1.05)
    plt.title("Ablation: coverage by attack vector")
    plt.legend(fontsize=9, loc="upper left")
    savefig("ablation_coverage.pdf")
else:
    print("Ablation results not found in results.json; skipping ablation figure")

print("All figures written to", FIG_DIR)
