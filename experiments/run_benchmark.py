"""Runs every experiment reported in the paper and writes results/results.json
plus results/raw_scores.npz (for figure generation in make_figures.py).

Hardware/software this was actually run on is recorded into the results file
(platform.platform(), torch version, thread count) so the paper's numbers are
reproducible and honestly scoped to this machine -- CPU inference on an Apple
M3 Pro laptop, not a GPU production cluster.
"""
import json
import importlib.metadata
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer

from guardrail.stage1_shield import SemanticShield
from guardrail.stage2_rag_control import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    RagContextControl,
    _word_windows,
)
from guardrail.stage3_entailment import EntailmentAuditor
from guardrail.baselines import LegacyRegexFilter, SecondaryTransformerVerifier, OpenSourceInjectionClassifier
from guardrail.pipeline import GuardrailPipeline

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
N_BOOTSTRAP = 2000
MPNET_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stratified_split(examples, label_fn, dev_frac=0.4, seed=SEED):
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
    rng.shuffle(test)
    return dev, test


def grouped_split(examples, group_fn, dev_frac=0.4, seed=SEED):
    """Split whole groups so related rows cannot cross development/test."""
    rng = random.Random(seed)
    groups = {}
    for example in examples:
        groups.setdefault(group_fn(example), []).append(example)
    keys = list(groups)
    rng.shuffle(keys)
    cut = int(len(keys) * dev_frac)
    dev_keys = set(keys[:cut])
    dev = [example for key in keys if key in dev_keys for example in groups[key]]
    test = [example for key in keys if key not in dev_keys for example in groups[key]]
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def stage1_reference_holdout_split(examples, seed=SEED):
    """Balanced 240/360 split with reference-omitted families test-only.

    The 75 attacks from families absent from the semantic reference bank must
    not influence threshold selection if their result is described as a
    family-level holdout. Development therefore contains 120 benign examples
    and 120 attacks from represented families; test contains the remaining 180
    benign, 105 represented-family attacks, and all 75 omitted-family attacks.
    """
    rng = random.Random(seed)
    benign = [e for e in examples if e["label"] == "benign"]
    represented = [
        e for e in examples
        if e["label"] == "adversarial" and e["in_reference_set"]
    ]
    omitted = [
        e for e in examples
        if e["label"] == "adversarial" and not e["in_reference_set"]
    ]
    for group in (benign, represented, omitted):
        rng.shuffle(group)
    if (len(benign), len(represented), len(omitted)) != (300, 225, 75):
        raise ValueError("unexpected Stage-1 class/family counts")
    dev = benign[:120] + represented[:120]
    test = benign[120:] + represented[120:] + omitted
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def select_threshold(dev_scores, dev_labels, target_fpr=0.05):
    """Smallest threshold (highest recall) whose FPR on dev stays <= target_fpr.
    Falls back to the FPR-minimizing threshold if no candidate meets the budget."""
    dev_scores = np.asarray(dev_scores)
    dev_labels = np.asarray(dev_labels)
    unique = np.unique(dev_scores)
    # With the inclusive decision rule ``score >= t``, an observed-value-only
    # sweep cannot place the boundary immediately above a tied benign score.
    # Include the representable boundary on both sides of every tie.
    candidates = sorted(set(np.r_[unique, np.nextafter(unique, np.inf)].tolist()), reverse=True)
    neg = dev_labels == 0
    n_neg = max(neg.sum(), 1)
    best = None
    best_fpr = None
    for t in candidates:
        pred = dev_scores >= t
        fpr = (pred & neg).sum() / n_neg
        if fpr <= target_fpr:
            best = t  # candidates sorted descending -> last one meeting budget is smallest/most-permissive
        if best_fpr is None or fpr < best_fpr:
            best_fpr = fpr
            fallback = t
    return best if best is not None else fallback


def latency_stats(latencies_s):
    ms = sorted(x * 1000 for x in latencies_s)
    n = len(ms)
    return {
        "n": n,
        "mean_ms": round(statistics.mean(ms), 3),
        "p50_ms": round(ms[int(0.50 * (n - 1))], 3),
        "p95_ms": round(ms[int(0.95 * (n - 1))], 3),
        "p99_ms": round(ms[int(0.99 * (n - 1))], 3),
        "throughput_qps": round(1.0 / statistics.mean(ms) * 1000, 2) if statistics.mean(ms) > 0 else None,
    }


def bootstrap_ci(scores, labels, threshold, n_boot=N_BOOTSTRAP, seed=SEED):
    """Percentile bootstrap (resampling test-set rows with replacement) for
    recall, precision, and AUROC. Returns (point_estimate, lo95, hi95) per
    metric. This is a standard, cheap way to attach uncertainty to a fixed-size
    test-set estimate without assuming a parametric sampling distribution."""
    rng = np.random.RandomState(seed)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(scores)
    recalls, precisions, aurocs = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        s, l = scores[idx], labels[idx]
        if len(set(l.tolist())) < 2:
            continue
        pred = (s >= threshold).astype(int)
        tp = int(((pred == 1) & (l == 1)).sum())
        fp = int(((pred == 1) & (l == 0)).sum())
        fn = int(((pred == 0) & (l == 1)).sum())
        recalls.append(tp / max(tp + fn, 1))
        precisions.append(tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0)
        aurocs.append(roc_auc_score(l, s))
    def ci(vals):
        if not vals:
            return None
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"lo95": round(float(lo), 4), "hi95": round(float(hi), 4)}
    return {"recall_ci": ci(recalls), "precision_ci": ci(precisions), "auroc_ci": ci(aurocs)}


def mcnemar_test(labels, scores_a, threshold_a, scores_b, threshold_b):
    """Exact (binomial) McNemar's test comparing two classifiers' correctness
    on the same paired test items. Tests whether the two systems' error
    patterns are symmetric; a small p-value means the accuracy difference is
    unlikely to be noise from this particular 360-item sample."""
    labels = np.asarray(labels)
    pred_a = (np.asarray(scores_a) >= threshold_a).astype(int)
    pred_b = (np.asarray(scores_b) >= threshold_b).astype(int)
    correct_a = pred_a == labels
    correct_b = pred_b == labels
    b = int((correct_a & ~correct_b).sum())  # A right, B wrong
    c = int((~correct_a & correct_b).sum())  # A wrong, B right
    n_discordant = b + c
    if n_discordant == 0:
        return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "p_value": 1.0}
    p = scipy_stats.binomtest(min(b, c), n_discordant, 0.5).pvalue
    return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "p_value": round(float(p), 6)}


def eval_binary(scores, labels, threshold, with_ci=True):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    preds = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fpr = fp / max(fp + tn, 1)
    auroc = roc_auc_score(labels, scores) if len(set(labels.tolist())) > 1 else None
    auprc = average_precision_score(labels, scores) if len(set(labels.tolist())) > 1 else None
    out = {
        "threshold": float(threshold),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "auroc": round(auroc, 4) if auroc is not None else None,
        "auprc": round(auprc, 4) if auprc is not None else None,
    }
    if with_ci:
        out["bootstrap_95ci"] = bootstrap_ci(scores, labels, threshold)
    return out


def count_params(model):
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return None


def maximum_full_coverage_matches(true_spans, predicted_spans):
    """Maximum one-to-one matching when a prediction fully covers a truth span.

    One broad prediction must not receive credit for multiple ground-truth
    entities. Labels are intentionally ignored because this benchmark measures
    whether sensitive text is redacted, not whether its PII type is classified.
    """
    edges = {
        ti: [
            pi for pi, pred in enumerate(predicted_spans)
            if pred[0] <= truth["start"] and pred[1] >= truth["end"]
        ]
        for ti, truth in enumerate(true_spans)
    }
    pred_to_true = {}

    def augment(ti, seen):
        for pi in edges[ti]:
            if pi in seen:
                continue
            seen.add(pi)
            if pi not in pred_to_true or augment(pred_to_true[pi], seen):
                pred_to_true[pi] = ti
                return True
        return False

    for ti in edges:
        augment(ti, set())
    return {(ti, pi) for pi, ti in pred_to_true.items()}


def main():
    t_run_start = time.time()
    torch.manual_seed(SEED)
    results = {
        "meta": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "seed": SEED,
            "bootstrap_resamples": N_BOOTSTRAP,
            "package_versions": {
                package: importlib.metadata.version(package)
                for package in (
                    "torch", "transformers", "sentence-transformers", "scikit-learn",
                    "numpy", "scipy", "datasets", "matplotlib",
                )
            },
        }
    }
    raw = {}  # raw score/label arrays exported for make_figures.py

    print("Loading models...", flush=True)
    reference_bank = json.loads((DATA_DIR / "reference_bank.json").read_text())
    shield = SemanticShield(reference_bank, device="cpu")
    rag_control = RagContextControl(shield)
    auditor = EntailmentAuditor(device="cpu")
    legacy = LegacyRegexFilter()
    secondary = SecondaryTransformerVerifier(device="cpu")
    opensource = OpenSourceInjectionClassifier(device="cpu")
    dedicated_embedder = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", revision=MPNET_REVISION, device="cpu"
    )

    results["models"] = {
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        "embedder_params": count_params(shield.embedder),
        "ner": "dslim/bert-base-NER",
        "entailment": "cross-encoder/nli-deberta-v3-small",
        "entailment_params": count_params(auditor.model.model),
        "secondary_verifier": "facebook/bart-large-mnli",
        "secondary_verifier_params": count_params(secondary.clf.model),
        "open_source_injection_classifier": "deepset/deberta-v3-base-injection",
        "open_source_injection_classifier_params": count_params(opensource.clf.model),
        "dedicated_embedder_ablation": "sentence-transformers/all-mpnet-base-v2",
        "dedicated_embedder_ablation_params": count_params(dedicated_embedder),
        "reference_bank_size": len(reference_bank),
        "revisions": {
            "embedder": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "ner": "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc",
            "entailment": "fa2804872c3b4bd748f38c0185cc85775361e735",
            "secondary_verifier": "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce",
            "open_source_injection_classifier": "80dda00d0b0d9a03917a7685e2ddbcd28e04dbb1",
            "dedicated_embedder_ablation": MPNET_REVISION,
        },
    }

    # ------------------------------------------------------------------ #
    # Stage 1: injection detection
    # ------------------------------------------------------------------ #
    print("Stage 1: injection detection...", flush=True)
    stage1 = load_jsonl(DATA_DIR / "stage1_eval.jsonl")
    dev1, test1 = stage1_reference_holdout_split(stage1)

    def run_stage1_system(score_fn, examples):
        scores, latencies = [], []
        for ex in examples:
            t0 = time.perf_counter()
            s = score_fn(ex["text"])
            latencies.append(time.perf_counter() - t0)
            scores.append(s)
        return scores, latencies

    dev_shield_scores, _ = run_stage1_system(shield.injection_score, dev1)
    test_shield_scores, test_shield_lat = run_stage1_system(shield.injection_score, test1)
    dev_legacy_scores, _ = run_stage1_system(legacy.score, dev1)
    test_legacy_scores, test_legacy_lat = run_stage1_system(legacy.score, test1)
    print("  scoring with secondary transformer verifier (slow, ~1 fwd pass/label/item)...", flush=True)
    dev_secondary_scores, _ = run_stage1_system(secondary.score, dev1)
    test_secondary_scores, test_secondary_lat = run_stage1_system(secondary.score, test1)
    print("  scoring with open-source injection classifier...", flush=True)
    dev_opensource_scores, _ = run_stage1_system(opensource.score, dev1)
    test_opensource_scores, test_opensource_lat = run_stage1_system(opensource.score, test1)

    dev_labels1 = [1 if e["label"] == "adversarial" else 0 for e in dev1]
    test_labels1 = [1 if e["label"] == "adversarial" else 0 for e in test1]

    tau_injection = select_threshold(dev_shield_scores, dev_labels1, target_fpr=0.05)
    tau_legacy = select_threshold(dev_legacy_scores, dev_labels1, target_fpr=0.05)
    tau_secondary = select_threshold(dev_secondary_scores, dev_labels1, target_fpr=0.05)
    tau_opensource = select_threshold(dev_opensource_scores, dev_labels1, target_fpr=0.05)

    shield_eval = eval_binary(test_shield_scores, test_labels1, tau_injection)
    legacy_eval = eval_binary(test_legacy_scores, test_labels1, tau_legacy)
    secondary_eval = eval_binary(test_secondary_scores, test_labels1, tau_secondary)
    opensource_eval = eval_binary(test_opensource_scores, test_labels1, tau_opensource)

    mcnemar_shield_vs_legacy = mcnemar_test(test_labels1, test_shield_scores, tau_injection,
                                             test_legacy_scores, tau_legacy)
    mcnemar_shield_vs_secondary = mcnemar_test(test_labels1, test_shield_scores, tau_injection,
                                                test_secondary_scores, tau_secondary)
    mcnemar_shield_vs_opensource = mcnemar_test(test_labels1, test_shield_scores, tau_injection,
                                                 test_opensource_scores, tau_opensource)

    def family_recall(scores, examples, threshold, in_ref):
        idx = [i for i, e in enumerate(examples) if e["label"] == "adversarial" and e["in_reference_set"] == in_ref]
        if not idx:
            return None
        hits = sum(1 for i in idx if scores[i] >= threshold)
        return round(hits / len(idx), 4)

    results["stage1_injection_detection"] = {
        "dataset": {"n_dev": len(dev1), "n_test": len(test1), "n_families": 12},
        "proposed_semantic_shield": {**shield_eval, "latency": latency_stats(test_shield_lat)},
        "legacy_regex": {**legacy_eval, "latency": latency_stats(test_legacy_lat)},
        "secondary_transformer_verifier": {**secondary_eval, "latency": latency_stats(test_secondary_lat)},
        "open_source_injection_classifier": {
            **opensource_eval, "latency": latency_stats(test_opensource_lat),
            "note": "deepset/deberta-v3-base-injection, fine-tuned specifically for this task (unlike the "
                    "zero-shot secondary-transformer baseline). Fair comparison HERE because it was not "
                    "fine-tuned on our synthetic data; NOT evaluated on the public benchmark in Section VI-B "
                    "because it WAS fine-tuned on that dataset (train/test contamination).",
        },
        "shield_generalization": {
            "recall_in_reference_families": family_recall(test_shield_scores, test1, tau_injection, True),
            "recall_held_out_families": family_recall(test_shield_scores, test1, tau_injection, False),
        },
        "significance": {
            "shield_vs_legacy_mcnemar": mcnemar_shield_vs_legacy,
            "shield_vs_secondary_mcnemar": mcnemar_shield_vs_secondary,
            "shield_vs_opensource_mcnemar": mcnemar_shield_vs_opensource,
            "note": "exact (binomial) McNemar's test on paired test-set predictions; "
                    "small p-value means the two systems' error patterns are unlikely to be symmetric by chance.",
        },
    }
    raw["stage1_test_labels"] = np.array(test_labels1)
    raw["stage1_shield_scores"] = np.array(test_shield_scores)
    raw["stage1_legacy_scores"] = np.array(test_legacy_scores)
    raw["stage1_secondary_scores"] = np.array(test_secondary_scores)
    raw["stage1_opensource_scores"] = np.array(test_opensource_scores)

    # ------------------------------------------------------------------ #
    # Stage 1b: public external benchmark (deepset/prompt-injections)
    # Applies the SAME thresholds tuned above -- no retuning on this data --
    # as a genuine transfer test to a dataset this project had no hand in.
    # ------------------------------------------------------------------ #
    print("Stage 1b: public benchmark transfer test...", flush=True)
    pub_path = DATA_DIR / "public_benchmark_eval.jsonl"
    if pub_path.exists():
        pub = load_jsonl(pub_path)
        pub_scores, pub_lat = run_stage1_system(shield.injection_score, pub)
        pub_legacy_scores, _ = run_stage1_system(legacy.score, pub)
        pub_secondary_scores, _ = run_stage1_system(secondary.score, pub)
        pub_labels = [1 if e["label"] == "adversarial" else 0 for e in pub]

        mcnemar_shield_vs_secondary_public = mcnemar_test(pub_labels, pub_scores, tau_injection,
                                                            pub_secondary_scores, tau_secondary)

        results["stage1_public_benchmark_transfer"] = {
            "source": "deepset/prompt-injections (Hugging Face, train+test splits combined)",
            "caveat": "this dataset's positive label was built around a task-oriented chatbot "
                      "out-of-scope-detection use case, not a pure security-attack definition -- it "
                      "also labels some generic out-of-scope requests (e.g. casual roleplay, unrelated "
                      "code-generation asks) as positive alongside classic injection attempts. Recall "
                      "here measures transfer to a related but not identical construct.",
            "n": len(pub),
            "n_adversarial": sum(pub_labels),
            "n_benign": len(pub_labels) - sum(pub_labels),
            "proposed_semantic_shield": eval_binary(pub_scores, pub_labels, tau_injection, with_ci=True),
            "legacy_regex": eval_binary(pub_legacy_scores, pub_labels, tau_legacy, with_ci=True),
            "secondary_transformer_verifier": eval_binary(pub_secondary_scores, pub_labels, tau_secondary, with_ci=True),
            "significance": {
                "shield_vs_secondary_mcnemar": mcnemar_shield_vs_secondary_public,
                "note": "exact (binomial) McNemar's test on paired public-benchmark predictions, at the "
                        "SAME already-tuned thresholds used throughout this transfer test (no retuning); "
                        "small p-value means the two systems' error patterns are unlikely to be symmetric "
                        "by chance.",
            },
        }
        raw["public_labels"] = np.array(pub_labels)
        raw["public_shield_scores"] = np.array(pub_scores)
        raw["public_legacy_scores"] = np.array(pub_legacy_scores)
        raw["public_secondary_scores"] = np.array(pub_secondary_scores)
    else:
        print("  SKIPPED: data/public_benchmark_eval.jsonl not found "
              "(run data/fetch_public_benchmark.py first)", flush=True)

    # ------------------------------------------------------------------ #
    # Stage 2: RAG context poisoning detection (whole-chunk vs. windowed)
    # ------------------------------------------------------------------ #
    print("Stage 2: RAG context poisoning detection...", flush=True)
    stage2 = load_jsonl(DATA_DIR / "stage2_eval.jsonl")
    dev2, test2 = stratified_split(stage2, lambda e: e["label"], dev_frac=0.4)

    def run_stage2(examples):
        poison_scores, poison_scores_win, rel_scores, latencies, latencies_win = [], [], [], [], []
        for ex in examples:
            t0 = time.perf_counter()
            p = rag_control.poison_score(ex["chunk"])
            latencies.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            pw = rag_control.poison_score_windowed(ex["chunk"])
            latencies_win.append(time.perf_counter() - t0)
            r = rag_control.relevance_score(ex["query"], ex["chunk"])
            poison_scores.append(p)
            poison_scores_win.append(pw)
            rel_scores.append(r)
        return poison_scores, poison_scores_win, rel_scores, latencies, latencies_win

    dev2_poison, dev2_poison_win, _, _, _ = run_stage2(dev2)
    test2_poison, test2_poison_win, test2_rel, test2_lat, test2_lat_win = run_stage2(test2)
    dev_labels2 = [1 if e["label"] == "poisoned" else 0 for e in dev2]
    test_labels2 = [1 if e["label"] == "poisoned" else 0 for e in test2]

    tau_context = select_threshold(dev2_poison, dev_labels2, target_fpr=0.05)
    tau_context_win = select_threshold(dev2_poison_win, dev_labels2, target_fpr=0.05)
    stage2_eval = eval_binary(test2_poison, test_labels2, tau_context)
    stage2_eval_win = eval_binary(test2_poison_win, test_labels2, tau_context_win)
    mcnemar_windowed = mcnemar_test(test_labels2, test2_poison_win, tau_context_win, test2_poison, tau_context)

    clean_rel = [r for r, l in zip(test2_rel, test_labels2) if l == 0]
    poison_rel = [r for r, l in zip(test2_rel, test_labels2) if l == 1]

    results["stage2_rag_context_control"] = {
        "dataset": {"n_dev": len(dev2), "n_test": len(test2)},
        "selected_on_development": {
            "window_size": DEFAULT_WINDOW_SIZE,
            "stride": DEFAULT_STRIDE,
        },
        "whole_chunk": {**stage2_eval, "latency": latency_stats(test2_lat)},
        "windowed_sub_chunk": {**stage2_eval_win, "latency": latency_stats(test2_lat_win)},
        "windowed_vs_whole_mcnemar": mcnemar_windowed,
        "relevance_score_by_class": {
            "clean_mean": round(float(np.mean(clean_rel)), 4),
            "poisoned_mean": round(float(np.mean(poison_rel)), 4),
            "note": "similar relevance means => poisoned chunks are not trivially separable by "
                    "topical relevance alone; detection relies on the injection-similarity signal.",
        },
    }
    raw["stage2_test_labels"] = np.array(test_labels2)
    raw["stage2_whole_scores"] = np.array(test2_poison)
    raw["stage2_windowed_scores"] = np.array(test2_poison_win)

    # ------------------------------------------------------------------ #
    # Stage 2 ablation: dedicated (non-shared) embedder instead of reusing
    # Stage 1's already-loaded model. Quantifies what "sharing" actually buys:
    # a second full embedding model in memory, for a detection-quality delta
    # that is either a wash or a loss, not a systematic gain.
    # ------------------------------------------------------------------ #
    print("Stage 2 ablation: dedicated (non-shared) embedder...", flush=True)

    def l2n(mat):
        mat = np.atleast_2d(mat)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return mat / norms

    dedicated_ref_matrix = l2n(dedicated_embedder.encode(reference_bank, convert_to_numpy=True, show_progress_bar=False))

    def dedicated_injection_score(text):
        v = l2n(dedicated_embedder.encode([text], convert_to_numpy=True, show_progress_bar=False))[0]
        return float(np.max(dedicated_ref_matrix @ v))

    def dedicated_poison_score_windowed(chunk):
        windows = _word_windows(chunk)
        return max(dedicated_injection_score(w) for w in windows)

    def run_stage2_dedicated(examples):
        scores, latencies = [], []
        for ex in examples:
            t0 = time.perf_counter()
            s = dedicated_poison_score_windowed(ex["chunk"])
            latencies.append(time.perf_counter() - t0)
            scores.append(s)
        return scores, latencies

    dev2_dedicated, _ = run_stage2_dedicated(dev2)
    test2_dedicated, test2_dedicated_lat = run_stage2_dedicated(test2)
    tau_context_dedicated = select_threshold(dev2_dedicated, dev_labels2, target_fpr=0.05)
    stage2_eval_dedicated = eval_binary(test2_dedicated, test_labels2, tau_context_dedicated)
    mcnemar_shared_vs_dedicated = mcnemar_test(test_labels2, test2_poison_win, tau_context_win,
                                                test2_dedicated, tau_context_dedicated)

    results["stage2_ablation_dedicated_embedder"] = {
        "note": "same windowed scoring mechanism as the shared-embedding result above, but using a second, "
                "independently-loaded encoder (all-mpnet-base-v2) instead of reusing Stage 1's MiniLM instance "
                "and reference bank.",
        "embedder": "sentence-transformers/all-mpnet-base-v2",
        "extra_params_loaded": count_params(dedicated_embedder),
        **stage2_eval_dedicated,
        "latency": latency_stats(test2_dedicated_lat),
        "shared_vs_dedicated_mcnemar": mcnemar_shared_vs_dedicated,
    }
    raw["stage2_dedicated_scores"] = np.array(test2_dedicated)

    # ------------------------------------------------------------------ #
    # Stage 3: entailment / hallucination detection (easy set)
    # ------------------------------------------------------------------ #
    print("Stage 3: entailment auditing...", flush=True)
    stage3 = load_jsonl(DATA_DIR / "stage3_eval.jsonl")
    # Each synthetic fact has a faithful and a hallucinated response. Keep the
    # entire factual context on one side of the split to prevent context-level
    # leakage into threshold selection.
    dev3, test3 = grouped_split(stage3, lambda e: e["context"], dev_frac=0.4)

    def run_stage3(examples):
        risk_scores, latencies = [], []
        for ex in examples:
            t0 = time.perf_counter()
            p = auditor.entailment_prob(ex["response"], ex["context"])
            latencies.append(time.perf_counter() - t0)
            risk_scores.append(1.0 - p)  # higher risk = more likely hallucinated
        return risk_scores, latencies

    dev3_risk, _ = run_stage3(dev3)
    test3_risk, test3_lat = run_stage3(test3)
    dev_labels3 = [1 if e["label"] == "hallucinated" else 0 for e in dev3]
    test_labels3 = [1 if e["label"] == "hallucinated" else 0 for e in test3]

    tau_risk = select_threshold(dev3_risk, dev_labels3, target_fpr=0.05)
    stage3_eval = eval_binary(test3_risk, test_labels3, tau_risk)
    tau_entailment = round(1.0 - tau_risk, 4)

    def type_recall(examples, risk_scores, labels, threshold, hallu_type):
        idx = [i for i, e in enumerate(examples) if e.get("type") == hallu_type]
        if not idx:
            return None
        hits = sum(1 for i in idx if risk_scores[i] >= threshold)
        return round(hits / len(idx), 4)

    results["stage3_entailment_auditing"] = {
        "dataset": {"n_dev": len(dev3), "n_test": len(test3)},
        "tau_entailment": tau_entailment,
        **stage3_eval,
        "latency": latency_stats(test3_lat),
        "recall_by_hallucination_type": {
            "numeric_contradiction": type_recall(test3, test3_risk, test_labels3, tau_risk, "numeric_contradiction"),
            "unsupported_addition": type_recall(test3, test3_risk, test_labels3, tau_risk, "unsupported_addition"),
        },
    }
    raw["stage3_test_labels"] = np.array(test_labels3)
    raw["stage3_risk_scores"] = np.array(test3_risk)

    # ------------------------------------------------------------------ #
    # Stage 3b: HARD entailment set -- subtle perturbations, transfer test
    # using the SAME tau_entailment tuned above (no retuning), plus a
    # threshold-free AUROC/AUPRC on the full hard set as the primary number.
    # ------------------------------------------------------------------ #
    print("Stage 3b: hard entailment set (subtle perturbations)...", flush=True)
    stage3_hard_path = DATA_DIR / "stage3_hard_eval.jsonl"
    if stage3_hard_path.exists():
        stage3_hard = load_jsonl(stage3_hard_path)
        hard_risk, hard_lat = run_stage3(stage3_hard)
        hard_labels = [1 if e["label"] == "hallucinated" else 0 for e in stage3_hard]
        hard_eval_at_easy_threshold = eval_binary(hard_risk, hard_labels, tau_risk)

        def hard_type_recall(hallu_type):
            idx = [i for i, e in enumerate(stage3_hard) if e.get("type") == hallu_type]
            if not idx:
                return None
            hits = sum(1 for i in idx if hard_risk[i] >= tau_risk)
            return round(hits / len(idx), 4)

        results["stage3_hard_entailment_auditing"] = {
            "note": "subtle perturbations (5-15% numeric drift, direction flips at unchanged "
                    "magnitude, topically-consistent unsupported additions) instead of the easy "
                    "set's 30-100% swings and off-topic fabrications. Evaluated at the SAME "
                    "tau_entailment tuned on the easy dev split -- a genuine transfer test, not a "
                    "re-optimized one.",
            "n": len(stage3_hard),
            **hard_eval_at_easy_threshold,
            "recall_by_hallucination_type": {
                "subtle_numeric_drift": hard_type_recall("subtle_numeric_drift"),
                "direction_flip": hard_type_recall("direction_flip"),
                "subtle_unsupported_addition": hard_type_recall("subtle_unsupported_addition"),
            },
            "latency": latency_stats(hard_lat),
        }
        raw["stage3_hard_labels"] = np.array(hard_labels)
        raw["stage3_hard_risk_scores"] = np.array(hard_risk)
    else:
        print("  SKIPPED: data/stage3_hard_eval.jsonl not found", flush=True)
        stage3_hard = []

    # ------------------------------------------------------------------ #
    # Ablation: incremental pipeline coverage (Stage1-only / +Stage2 / full)
    # over a mixed set spanning all three attack vectors plus a benign
    # negative control, all scored ONCE per trial and then combined
    # per-configuration so latency numbers are additive and comparable.
    # ------------------------------------------------------------------ #
    if not stage3_hard:
        print("  SKIPPED: requires stage3_hard_eval.jsonl (see Stage 3b above)", flush=True)
        results["ablation_pipeline_coverage"] = None
    else:
        print("Ablation: incremental pipeline coverage...", flush=True)
        rng_abl = random.Random(SEED)
        N_PER_GROUP = 100

        benign1 = [e["text"] for e in stage1 if e["label"] == "benign"]
        adv1 = [e["text"] for e in stage1 if e["label"] == "adversarial"]
        clean_chunks_ = [(e["query"], e["chunk"]) for e in stage2 if e["label"] == "clean"]
        poison_chunks_ = [(e["query"], e["chunk"]) for e in stage2 if e["label"] == "poisoned"]
        faithful_pairs_ = [(e["response"], e["context"]) for e in stage3 if e["label"] == "faithful"]
        hard_hallu_pairs_ = [(e["response"], e["context"]) for e in stage3_hard if e["label"] == "hallucinated"]

        # stage2_chunk (the retrieved document Stage 2 screens for poisoning) and
        # stage3_context (the grounding text Stage 3 checks the response against)
        # are DELIBERATELY separate fields: Stage 2's clean/poisoned RAG chunks
        # (HR/finance/legal/product policy facts) and Stage 3's faithful/
        # hallucinated pairs (synthetic company-financials records) were built
        # from disjoint fact universes (Section V-A) and are not topically
        # interchangeable. Scoring entailment against a random, unrelated
        # stage2_chunk instead of a response's own paired context was an
        # earlier bug in this ablation (near-zero entailment for almost every
        # trial regardless of true label, collapsing false-positive and
        # hallucination-coverage numbers into noise) -- fixed by keeping each
        # response paired with its own context throughout.
        def make_trials(n, prompt_pool, chunk_pool, response_context_pool):
            trials = []
            for _ in range(n):
                prompt = rng_abl.choice(prompt_pool)
                _, stage2_chunk = rng_abl.choice(chunk_pool)
                response, stage3_context = rng_abl.choice(response_context_pool)
                trials.append((prompt, stage2_chunk, response, stage3_context))
            return trials

        trial_groups = {
            "injection": make_trials(N_PER_GROUP, adv1, clean_chunks_, faithful_pairs_),
            "poisoning": make_trials(N_PER_GROUP, benign1, poison_chunks_, faithful_pairs_),
            "hallucination": make_trials(N_PER_GROUP, benign1, clean_chunks_, hard_hallu_pairs_),
            "benign": make_trials(N_PER_GROUP, benign1, clean_chunks_, faithful_pairs_),
        }

        def score_trial(prompt, stage2_chunk, response, stage3_context):
            s1 = shield.injection_score(prompt)
            s2 = rag_control.poison_score_windowed(stage2_chunk)
            s3 = 1.0 - auditor.entailment_prob(response, stage3_context)
            return s1, s2, s3

        scored = {group: [score_trial(*t) for t in trials] for group, trials in trial_groups.items()}

        CONFIGS = {
            "stage1_only": ("stage1",),
            "stage1_plus_stage2": ("stage1", "stage2"),
            "full_pipeline": ("stage1", "stage2", "stage3"),
        }
        STAGE_LATENCY_MS = {
            "stage1": results["stage1_injection_detection"]["proposed_semantic_shield"]["latency"]["mean_ms"],
            "stage2": results["stage2_rag_context_control"]["windowed_sub_chunk"]["latency"]["mean_ms"],
            "stage3": results["stage3_entailment_auditing"]["latency"]["mean_ms"],
        }

        def blocked(s1, s2, s3, active_stages):
            return (("stage1" in active_stages and s1 >= tau_injection)
                    or ("stage2" in active_stages and s2 >= tau_context_win)
                    or ("stage3" in active_stages and s3 >= tau_risk))

        ablation_results = {}
        for config_name, active in CONFIGS.items():
            attack_trials = scored["injection"] + scored["poisoning"] + scored["hallucination"]
            n_caught = sum(1 for (s1, s2, s3) in attack_trials if blocked(s1, s2, s3, active))
            n_benign_blocked = sum(1 for (s1, s2, s3) in scored["benign"] if blocked(s1, s2, s3, active))
            ablation_results[config_name] = {
                "active_stages": list(active),
                "overall_attack_coverage": round(n_caught / len(attack_trials), 4),
                "coverage_by_vector": {
                    v: round(sum(1 for (s1, s2, s3) in scored[v] if blocked(s1, s2, s3, active)) / len(scored[v]), 4)
                    for v in ("injection", "poisoning", "hallucination")
                },
                "false_positive_rate_on_benign": round(n_benign_blocked / len(scored["benign"]), 4),
                "additive_mean_latency_ms": round(sum(STAGE_LATENCY_MS[s] for s in active), 3),
            }

        results["ablation_pipeline_coverage"] = {
            "note": "each configuration's coverage is measured over the SAME n=100-per-vector mixed trial set "
                    "(injection / RAG poisoning / hard-set hallucination) plus a benign negative control, using "
                    "already-tuned per-stage thresholds throughout (no re-tuning per configuration). Latency is "
                    "additive from each stage's independently measured mean latency, not re-timed as a pipeline.",
            "n_per_vector": N_PER_GROUP,
            "configurations": ablation_results,
        }

    # ------------------------------------------------------------------ #
    # PII masking (Stage 1 secondary function)
    # ------------------------------------------------------------------ #
    print("PII masking evaluation...", flush=True)
    pii_examples = load_jsonl(DATA_DIR / "pii_eval.jsonl")
    tp = fp = fn = 0
    tp_by_type, fn_by_type = {}, {}
    mask_latencies = []
    for ex in pii_examples:
        t0 = time.perf_counter()
        masked_text, pred_spans = shield.mask_entities(ex["text"])
        mask_latencies.append(time.perf_counter() - t0)
        true_spans = ex["pii_spans"]
        matches = maximum_full_coverage_matches(true_spans, pred_spans)
        matched_true = {ti for ti, _ in matches}
        matched_pred = {pi for _, pi in matches}
        for ti, t in enumerate(true_spans):
            if ti in matched_true:
                tp += 1
                tp_by_type[t["type"]] = tp_by_type.get(t["type"], 0) + 1
            else:
                fn += 1
                fn_by_type[t["type"]] = fn_by_type.get(t["type"], 0) + 1
        fp += len(pred_spans) - len(matched_pred)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    results["pii_masking"] = {
        "n_sentences": len(pii_examples),
        "n_true_spans": sum(len(e["pii_spans"]) for e in pii_examples),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "recall_by_type": {
            t: round(tp_by_type.get(t, 0) / max(tp_by_type.get(t, 0) + fn_by_type.get(t, 0), 1), 4)
            for t in set(list(tp_by_type) + list(fn_by_type))
        },
        "latency": latency_stats(mask_latencies),
    }

    # ------------------------------------------------------------------ #
    # End-to-end pipeline latency (worst case: all 3 stages executed)
    # ------------------------------------------------------------------ #
    print("End-to-end pipeline latency...", flush=True)
    rng = random.Random(SEED)
    # Labels alone do not guarantee a complete path because every tuned stage
    # can produce false positives. Filter the timing pool by the deployed
    # thresholds, and keep each faithful response paired with its own context.
    benign_prompts = [
        e["text"] for e in stage1
        if e["label"] == "benign" and shield.injection_score(e["text"]) < tau_injection
    ]
    clean_chunks = [
        e["chunk"] for e in stage2
        if e["label"] == "clean"
        and rag_control.poison_score_windowed(e["chunk"]) < tau_context_win
    ]
    faithful_pairs = [
        (e["response"], e["context"]) for e in stage3
        if e["label"] == "faithful"
        and auditor.entailment_prob(e["response"], e["context"]) >= tau_entailment
    ]
    if not benign_prompts or not clean_chunks or not faithful_pairs:
        raise RuntimeError("no complete-path examples remain for end-to-end timing")

    n_e2e = 200
    pipeline = GuardrailPipeline(
        shield, rag_control, auditor, tau_injection, tau_context_win, tau_entailment
    )
    e2e_latencies = []
    for _ in range(n_e2e):
        prompt = rng.choice(benign_prompts)
        chunk = rng.choice(clean_chunks)
        response, grounding_context = rng.choice(faithful_pairs)
        trace = pipeline.run(prompt, chunk, response, grounding_context)
        if trace["decision"] != "approved" or "stage3_latency_s" not in trace:
            raise AssertionError("end-to-end timing trial exited before completing all stages")
        e2e_latencies.append(trace["total_latency_s"])

    results["end_to_end_pipeline_latency"] = {
        "note": "complete path sampled only from threshold-passing benign prompts, clean Stage-2 "
                "chunks, and faithful response/grounding-context pairs; all 3 stages execute with "
                "no early exit. Sequential single-request timing, CPU only.",
        "eligible_pool": {
            "benign_prompts": len(benign_prompts),
            "clean_stage2_chunks": len(clean_chunks),
            "faithful_response_context_pairs": len(faithful_pairs),
        },
        "n_trials": n_e2e,
        "latency": latency_stats(e2e_latencies),
    }

    # ------------------------------------------------------------------ #
    # Comparative summary table (mirrors the paper's Table I)
    # ------------------------------------------------------------------ #
    legacy_qps = results["stage1_injection_detection"]["legacy_regex"]["latency"]["throughput_qps"]
    results["comparative_summary"] = {
        metric_name: {
            "legacy_regex_filters": legacy_val,
            "secondary_transformer_verification": secondary_val,
            "proposed_stage1_shield": shield_val,
        }
        for metric_name, legacy_val, secondary_val, shield_val in [
            ("injection_detection_recall",
             results["stage1_injection_detection"]["legacy_regex"]["recall"],
             results["stage1_injection_detection"]["secondary_transformer_verifier"]["recall"],
             results["stage1_injection_detection"]["proposed_semantic_shield"]["recall"]),
            ("false_positive_rate",
             results["stage1_injection_detection"]["legacy_regex"]["fpr"],
             results["stage1_injection_detection"]["secondary_transformer_verifier"]["fpr"],
             results["stage1_injection_detection"]["proposed_semantic_shield"]["fpr"]),
            ("p99_latency_ms",
             results["stage1_injection_detection"]["legacy_regex"]["latency"]["p99_ms"],
             results["stage1_injection_detection"]["secondary_transformer_verifier"]["latency"]["p99_ms"],
             results["stage1_injection_detection"]["proposed_semantic_shield"]["latency"]["p99_ms"]),
            ("throughput_qps",
             results["stage1_injection_detection"]["legacy_regex"]["latency"]["throughput_qps"],
             results["stage1_injection_detection"]["secondary_transformer_verifier"]["latency"]["throughput_qps"],
             results["stage1_injection_detection"]["proposed_semantic_shield"]["latency"]["throughput_qps"]),
            ("relative_throughput_vs_legacy_pct",
             100.0,
             round(results["stage1_injection_detection"]["secondary_transformer_verifier"]["latency"]["throughput_qps"] / legacy_qps * 100, 1),
             round(results["stage1_injection_detection"]["proposed_semantic_shield"]["latency"]["throughput_qps"] / legacy_qps * 100, 1)),
        ]
    }

    results["meta"]["total_runtime_s"] = round(time.time() - t_run_start, 1)

    out_path = RESULTS_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    npz_path = RESULTS_DIR / "raw_scores.npz"
    np.savez(npz_path, **raw)
    print(f"\nWrote {out_path}")
    print(f"Wrote {npz_path}")
    print(json.dumps(results["comparative_summary"], indent=2))


if __name__ == "__main__":
    main()
