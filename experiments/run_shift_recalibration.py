"""Evaluate drift-triggered conformal recalibration across external shifts.

This experiment is intentionally separate from ``run_benchmark.py``: it
freezes every detector, uses a source development distribution to establish
the original operating point, and then evaluates threshold transfer and
recalibration on disjoint monitor/calibration/test partitions of four public
target datasets.  One hundred fixed seeds are reported as overlapping sensitivity
analyses, not independent replications.
"""

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardrail.conformal_recalibration import (
    conformal_predict,
    detect_score_shift,
    empirical_fpr_threshold,
    stratified_three_way_indices,
)
from guardrail.stage1_shield import EMBEDDER_REVISION


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
ALPHA = 0.05
DRIFT_SIGNIFICANCE = 0.05
SEEDS = list(range(100))
MODEL_SPECS = {
    "protectai_v2": {
        "id": "protectai/deberta-v3-base-prompt-injection-v2",
        "revision": "90c9989b1a342275dd0d1a95aad283c04e075671",
    },
    "piguard": {
        "id": "leolee99/PIGuard",
        "revision": "dd78b24e330193a22d2293ac66922dd4f982f563",
    },
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detector_fingerprints(reference_bank):
    specs = {
        "semantic_shield": {
            "id": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": EMBEDDER_REVISION,
            "reference_bank": list(reference_bank),
        },
        **MODEL_SPECS,
    }
    return {
        detector: hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for detector, spec in specs.items()
    }


def cache_entry_is_current(cache, detector, domain, fingerprint, n_rows, model_fingerprint):
    score_key = f"{detector}__{domain}"
    # Bind provenance to each score array. A domain-only hash can be updated
    # when one detector is rescored and then make another detector's stale
    # array appear current.
    hash_key = f"data_sha256__{detector}__{domain}"
    model_key = f"model_sha256__{detector}"
    return (
        score_key in cache
        and hash_key in cache
        and model_key in cache
        and str(cache[hash_key]) == fingerprint
        and str(cache[model_key]) == model_fingerprint
        and len(cache[score_key]) == n_rows
    )


def l2_normalize(values):
    values = np.atleast_2d(values)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return values / norms


def score_shield(texts, reference_bank):
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", revision=EMBEDDER_REVISION, device="cpu"
    )
    reference = l2_normalize(model.encode(reference_bank, batch_size=64, convert_to_numpy=True,
                                          show_progress_bar=False))
    embedded = l2_normalize(model.encode(texts, batch_size=64, convert_to_numpy=True,
                                         show_progress_bar=True))
    return np.max(embedded @ reference.T, axis=1)


def score_classifier(texts, spec, batch_size=24):
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["revision"], trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["id"], revision=spec["revision"], trust_remote_code=True
    )
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512,
                                return_tensors="pt")
            logits = model(**encoded).logits
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
    del model
    return np.asarray(scores, dtype=float)


def source_development_indices(labels, fraction=0.4, seed=42):
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    selected = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        selected.extend(idx[:int(np.floor(len(idx) * fraction))])
    return np.asarray(selected)


def metrics_from_predictions(scores, labels, predictions):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    return {
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "fpr": fp / max(fp + tn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "auroc": roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else None,
        "auprc": average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else None,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def rounded(record):
    return {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)
            for k, v in record.items()}


def evaluate_target(
    source_scores,
    source_labels,
    target_scores,
    target_labels,
    seed,
    monitor_fraction=0.2,
    calibration_fraction=0.4,
):
    source_scores = np.asarray(source_scores)
    source_labels = np.asarray(source_labels)
    target_scores = np.asarray(target_scores)
    target_labels = np.asarray(target_labels)
    source_dev = source_development_indices(source_labels)
    source_threshold = empirical_fpr_threshold(
        source_scores[source_dev], source_labels[source_dev], ALPHA
    )
    source_benign = source_scores[source_dev][source_labels[source_dev] == 0]

    monitor, calibration, test = stratified_three_way_indices(
        target_labels,
        monitor_fraction=monitor_fraction,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    drift = detect_score_shift(source_scores[source_dev], target_scores[monitor], DRIFT_SIGNIFICANCE)
    cal_benign = target_scores[calibration][target_labels[calibration] == 0]
    cal_labels = target_labels[calibration]
    test_scores = target_scores[test]
    test_labels = target_labels[test]

    target_threshold = empirical_fpr_threshold(target_scores[calibration], cal_labels, ALPHA)
    frozen_pred = (test_scores >= source_threshold).astype(int)
    empirical_pred = (test_scores >= target_threshold).astype(int)
    source_conformal_pred = conformal_predict(test_scores, source_benign, ALPHA)
    target_conformal_pred = conformal_predict(test_scores, cal_benign, ALPHA)
    selected_reference = cal_benign if drift.triggered else source_benign
    dct_pred = conformal_predict(test_scores, selected_reference, ALPHA)

    return {
        "seed": seed,
        "n_monitor": int(len(monitor)),
        "n_calibration": int(len(calibration)),
        "n_calibration_benign": int(len(cal_benign)),
        "n_test": int(len(test)),
        "drift_statistic": round(drift.statistic, 6),
        "drift_p_value": round(drift.p_value, 8),
        "drift_triggered": drift.triggered,
        "target_labels_spent": int(len(calibration)) if drift.triggered else 0,
        "source_threshold": round(source_threshold, 8),
        "target_empirical_threshold": round(target_threshold, 8),
        "methods": {
            "frozen_source_threshold": rounded(metrics_from_predictions(test_scores, test_labels, frozen_pred)),
            "source_negative_conformal": rounded(metrics_from_predictions(test_scores, test_labels, source_conformal_pred)),
            "periodic_empirical_recalibration": rounded(metrics_from_predictions(test_scores, test_labels, empirical_pred)),
            "periodic_target_conformal": rounded(metrics_from_predictions(test_scores, test_labels, target_conformal_pred)),
            "drift_triggered_conformal": rounded(metrics_from_predictions(test_scores, test_labels, dct_pred)),
        },
    }


def aggregate_runs(runs):
    methods = runs[0]["methods"]
    aggregate = {}
    for method in methods:
        aggregate[method] = {}
        for metric in ("recall", "precision", "fpr", "f1", "auroc", "auprc"):
            values = [run["methods"][method][metric] for run in runs]
            aggregate[method][f"mean_{metric}"] = round(float(np.mean(values)), 6)
            aggregate[method][f"sd_{metric}"] = round(float(np.std(values, ddof=1)), 6)
        aggregate[method]["fpr_budget_exceedance_rate"] = round(
            float(np.mean([run["methods"][method]["fpr"] > ALPHA for run in runs])), 6
        )
    aggregate["trigger_rate"] = round(float(np.mean([r["drift_triggered"] for r in runs])), 6)
    aggregate["mean_target_labels_spent"] = round(float(np.mean([r["target_labels_spent"] for r in runs])), 2)
    return aggregate


def main():
    source_path = DATA / "stage1_eval.jsonl"
    source_rows = load_jsonl(source_path)
    domains = {"source_synthetic": source_rows}
    domain_paths = {"source_synthetic": source_path}
    for path in sorted((DATA / "external_shift").glob("*.jsonl")):
        domains[path.stem] = load_jsonl(path)
        domain_paths[path.stem] = path

    domain_texts = {name: [row["text"] for row in rows] for name, rows in domains.items()}
    domain_labels = {
        name: np.asarray([1 if row["label"] in (1, "adversarial") else 0 for row in rows])
        for name, rows in domains.items()
    }
    cache_path = RESULTS / "shift_scores.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    fingerprints = {name: file_sha256(path) for name, path in domain_paths.items()}
    reference_bank = json.loads((DATA / "reference_bank.json").read_text())
    model_fingerprints = detector_fingerprints(reference_bank)

    detector_specs = {"semantic_shield": None, **MODEL_SPECS}
    for detector, spec in detector_specs.items():
        missing = [name for name in domains if not cache_entry_is_current(
            cache, detector, name, fingerprints[name], len(domains[name]),
            model_fingerprints[detector],
        )]
        if not missing:
            continue
        joined = [text for name in missing for text in domain_texts[name]]
        print(f"Scoring {len(joined)} examples with {detector}...", flush=True)
        scores = score_shield(joined, reference_bank) if spec is None else score_classifier(joined, spec)
        cursor = 0
        for name in missing:
            n = len(domain_texts[name])
            cache[f"{detector}__{name}"] = scores[cursor:cursor + n]
            cache[f"data_sha256__{detector}__{name}"] = np.asarray(fingerprints[name])
            cache[f"model_sha256__{detector}"] = np.asarray(model_fingerprints[detector])
            cursor += n
        np.savez(cache_path, **cache)

    output = {
        "protocol": {
            "alpha": ALPHA,
            "drift_significance": DRIFT_SIGNIFICANCE,
            "seeds": SEEDS,
            "target_split": "20% label-hidden monitor / 40% labelled calibration / 40% test, stratified",
            "note": "Seeds are overlapping sensitivity analyses, not independent replications. "
                    "The target calibration labels are used only after the independent KS trigger. "
                    "Model-training overlap with third-party benchmark sources cannot be ruled out.",
        },
        "detectors": {
            "semantic_shield": {"id": "sentence-transformers/all-MiniLM-L6-v2 + fixed reference bank"},
            **MODEL_SPECS,
        },
        "datasets": json.loads((DATA / "shift_benchmarks_manifest.json").read_text()),
        "results": {},
    }
    for detector in detector_specs:
        output["results"][detector] = {}
        source_scores = cache[f"{detector}__source_synthetic"]
        source_labels = domain_labels["source_synthetic"]
        for domain in domains:
            if domain == "source_synthetic":
                continue
            print(f"Evaluating {detector} -> {domain}", flush=True)
            runs = [evaluate_target(source_scores, source_labels, cache[f"{detector}__{domain}"],
                                    domain_labels[domain], seed) for seed in SEEDS]
            output["results"][detector][domain] = {
                "runs": runs,
                "aggregate": aggregate_runs(runs),
                "label_budget_sensitivity": {},
            }
            for cal_fraction in (0.1, 0.2, 0.4):
                budget_runs = [
                    evaluate_target(
                        source_scores,
                        source_labels,
                        cache[f"{detector}__{domain}"],
                        domain_labels[domain],
                        seed,
                        calibration_fraction=cal_fraction,
                    )
                    for seed in SEEDS
                ]
                output["results"][detector][domain]["label_budget_sensitivity"][str(cal_fraction)] = {
                    "aggregate": aggregate_runs(budget_runs),
                }

        # A disjoint same-distribution control estimates how often the trigger
        # spends labels when no constructed shift is present.
        rng = np.random.RandomState(1729)
        ref_idx, target_idx = [], []
        for label in np.unique(source_labels):
            idx = np.flatnonzero(source_labels == label)
            rng.shuffle(idx)
            cut = len(idx) // 2
            ref_idx.extend(idx[:cut])
            target_idx.extend(idx[cut:])
        control_runs = [
            evaluate_target(
                source_scores[np.asarray(ref_idx)],
                source_labels[np.asarray(ref_idx)],
                source_scores[np.asarray(target_idx)],
                source_labels[np.asarray(target_idx)],
                seed,
            )
            for seed in SEEDS
        ]
        output["results"][detector]["same_distribution_control"] = {
            "note": "Disjoint halves of the synthetic source corpus; estimates trigger/label cost without a constructed shift.",
            "aggregate": aggregate_runs(control_runs),
        }

    # Auxiliary detector-type transfer: easy-set NLI risk -> hard-set NLI risk.
    raw = np.load(RESULTS / "raw_scores.npz")
    easy_scores = raw["stage3_risk_scores"]
    easy_labels = raw["stage3_test_labels"]
    hard_scores = raw["stage3_hard_risk_scores"]
    hard_labels = raw["stage3_hard_labels"]
    nli_runs = [evaluate_target(easy_scores, easy_labels, hard_scores, hard_labels, seed) for seed in SEEDS]
    output["stage3_auxiliary"] = {
        "note": "Author-constructed easy-to-hard NLI stress transfer; auxiliary, not an external replication.",
        "runs": nli_runs,
        "aggregate": aggregate_runs(nli_runs),
    }

    out_path = RESULTS / "shift_recalibration_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
