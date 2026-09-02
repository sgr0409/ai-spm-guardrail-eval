"""Prequential evaluation of regime-adaptive DTCR on constructed streams.

Each run contains twelve ordered, non-overlapping monitor/calibration/test
windows: two source windows, then two windows from each new public regime,
followed by a recurrence of xTRam. Labels in monitor/test windows are used
only for offline evaluation. Periodic conformalization buys every calibration
window; source-referenced DTCR and regime-adaptive DTCR buy labels only after
their label-hidden monitor triggers.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardrail.conformal_recalibration import (
    RegimeAdaptiveDTCR,
    conformal_predict,
    detect_score_shift,
)
from experiments.run_shift_recalibration import (
    cache_entry_is_current,
    detector_fingerprints,
    file_sha256,
    metrics_from_predictions,
    rounded,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = Path(__file__).resolve().parent / "results"
SEEDS = list(range(100))
ALPHA = 0.05
FALSE_TRIGGER_BUDGET = 0.05
SCHEDULE = (
    ["source_synthetic"] * 2
    + ["deepset"] * 2
    + ["xtram"] * 2
    + ["neuralchemy"] * 2
    + ["safeguard"] * 2
    + ["xtram"] * 2
)
PER_LABEL = {"monitor": 80, "calibration": 20, "test": 20}
SOURCE_REFERENCE_PER_LABEL = 60
DETECTORS = ("semantic_shield", "protectai_v2", "piguard")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def allocate_stream(labels, seed):
    """Return a source reference and disjoint ordered windows for one seed."""
    rng = np.random.RandomState(seed)
    pools, offsets = {}, {}
    for domain, y in labels.items():
        pools[domain], offsets[domain] = {}, {0: 0, 1: 0}
        for label in (0, 1):
            idx = np.flatnonzero(y == label)
            rng.shuffle(idx)
            pools[domain][label] = idx

    source_reference = []
    for label in (0, 1):
        source_reference.extend(pools["source_synthetic"][label][:SOURCE_REFERENCE_PER_LABEL])
        offsets["source_synthetic"][label] = SOURCE_REFERENCE_PER_LABEL

    windows = []
    per_label_total = sum(PER_LABEL.values())
    for position, domain in enumerate(SCHEDULE):
        parts = {name: [] for name in PER_LABEL}
        for label in (0, 1):
            start = offsets[domain][label]
            stop = start + per_label_total
            selected = pools[domain][label][start:stop]
            if len(selected) != per_label_total:
                raise ValueError(f"insufficient class-{label} rows for {domain} at window {position}")
            offsets[domain][label] = stop
            cursor = 0
            for name, size in PER_LABEL.items():
                parts[name].extend(selected[cursor:cursor + size])
                cursor += size
        windows.append({name: np.asarray(idx, dtype=int) for name, idx in parts.items()})
    return np.asarray(source_reference, dtype=int), windows


def aggregate_seed_runs(runs):
    aggregate = {"methods": {}}
    for method in runs[0]["methods"]:
        aggregate["methods"][method] = {}
        for metric in ("recall", "precision", "fpr", "f1"):
            values = [run["methods"][method][metric] for run in runs]
            aggregate["methods"][method][f"mean_{metric}"] = round(float(np.mean(values)), 6)
            aggregate["methods"][method][f"sd_{metric}"] = round(float(np.std(values, ddof=1)), 6)
        aggregate["methods"][method]["mean_labels"] = round(
            float(np.mean([run["methods"][method]["labels"] for run in runs])), 2
        )
    for key in (
        "ra_triggers", "ra_false_triggers", "ra_detected_transitions",
        "ra_first_window_detections", "static_triggers",
    ):
        aggregate[f"mean_{key}"] = round(float(np.mean([run[key] for run in runs])), 4)
    periodic = aggregate["methods"]["periodic_target_conformal"]["mean_labels"]
    adaptive = aggregate["methods"]["regime_adaptive_dtcr"]["mean_labels"]
    static = aggregate["methods"]["source_referenced_dtcr"]["mean_labels"]
    aggregate["adaptive_label_reduction_vs_periodic"] = round(1.0 - adaptive / periodic, 6)
    aggregate["adaptive_label_reduction_vs_source_referenced"] = round(1.0 - adaptive / static, 6)
    aggregate["first_window_transition_detection_rate"] = round(
        aggregate["mean_ra_first_window_detections"] / 5.0, 6
    )
    aggregate["within_two_windows_transition_detection_rate"] = round(
        aggregate["mean_ra_detected_transitions"] / 5.0, 6
    )
    return aggregate


def evaluate_detector(detector, scores, labels):
    runs = []
    horizon = len(SCHEDULE)
    significance = FALSE_TRIGGER_BUDGET / horizon
    for seed in SEEDS:
        source_reference_idx, windows = allocate_stream(labels, seed)
        source_scores = scores["source_synthetic"]
        source_labels = labels["source_synthetic"]
        source_monitor = source_scores[source_reference_idx]
        source_benign = source_monitor[source_labels[source_reference_idx] == 0]
        controller = RegimeAdaptiveDTCR(
            source_monitor, source_benign, horizon,
            false_trigger_budget=FALSE_TRIGGER_BUDGET, alpha=ALPHA,
        )
        current_regime = "source_synthetic"
        previous_stream_regime = "source_synthetic"
        accumulated = {
            method: {"pred": [], "scores": [], "labels": [], "label_cost": 0}
            for method in (
                "frozen_source_conformal", "periodic_target_conformal",
                "source_referenced_dtcr", "regime_adaptive_dtcr",
            )
        }
        epoch_records = []
        ra_triggers = ra_false = ra_detected = ra_first = static_triggers = 0

        for position, (domain, split) in enumerate(zip(SCHEDULE, windows)):
            domain_scores, domain_labels = scores[domain], labels[domain]
            monitor_scores = domain_scores[split["monitor"]]
            calibration_scores = domain_scores[split["calibration"]]
            calibration_labels = domain_labels[split["calibration"]]
            benign_calibration = calibration_scores[calibration_labels == 0]
            test_scores = domain_scores[split["test"]]
            test_labels = domain_labels[split["test"]]

            adaptive_decision = controller.monitor(monitor_scores)
            reference_regime_before = current_regime
            changed_from_reference = domain != current_regime
            stream_boundary = domain != previous_stream_regime
            if adaptive_decision.triggered:
                ra_triggers += 1
                ra_false += int(not changed_from_reference)
                ra_detected += int(changed_from_reference)
                ra_first += int(stream_boundary)
                controller.recalibrate(monitor_scores, calibration_scores, calibration_labels)
                current_regime = domain

            static_decision = detect_score_shift(source_monitor, monitor_scores, significance)
            static_reference = benign_calibration if static_decision.triggered else source_benign
            static_triggers += int(static_decision.triggered)

            predictions = {
                "frozen_source_conformal": conformal_predict(test_scores, source_benign, ALPHA),
                "periodic_target_conformal": conformal_predict(test_scores, benign_calibration, ALPHA),
                "source_referenced_dtcr": conformal_predict(test_scores, static_reference, ALPHA),
                "regime_adaptive_dtcr": controller.predict(test_scores),
            }
            for method, prediction in predictions.items():
                accumulated[method]["pred"].extend(prediction)
                accumulated[method]["scores"].extend(test_scores)
                accumulated[method]["labels"].extend(test_labels)
            accumulated["periodic_target_conformal"]["label_cost"] += len(calibration_labels)
            accumulated["source_referenced_dtcr"]["label_cost"] += (
                len(calibration_labels) if static_decision.triggered else 0
            )
            accumulated["regime_adaptive_dtcr"]["label_cost"] += (
                len(calibration_labels) if adaptive_decision.triggered else 0
            )
            epoch_records.append({
                "position": position, "domain": domain,
                "adaptive_p_value": round(adaptive_decision.p_value, 8),
                "adaptive_triggered": adaptive_decision.triggered,
                "source_referenced_triggered": static_decision.triggered,
                "reference_regime_before_monitor": reference_regime_before,
                "stream_boundary": stream_boundary,
            })
            previous_stream_regime = domain

        methods = {}
        for method, values in accumulated.items():
            metrics = metrics_from_predictions(values["scores"], values["labels"], values["pred"])
            methods[method] = {**rounded(metrics), "labels": values["label_cost"]}
        runs.append({
            "seed": seed, "methods": methods, "ra_triggers": ra_triggers,
            "ra_false_triggers": ra_false, "ra_detected_transitions": ra_detected,
            "ra_first_window_detections": ra_first, "static_triggers": static_triggers,
            "epochs": epoch_records,
        })
    return {"runs": runs, "aggregate": aggregate_seed_runs(runs)}


def main():
    paths = {"source_synthetic": DATA / "stage1_eval.jsonl"}
    paths.update({p.stem: p for p in (DATA / "external_shift").glob("*.jsonl")})
    rows = {name: load_jsonl(path) for name, path in paths.items()}
    labels = {
        name: np.asarray([1 if row["label"] in (1, "adversarial") else 0 for row in values])
        for name, values in rows.items()
    }
    cache = dict(np.load(RESULTS / "shift_scores.npz"))
    model_fingerprints = detector_fingerprints(
        json.loads((DATA / "reference_bank.json").read_text())
    )
    for detector in DETECTORS:
        for domain, path in paths.items():
            if not cache_entry_is_current(
                cache, detector, domain, file_sha256(path), len(rows[domain]),
                model_fingerprints[detector],
            ):
                raise RuntimeError(f"stale score cache for {detector}/{domain}; run run_shift_recalibration.py")
    scores = {
        detector: {domain: cache[f"{detector}__{domain}"] for domain in paths}
        for detector in DETECTORS
    }
    output = {
        "protocol": {
            "seeds": SEEDS, "alpha": ALPHA,
            "false_trigger_budget": FALSE_TRIGGER_BUDGET,
            "horizon": len(SCHEDULE),
            "per_window_significance": FALSE_TRIGGER_BUDGET / len(SCHEDULE),
            "schedule": SCHEDULE,
            "source_reference_per_label": SOURCE_REFERENCE_PER_LABEL,
            "per_window_per_label": PER_LABEL,
            "transition_count": 5,
            "note": "Constructed prequential regime stream; rows are disjoint within each seed, but datasets lack timestamps and seeds overlap. Monitor labels are hidden from controllers and used only for offline allocation/evaluation.",
        },
        "results": {},
    }
    for detector in DETECTORS:
        print(f"Evaluating regime stream for {detector}", flush=True)
        output["results"][detector] = evaluate_detector(detector, scores[detector], labels)
    path = RESULTS / "regime_stream_results.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
