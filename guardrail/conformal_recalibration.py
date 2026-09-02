"""Drift-triggered negative-class conformal recalibration.

The detector is treated as a frozen scoring function: larger scores mean
"more hazardous."  A source benign reference bank is used while the score
distribution appears stable.  After an independently measured score shift,
a small labelled target calibration window replaces that bank.  Decisions
are made from one-sided conformal p-values, so the false-positive statement
depends on exchangeability of benign calibration and future benign scores,
not on score calibration or a parametric model.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DriftDecision:
    statistic: float
    p_value: float
    triggered: bool


class RegimeAdaptiveDTCR:
    """Stateful DTCR with horizon-wide false-trigger error spending.

    The controller tests at ``false_trigger_budget / horizon`` in every
    monitor window. After a trigger, the caller supplies the independently
    labelled calibration window and the controller refreshes both its
    unlabeled regime reference and benign conformal reference. The monitoring
    budget is not reset, so repeated use cannot silently multiply the declared
    horizon-wide false-trigger budget.
    """

    def __init__(
        self,
        monitor_reference_scores,
        benign_calibration_scores,
        horizon,
        false_trigger_budget=0.05,
        alpha=0.05,
    ):
        self.monitor_reference_scores = np.asarray(monitor_reference_scores, dtype=float)
        self.benign_calibration_scores = np.asarray(benign_calibration_scores, dtype=float)
        if self.monitor_reference_scores.size == 0:
            raise ValueError("monitor reference must be non-empty")
        if self.benign_calibration_scores.size == 0:
            raise ValueError("benign calibration reference must be non-empty")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if not 0.0 < false_trigger_budget < 1.0:
            raise ValueError("false-trigger budget must lie in (0, 1)")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.horizon = horizon
        self.false_trigger_budget = false_trigger_budget
        self.alpha = alpha
        self.window_significance = false_trigger_budget / horizon
        self.windows_seen = 0
        self._awaiting_recalibration = False

    def monitor(self, scores):
        """Test one label-hidden window against the currently deployed regime."""
        if self._awaiting_recalibration:
            raise RuntimeError("recalibrate the triggered window before monitoring again")
        if self.windows_seen >= self.horizon:
            raise RuntimeError("declared monitoring horizon is exhausted")
        self.windows_seen += 1
        decision = detect_score_shift(
            self.monitor_reference_scores, scores, self.window_significance
        )
        self._awaiting_recalibration = decision.triggered
        return decision

    def recalibrate(self, monitor_scores, calibration_scores, calibration_labels):
        """Refresh regime and benign references after a monitor trigger."""
        if not self._awaiting_recalibration:
            raise RuntimeError("recalibration is permitted only after a trigger")
        monitor_scores = np.asarray(monitor_scores, dtype=float)
        calibration_scores = np.asarray(calibration_scores, dtype=float)
        labels = np.asarray(calibration_labels, dtype=int)
        if monitor_scores.size == 0:
            raise ValueError("monitor scores must be non-empty")
        if calibration_scores.size != labels.size or labels.size == 0:
            raise ValueError("calibration scores and labels must be non-empty and aligned")
        benign = calibration_scores[labels == 0]
        if benign.size == 0:
            raise ValueError("recalibration requires labelled benign scores")
        self.monitor_reference_scores = monitor_scores.copy()
        self.benign_calibration_scores = benign.copy()
        self._awaiting_recalibration = False

    def predict(self, scores):
        if self._awaiting_recalibration:
            raise RuntimeError("triggered window must be recalibrated before prediction")
        return conformal_predict(scores, self.benign_calibration_scores, self.alpha)


def detect_score_shift(
    source_scores: Iterable[float],
    target_monitor_scores: Iterable[float],
    significance: float = 0.01,
) -> DriftDecision:
    """Two-sample KS trigger computed without labels."""
    source = np.asarray(list(source_scores), dtype=float)
    target = np.asarray(list(target_monitor_scores), dtype=float)
    if source.size == 0 or target.size == 0:
        raise ValueError("source and target monitor scores must be non-empty")
    if not 0.0 < significance < 1.0:
        raise ValueError("significance must lie in (0, 1)")
    result = stats.ks_2samp(source, target, alternative="two-sided", method="auto")
    return DriftDecision(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        triggered=bool(result.pvalue < significance),
    )


def negative_conformal_pvalues(
    scores: Iterable[float], benign_calibration_scores: Iterable[float]
) -> np.ndarray:
    """Return upper-tail conformal p-values for hazardousness scores.

    With ``n`` exchangeable benign calibration scores, the p-value for a new
    score ``s`` is ``(1 + #{calibration score >= s}) / (n + 1)``.  Ties are
    handled conservatively.  Consequently, flagging when ``p <= alpha`` has
    marginal benign false-positive probability at most ``alpha``.
    """
    test = np.asarray(list(scores), dtype=float)
    benign = np.asarray(list(benign_calibration_scores), dtype=float)
    if benign.size == 0:
        raise ValueError("at least one labelled benign calibration score is required")
    return (1.0 + (benign[None, :] >= test[:, None]).sum(axis=1)) / (benign.size + 1.0)


def conformal_predict(
    scores: Iterable[float],
    benign_calibration_scores: Iterable[float],
    alpha: float = 0.05,
) -> np.ndarray:
    """Flag scores whose benign conformal p-value is at most ``alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return (negative_conformal_pvalues(scores, benign_calibration_scores) <= alpha).astype(int)


def empirical_fpr_threshold(
    scores: Iterable[float], labels: Iterable[int], target_fpr: float = 0.05
) -> float:
    """Most permissive observed threshold satisfying an empirical FPR budget."""
    values = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    if values.size != y.size or values.size == 0:
        raise ValueError("scores and labels must be non-empty and equally sized")
    benign = values[y == 0]
    if benign.size == 0:
        raise ValueError("threshold selection requires benign calibration examples")
    unique = np.unique(values)
    # Predictions use ``score >= threshold``.  Considering observed values
    # alone cannot represent the boundary immediately above a tied benign
    # score and can therefore jump all the way to the smallest positive score.
    # Include both sides of every observed tie.
    candidates = np.unique(np.r_[np.inf, unique, np.nextafter(unique, np.inf)])
    feasible = [t for t in candidates if np.mean(benign >= t) <= target_fpr]
    return float(min(feasible))


def stratified_three_way_indices(
    labels: Iterable[int],
    monitor_fraction: float = 0.2,
    calibration_fraction: float = 0.4,
    seed: int = 0,
):
    """Return disjoint monitor/calibration/test indices, stratified by label."""
    y = np.asarray(list(labels), dtype=int)
    if monitor_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("monitor and calibration fractions must be positive")
    if monitor_fraction + calibration_fraction >= 1:
        raise ValueError("monitor and calibration fractions must leave a test split")
    rng = np.random.RandomState(seed)
    monitor, calibration, test = [], [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_monitor = max(1, int(np.floor(len(idx) * monitor_fraction)))
        n_cal = max(1, int(np.floor(len(idx) * calibration_fraction)))
        if n_monitor + n_cal >= len(idx):
            n_cal = max(1, len(idx) - n_monitor - 1)
        monitor.extend(idx[:n_monitor])
        calibration.extend(idx[n_monitor:n_monitor + n_cal])
        test.extend(idx[n_monitor + n_cal:])
    rng.shuffle(monitor)
    rng.shuffle(calibration)
    rng.shuffle(test)
    return np.asarray(monitor), np.asarray(calibration), np.asarray(test)
