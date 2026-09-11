"""Online machine-learning activity predictor.

Predicts the probability that a band will be active, learning online from
observed (features -> detected?) pairs with a logistic SGD classifier. This is a
stronger "when will it be active" model than a fixed EWMA: it weighs recency,
hit ratio, SNR and periodicity phase together and adapts as evidence arrives.
"""

from __future__ import annotations

import numpy as np

from smartscan.core.models import BandState
from smartscan.prediction.base import BasePredictor


def _features(state: BandState, at_time: float) -> np.ndarray:
    tss = state.time_since_scan(at_time)
    tsd = state.time_since_detection(at_time)
    tss = 100.0 if np.isinf(tss) else tss
    tsd = 100.0 if np.isinf(tsd) else tsd
    # periodicity phase: 0 near a predicted recurrence, 0.5 mid-cycle
    if state.estimated_period and state.last_detection_time >= 0 and state.estimated_period > 0:
        phase = ((at_time - state.last_detection_time) % state.estimated_period) / state.estimated_period
        phase_prox = min(phase, 1.0 - phase)
    else:
        phase_prox = 0.5
    return np.array([
        state.rolling_activity_prob,
        state.activity_ratio,
        np.tanh(tss / 10.0),
        np.tanh(tsd / 10.0),
        state.avg_snr_db / 30.0,
        state.confidence,
        phase_prox,
    ], dtype=np.float64)


class MLActivityPredictor(BasePredictor):
    """Logistic-SGD online predictor of band activity probability."""

    def __init__(self) -> None:
        from sklearn.linear_model import SGDClassifier
        self._clf = SGDClassifier(loss="log_loss", alpha=1e-3, learning_rate="optimal")
        self._trained_classes: set[int] = set()
        self._n_updates = 0

    def predict_activity_probability(self, band_state: BandState, future_time: float) -> float:
        # cold start / single-class: fall back to the band's rolling estimate
        if len(self._trained_classes) < 2:
            return float(band_state.rolling_activity_prob)
        x = _features(band_state, future_time).reshape(1, -1)
        try:
            proba = self._clf.predict_proba(x)[0]
            # class order follows self._clf.classes_
            classes = list(self._clf.classes_)  # pyright: ignore[reportAttributeAccessIssue]  # set by fit()
            return float(proba[classes.index(1)])
        except Exception:  # noqa: BLE001 - unfitted/degenerate model: fall back to the EWMA belief
            return float(band_state.rolling_activity_prob)

    def update(self, band_state: BandState, detected: bool, timestamp: float) -> None:
        x = _features(band_state, timestamp).reshape(1, -1)
        y = np.array([1 if detected else 0])
        self._clf.partial_fit(x, y, classes=np.array([0, 1]))
        self._trained_classes.add(int(y[0]))
        self._n_updates += 1

    def reset(self) -> None:
        self.__init__()
