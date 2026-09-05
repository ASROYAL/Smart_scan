"""Statistical activity predictors.

These estimate the probability a band is active at a future time from its
observation history alone (no ground truth). Two complementary signals:
  - EWMA of recent activity (captures overall activity level)
  - periodicity phase (captures recurring bursts)
"""

from __future__ import annotations

import numpy as np

from smartscan.core.models import BandState
from smartscan.prediction.base import BasePredictor
from smartscan.prediction.periodicity import estimate_periodicity


class EWMAPredictor(BasePredictor):
    """Predicts activity probability as an exponentially-weighted moving average.

    Ignores future_time (memoryless): the best estimate of near-future activity is
    the recent activity level.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._ewma: dict[int, float] = {}

    def predict_activity_probability(
        self, band_state: BandState, future_time: float,
    ) -> float:
        return self._ewma.get(band_state.band_id, band_state.rolling_activity_prob)

    def update(self, band_state: BandState, detected: bool, timestamp: float) -> None:
        prev = self._ewma.get(band_state.band_id, 0.5)
        target = 1.0 if detected else 0.0
        self._ewma[band_state.band_id] = self.alpha * target + (1 - self.alpha) * prev

    def reset(self) -> None:
        self._ewma.clear()


class PeriodicityAwarePredictor(BasePredictor):
    """Combines EWMA baseline with a periodicity-phase boost.

    If a band has an estimated period and the future time falls near a predicted
    recurrence, the probability is boosted toward 1; otherwise it decays toward the
    EWMA baseline.
    """

    def __init__(self, alpha: float = 0.1, phase_tolerance: float = 0.15) -> None:
        self.alpha = alpha
        self.phase_tolerance = phase_tolerance
        self._ewma: dict[int, float] = {}
        self._detection_times: dict[int, list[float]] = {}

    def predict_activity_probability(
        self, band_state: BandState, future_time: float,
    ) -> float:
        bid = band_state.band_id
        baseline = self._ewma.get(bid, band_state.rolling_activity_prob)

        times = self._detection_times.get(bid, [])
        est = estimate_periodicity(times, future_time)
        if est.estimated_period is None or est.predicted_next_activity_time is None:
            return baseline

        # Distance (in cycles) from the predicted recurrence
        period = est.estimated_period
        phase_dist = abs(future_time - est.predicted_next_activity_time) / period
        phase_dist = min(phase_dist, 1.0 - phase_dist)  # circular distance

        if phase_dist <= self.phase_tolerance:
            # Near a predicted recurrence → boost by confidence
            return float(np.clip(baseline + est.confidence * (1 - baseline), 0.0, 1.0))
        return baseline

    def update(self, band_state: BandState, detected: bool, timestamp: float) -> None:
        bid = band_state.band_id
        prev = self._ewma.get(bid, 0.5)
        target = 1.0 if detected else 0.0
        self._ewma[bid] = self.alpha * target + (1 - self.alpha) * prev
        if detected:
            self._detection_times.setdefault(bid, []).append(timestamp)

    def reset(self) -> None:
        self._ewma.clear()
        self._detection_times.clear()
