"""Ground-truth-free online temporal belief ensemble for spectrum activity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartscan.core.models import BandObservation, BandState


@dataclass
class _Belief:
    alpha: float = 1.0
    beta: float = 1.0
    hmm_active: float = 0.5
    last_label: bool = False
    change_score: float = 0.0


class TemporalBeliefEnsemble:
    """Blend Bayesian rate, two-state persistence, periodic phase and change evidence."""

    def __init__(self, num_bands: int) -> None:
        self._beliefs = [_Belief() for _ in range(num_bands)]

    def predict(self, state: BandState, at_time: float) -> float:
        belief = self._beliefs[state.band_id]
        beta_mean = belief.alpha / (belief.alpha + belief.beta)
        elapsed = state.time_since_scan(at_time)
        decay = 0.0 if np.isinf(elapsed) else float(np.exp(-elapsed / 0.5))
        hmm = 0.5 + (belief.hmm_active - 0.5) * decay
        periodic = 0.0
        if state.estimated_period and state.last_detection_time >= 0:
            phase = ((at_time - state.last_detection_time) % state.estimated_period) / state.estimated_period
            proximity = min(phase, 1.0 - phase)
            periodic = float(np.exp(-((proximity / 0.16) ** 2)))
        score = 0.35 * beta_mean + 0.35 * hmm + 0.2 * state.rolling_activity_prob + 0.1 * periodic
        return float(np.clip(score, 0.0, 1.0))

    def update(self, observation: BandObservation) -> None:
        belief = self._beliefs[observation.band_id]
        weight = 0.5 + 0.5 * observation.confidence
        if observation.detected:
            belief.alpha += weight
        else:
            belief.beta += weight
        transition = 0.82 if observation.detected == belief.last_label else 0.38
        target = 1.0 if observation.detected else 0.0
        belief.hmm_active = transition * belief.hmm_active + (1.0 - transition) * target
        surprise = abs(target - belief.hmm_active)
        belief.change_score = 0.85 * belief.change_score + 0.15 * surprise
        belief.last_label = observation.detected

    def change_score(self, band_id: int) -> float:
        return self._beliefs[band_id].change_score

