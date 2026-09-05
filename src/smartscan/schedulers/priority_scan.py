"""Weighted-priority scheduler.

Scores each band by a weighted combination of estimated activity, recency
(time since last visit), and uncertainty. Includes anti-starvation so no band
is ignored forever.
"""

from __future__ import annotations

import numpy as np

from smartscan.core.config import ReceiverConfig, SchedulerConfig
from smartscan.core.models import BandObservation, BandState, ScanDecision
from smartscan.schedulers.base import BaseScheduler


class PriorityScanScheduler(BaseScheduler):
    """Selects the band with the highest priority score.

    priority = activity_weight * activity_prob
             + recency_weight * recency_factor
             + uncertainty_weight * uncertainty_factor
    """

    def __init__(
        self,
        receiver_config: ReceiverConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self._rx_config = receiver_config
        self._cfg = scheduler_config

    @property
    def name(self) -> str:
        return "priority"

    def _score(self, state: BandState, current_time: float) -> float:
        # Activity component: rolling probability of activity
        activity = state.rolling_activity_prob

        # Recency: longer since last scan → higher priority (normalized by starvation
        # threshold). Never-scanned bands get a recency strictly above the saturation
        # cap so unexplored bands always outrank already-visited ones on ties.
        tss = state.time_since_scan(current_time)
        if np.isinf(tss):
            recency = 2.0  # never scanned → exploration priority beats any visited band
        else:
            recency = float(np.clip(tss / self._cfg.starvation_threshold, 0.0, 1.0))

        # Uncertainty: low confidence → high priority to gather more info
        uncertainty = 1.0 - state.confidence

        return (
            self._cfg.exploration_weight * activity
            + self._cfg.recency_weight * recency
            + self._cfg.uncertainty_weight * uncertainty
        )

    def select_band(
        self, band_states: list[BandState], current_time: float,
    ) -> ScanDecision:
        # Anti-starvation: prioritize any band whose time-since-scan exceeds the
        # threshold. Never-visited bands (time_since_scan == inf) are the MOST
        # starved and must be included, not excluded.
        starved = [
            s for s in band_states
            if s.time_since_scan(current_time) > self._cfg.starvation_threshold
        ]

        if starved:
            # Pick the most-starved band (never-visited bands, with inf, come first)
            state = max(starved, key=lambda s: s.time_since_scan(current_time))
            reason = "anti-starvation"
            score = self._score(state, current_time)
        else:
            scores = [(self._score(s, current_time), s) for s in band_states]
            score, state = max(scores, key=lambda x: x[0])
            reason = "highest priority"

        return ScanDecision(
            band_id=state.band_id,
            center_frequency=state.center_frequency,
            bandwidth=min(state.bandwidth, self._rx_config.instantaneous_bandwidth),
            dwell_time=self._rx_config.dwell_time,
            priority_score=score,
            reason=reason,
            timestamp=current_time,
        )

    def update(self, decision: ScanDecision, observation: BandObservation) -> None:
        pass  # State updates happen in the SpectrumStateManager

    def reset(self) -> None:
        pass
