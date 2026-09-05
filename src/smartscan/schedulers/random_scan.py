"""Random scheduler — selects the next band uniformly at random."""

from __future__ import annotations

import numpy as np

from smartscan.core.config import ReceiverConfig
from smartscan.core.models import BandObservation, BandState, ScanDecision
from smartscan.schedulers.base import BaseScheduler


class RandomScanScheduler(BaseScheduler):
    """Chooses the next band uniformly at random."""

    def __init__(self, receiver_config: ReceiverConfig, seed: int = 0) -> None:
        self._rx_config = receiver_config
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    @property
    def name(self) -> str:
        return "random"

    def select_band(
        self, band_states: list[BandState], current_time: float,
    ) -> ScanDecision:
        idx = int(self._rng.integers(0, len(band_states)))
        state = band_states[idx]

        return ScanDecision(
            band_id=state.band_id,
            center_frequency=state.center_frequency,
            bandwidth=min(state.bandwidth, self._rx_config.instantaneous_bandwidth),
            dwell_time=self._rx_config.dwell_time,
            priority_score=0.0,
            reason="random selection",
            timestamp=current_time,
        )

    def update(self, decision: ScanDecision, observation: BandObservation) -> None:
        pass

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
