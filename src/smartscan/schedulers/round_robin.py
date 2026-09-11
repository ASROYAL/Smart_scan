"""Round-robin scheduler — cycles through all bands sequentially."""

from __future__ import annotations

from smartscan.core.config import ReceiverConfig
from smartscan.core.models import BandObservation, BandState, ScanDecision
from smartscan.schedulers.base import BaseScheduler


class RoundRobinScheduler(BaseScheduler):
    """Visits bands in fixed sequential order: B0 → B1 → ... → BN → B0."""

    def __init__(self, receiver_config: ReceiverConfig) -> None:
        self._rx_config = receiver_config
        self._next_index = 0

    @property
    def name(self) -> str:
        return "round_robin"

    def select_band(
        self, band_states: list[BandState], current_time: float,
    ) -> ScanDecision:
        state = band_states[self._next_index % len(band_states)]
        self._next_index += 1

        return ScanDecision(
            band_id=state.band_id,
            center_frequency=state.center_frequency,
            bandwidth=min(state.bandwidth, self._rx_config.instantaneous_bandwidth),
            dwell_time=self._rx_config.dwell_time,
            priority_score=0.0,
            reason="round_robin sequential",
            timestamp=current_time,
        )

    def update(self, decision: ScanDecision, observation: BandObservation, reward: float = 0.0) -> None:
        pass  # Round robin is stateless w.r.t. observations

    def reset(self) -> None:
        self._next_index = 0
