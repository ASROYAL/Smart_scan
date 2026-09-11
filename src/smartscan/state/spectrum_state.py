"""Spectrum state manager — maintains BandState and history for all bands.

This is the scheduler's ONLY view of the world. It is built purely from
observations (detector output), never from ground truth.
"""

from __future__ import annotations

import numpy as np

from smartscan.core.models import BandObservation, BandState
from smartscan.prediction.periodicity import estimate_periodicity
from smartscan.state.band_history import BandHistory


class SpectrumStateManager:
    """Manages state for a set of frequency bands partitioned across the spectrum."""

    def __init__(
        self,
        num_bands: int,
        total_bandwidth: float,
        center_frequency: float,
        ewma_alpha: float = 0.1,
        history_window: int = 20,
        episode_gap: float = 0.05,
    ) -> None:
        self.num_bands = num_bands
        self.total_bandwidth = total_bandwidth
        self.center_frequency = center_frequency
        self.ewma_alpha = ewma_alpha
        self.history_window = history_window
        self.episode_gap = episode_gap

        self._band_width = total_bandwidth / num_bands
        self._start_freq = center_frequency - total_bandwidth / 2

        self._states: dict[int, BandState] = {}
        self._histories: dict[int, BandHistory] = {}

        for band_id in range(num_bands):
            fstart = self._start_freq + band_id * self._band_width
            fend = fstart + self._band_width
            self._states[band_id] = BandState(
                band_id=band_id, freq_start=fstart, freq_end=fend,
            )
            self._histories[band_id] = BandHistory(band_id=band_id)

    @property
    def band_width(self) -> float:
        return self._band_width

    def get_state(self, band_id: int) -> BandState:
        return self._states[band_id]

    def get_history(self, band_id: int) -> BandHistory:
        return self._histories[band_id]

    def all_states(self) -> list[BandState]:
        return [self._states[i] for i in range(self.num_bands)]

    def band_id_for_frequency(self, frequency: float) -> int:
        """Map a frequency to its band id, clamped to valid range."""
        idx = int((frequency - self._start_freq) / self._band_width)
        return max(0, min(self.num_bands - 1, idx))

    def bands_in_window(self, center_frequency: float, bandwidth: float) -> list[int]:
        """Return band ids that overlap an observation window."""
        wstart = center_frequency - bandwidth / 2
        wend = center_frequency + bandwidth / 2
        ids = []
        for bid, state in self._states.items():
            if state.freq_end > wstart and state.freq_start < wend:
                ids.append(bid)
        return sorted(ids)

    def update(self, band_id: int, obs: BandObservation) -> None:
        """Update a band's state and history with a new observation."""
        state = self._states[band_id]
        history = self._histories[band_id]

        history.add(obs)

        state.observation_count += 1
        state.last_scan_time = obs.timestamp
        if obs.detected:
            state.hit_count += 1
            state.last_detection_time = obs.timestamp
        else:
            state.miss_count += 1

        # EWMA of activity
        target = 1.0 if obs.detected else 0.0
        state.rolling_activity_prob = (
            self.ewma_alpha * target + (1 - self.ewma_alpha) * state.rolling_activity_prob
        )

        # Running averages of power and SNR
        n = state.observation_count
        state.avg_power_db += (obs.avg_power_db - state.avg_power_db) / n
        state.avg_snr_db += (obs.estimated_snr_db - state.avg_snr_db) / n

        # Confidence grows with observation count (more data → more confident estimate)
        state.confidence = float(np.clip(n / (n + 10), 0.0, 1.0))

        # Better predictions: estimate the recurrence period from detection history
        # so the scheduler can time revisits and the evaluator can score intercept
        # time error. Only recompute when a fresh detection arrives.
        if obs.detected and len(history.detection_times) >= 4:
            est = estimate_periodicity(
                history.detection_times, obs.timestamp, episode_gap=self.episode_gap)
            if est.estimated_period is not None and est.confidence >= 0.3:
                state.estimated_period = est.estimated_period

    def reset(self) -> None:
        for band_id in range(self.num_bands):
            fstart = self._start_freq + band_id * self._band_width
            fend = fstart + self._band_width
            self._states[band_id] = BandState(
                band_id=band_id, freq_start=fstart, freq_end=fend,
            )
            self._histories[band_id] = BandHistory(band_id=band_id)
