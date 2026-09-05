"""Feature extraction from band state and history."""

from __future__ import annotations

import numpy as np

from smartscan.core.models import BandFeatures, BandState
from smartscan.prediction.periodicity import estimate_periodicity
from smartscan.state.band_history import BandHistory


class FeatureExtractor:
    """Extracts feature vectors for schedulers from band state and history."""

    def __init__(self, history_window: int = 20) -> None:
        self.history_window = history_window

    def extract(
        self,
        state: BandState,
        history: BandHistory,
        current_time: float,
    ) -> BandFeatures:
        """Build a feature vector for one band at the current time."""
        recent_hits = history.recent_hits(self.history_window)
        recent_misses = history.recent_misses(self.history_window)

        # Periodicity
        period_est = estimate_periodicity(history.detection_times, current_time)

        # Average active/inactive durations from detection sequence
        avg_active, avg_inactive = self._avg_durations(history)

        # Revisit interval: average time between consecutive scans
        revisit_interval = self._avg_revisit_interval(history)

        return BandFeatures(
            band_id=state.band_id,
            current_power_db=state.avg_power_db,
            estimated_snr_db=state.avg_snr_db,
            time_since_scan=state.time_since_scan(current_time),
            time_since_detection=state.time_since_detection(current_time),
            recent_hits=recent_hits,
            recent_misses=recent_misses,
            activity_ratio=state.activity_ratio,
            ewma_activity=state.rolling_activity_prob,
            avg_active_duration=avg_active,
            avg_inactive_duration=avg_inactive,
            revisit_interval=revisit_interval,
            estimated_period=period_est.estimated_period,
            confidence=state.confidence,
            observation_count=state.observation_count,
        )

    def _avg_durations(self, history: BandHistory) -> tuple[float, float]:
        """Estimate average active and inactive run lengths (in observations)."""
        seq = history.detection_sequence(self.history_window)
        if not seq:
            return 0.0, 0.0

        active_runs = []
        inactive_runs = []
        current_run = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                current_run += 1
            else:
                if seq[i - 1]:
                    active_runs.append(current_run)
                else:
                    inactive_runs.append(current_run)
                current_run = 1
        # Close the final run
        if seq[-1]:
            active_runs.append(current_run)
        else:
            inactive_runs.append(current_run)

        avg_active = float(np.mean(active_runs)) if active_runs else 0.0
        avg_inactive = float(np.mean(inactive_runs)) if inactive_runs else 0.0
        return avg_active, avg_inactive

    def _avg_revisit_interval(self, history: BandHistory) -> float:
        obs = history.recent(self.history_window)
        if len(obs) < 2:
            return 0.0
        times = [o.timestamp for o in obs]
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        return float(np.mean(intervals)) if intervals else 0.0
