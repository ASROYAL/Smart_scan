"""Adaptive contextual-bandit scheduler (shared LinUCB).

Treats each band as an arm whose context is a feature vector derived purely from
its observation history (BandState). A single shared linear model maps context to
expected reward, so the scheduler generalizes across bands with similar features
and can make informed guesses about rarely-visited bands.

LinUCB selection:  score = theta^T x + alpha * sqrt(x^T A^-1 x)
  - first term: predicted reward (exploitation)
  - second term: uncertainty bonus (exploration), shrinks as similar contexts are seen

Reward is computed from the DETECTION outcome, never ground truth. Reward weights
are configurable. A periodicity-aware revisit bonus nudges the scheduler to return
to bands just as they are predicted to become active again.
"""

from __future__ import annotations

import numpy as np

from smartscan.core.config import ReceiverConfig, SchedulerConfig
from smartscan.core.models import BandObservation, BandState, ScanDecision
from smartscan.schedulers.base import BaseScheduler


def _context_vector(state: BandState, current_time: float, time_scale: float) -> np.ndarray:
    """Build a context feature vector from a band's observation-derived state.

    All features come from BandState (detector output + history). No ground truth.
    """
    tss = state.time_since_scan(current_time)
    tsd = state.time_since_detection(current_time)

    # Normalize/clip unbounded time features
    tss_norm = 1.0 if np.isinf(tss) else float(np.clip(tss / time_scale, 0.0, 2.0))
    tsd_norm = 1.0 if np.isinf(tsd) else float(np.clip(tsd / time_scale, 0.0, 2.0))

    snr_norm = float(np.clip(state.avg_snr_db / 30.0, -1.0, 2.0))

    return np.array([
        1.0,                              # bias
        state.rolling_activity_prob,      # EWMA activity
        state.activity_ratio,             # long-run hit fraction
        tss_norm,                         # recency of scan
        tsd_norm,                         # recency of detection
        state.confidence,                 # estimate maturity
        snr_norm,                         # signal strength
    ], dtype=np.float64)


class AdaptiveScheduler(BaseScheduler):
    """Contextual bandit scheduler with online LinUCB learning."""

    N_FEATURES = 7

    def __init__(
        self,
        receiver_config: ReceiverConfig,
        scheduler_config: SchedulerConfig,
        num_bands: int,
    ) -> None:
        self._rx_config = receiver_config
        self._cfg = scheduler_config
        self._num_bands = num_bands
        self._time_scale = max(scheduler_config.starvation_threshold, 1e-6)

        d = self.N_FEATURES
        self._A = np.eye(d)          # d x d design matrix
        self._b = np.zeros(d)        # d reward-weighted context sum
        self._alpha = scheduler_config.exploration_weight

        # Track last-decision contexts for the online update
        self._last_context: dict[int, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "adaptive"

    def select_band(
        self, band_states: list[BandState], current_time: float,
    ) -> ScanDecision:
        A_inv = np.linalg.inv(self._A)
        theta = A_inv @ self._b

        best_id = -1
        best_score = -np.inf
        best_context = None

        for state in band_states:
            x = _context_vector(state, current_time, self._time_scale)
            predicted = float(theta @ x)
            uncertainty = self._alpha * float(np.sqrt(x @ A_inv @ x))
            score = predicted + uncertainty

            # Periodicity-aware revisit bonus: if the band is predicted to be
            # active around now, nudge its score up.
            score += self._revisit_bonus(state, current_time)

            if score > best_score:
                best_score = score
                best_id = state.band_id
                best_context = x

        state = band_states[best_id]
        if best_context is not None:  # always true when band_states is non-empty
            self._last_context[best_id] = best_context

        return ScanDecision(
            band_id=best_id,
            center_frequency=state.center_frequency,
            bandwidth=min(state.bandwidth, self._rx_config.instantaneous_bandwidth),
            dwell_time=self._rx_config.dwell_time,
            priority_score=float(best_score),
            reason="adaptive LinUCB",
            timestamp=current_time,
        )

    def _revisit_bonus(self, state: BandState, current_time: float) -> float:
        """Bonus when a periodic band is predicted to become active soon."""
        if state.estimated_period is None or state.last_detection_time < 0:
            return 0.0
        period = state.estimated_period
        if period <= 0:
            return 0.0
        # Phase within the predicted cycle since last detection
        elapsed = current_time - state.last_detection_time
        phase = (elapsed % period) / period
        # Bonus peaks when phase is near 0 or 1 (i.e., near a predicted recurrence)
        proximity = min(phase, 1.0 - phase)  # 0 at recurrence, 0.5 mid-cycle
        return self._cfg.recency_weight * (0.5 - proximity)

    def update(self, decision: ScanDecision, observation: BandObservation,
               reward: float = 0.0) -> None:
        band_id = decision.band_id
        x = self._last_context.get(band_id)
        if x is None:
            return
        # LinUCB online update against the shared reward
        self._A += np.outer(x, x)
        self._b += reward * x

    def get_theta(self) -> np.ndarray:
        """Current learned weight vector (for inspection/debugging)."""
        return np.linalg.inv(self._A) @ self._b

    def reset(self) -> None:
        d = self.N_FEATURES
        self._A = np.eye(d)
        self._b = np.zeros(d)
        self._last_context.clear()
