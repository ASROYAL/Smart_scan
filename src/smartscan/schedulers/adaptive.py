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
from smartscan.prediction.temporal_belief import TemporalBeliefEnsemble
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
        self._belief = TemporalBeliefEnsemble(num_bands)
        self._decision_count = 0
        self._last_frequency: float | None = None

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
        best_components: dict[str, float] = {}

        hard_due = [
            state for state in band_states
            if state.time_since_scan(current_time) >= self._cfg.max_revisit_gap
        ]
        explore_every = max(1, round(1.0 / max(self._cfg.min_exploration_fraction, 1e-9)))
        forced_exploration = bool(hard_due) or (
            self._cfg.min_exploration_fraction > 0 and self._decision_count % explore_every == 0
        )
        candidates = hard_due if hard_due else band_states
        if hard_due:
            oldest_age = max(state.time_since_scan(current_time) for state in hard_due)
            candidates = [
                state for state in hard_due
                if (np.isinf(oldest_age) and np.isinf(state.time_since_scan(current_time)))
                or np.isclose(state.time_since_scan(current_time), oldest_age)
            ]
        if forced_exploration and not hard_due:
            minimum_observations = min(state.observation_count for state in candidates)
            candidates = [state for state in candidates if state.observation_count == minimum_observations]

        span = max(state.freq_end for state in band_states) - min(state.freq_start for state in band_states)

        for state in candidates:
            x = _context_vector(state, current_time, self._time_scale)
            predicted = float(theta @ x)
            uncertainty = self._alpha * float(np.sqrt(x @ A_inv @ x))
            temporal = self._belief.predict(state, current_time)
            novelty = state.novelty_score + self._belief.change_score(state.band_id)
            timing = self._revisit_bonus(state, current_time)
            age = state.time_since_scan(current_time)
            coverage = 1.0 if np.isinf(age) else min(age / self._cfg.max_revisit_gap, 1.0)
            tuning = 0.0 if self._last_frequency is None else (
                abs(state.center_frequency - self._last_frequency) / max(span, 1.0)
            )
            components = {
                "learned": predicted + uncertainty,
                "activity": temporal,
                "novelty": novelty,
                "threat": state.threat_score,
                "timing": timing,
                "coverage": coverage,
                "tuning_cost": tuning,
            }
            score = (
                components["learned"]
                + self._cfg.utility_activity_weight * temporal
                + self._cfg.utility_novelty_weight * novelty
                + self._cfg.utility_threat_weight * state.threat_score
                + self._cfg.utility_timing_weight * timing
                + self._cfg.utility_coverage_weight * coverage
                - self._cfg.utility_tuning_cost_weight * tuning
            )

            if score > best_score:
                best_score = score
                best_id = state.band_id
                best_context = x
                best_components = components

        state = band_states[best_id]
        if best_context is not None:  # always true when band_states is non-empty
            self._last_context[best_id] = best_context

        belief = self._belief.predict(state, current_time)
        dwell_fraction = float(np.clip(0.55 * belief + 0.45 * state.threat_score, 0.0, 1.0))
        dwell = self._rx_config.min_dwell_time + dwell_fraction * (
            self._rx_config.max_dwell_time - self._rx_config.min_dwell_time
        )
        if forced_exploration:
            dwell = self._rx_config.min_dwell_time
        revisit = current_time + (
            state.estimated_period if state.estimated_period else
            self._cfg.max_revisit_gap * max(0.2, 1.0 - belief)
        )
        self._decision_count += 1
        self._last_frequency = state.center_frequency
        return ScanDecision(
            band_id=best_id,
            center_frequency=state.center_frequency,
            bandwidth=min(state.bandwidth, self._rx_config.instantaneous_bandwidth),
            dwell_time=dwell,
            priority_score=float(best_score),
            reason=("hard revisit constraint" if hard_due else
                    "exploration budget" if forced_exploration else "multi-objective LinUCB"),
            timestamp=current_time,
            recommended_revisit_time=revisit,
            utility_components=best_components,
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
        self._belief.update(observation)

    def get_theta(self) -> np.ndarray:
        """Current learned weight vector (for inspection/debugging)."""
        return np.linalg.inv(self._A) @ self._b

    def reset(self) -> None:
        d = self.N_FEATURES
        self._A = np.eye(d)
        self._b = np.zeros(d)
        self._last_context.clear()
        self._belief = TemporalBeliefEnsemble(self._num_bands)
        self._decision_count = 0
        self._last_frequency = None
