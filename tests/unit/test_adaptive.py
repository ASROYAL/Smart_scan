"""Tests for the adaptive contextual-bandit scheduler."""

import numpy as np

from smartscan.core.config import ReceiverConfig, SchedulerConfig
from smartscan.core.models import BandObservation, BandState
from smartscan.schedulers.adaptive import AdaptiveScheduler, _context_vector


def make_states(n=5):
    return [
        BandState(band_id=i, freq_start=i * 20e6, freq_end=(i + 1) * 20e6)
        for i in range(n)
    ]


def make_obs(band_id, detected, confidence=0.8):
    return BandObservation(
        band_id=band_id, timestamp=0.0, detected=detected, confidence=confidence,
        peak_power_db=-50.0, avg_power_db=-60.0,
        noise_floor_db=-100.0, estimated_snr_db=20.0,
    )


def rx_cfg():
    return ReceiverConfig(instantaneous_bandwidth=20e6, dwell_time=0.01)


class TestContextVector:
    def test_length(self):
        state = BandState(band_id=0, freq_start=0, freq_end=20e6)
        x = _context_vector(state, 0.0, time_scale=10.0)
        assert len(x) == AdaptiveScheduler.N_FEATURES

    def test_bias_term(self):
        state = BandState(band_id=0, freq_start=0, freq_end=20e6)
        x = _context_vector(state, 0.0, time_scale=10.0)
        assert x[0] == 1.0

    def test_never_scanned_time_features(self):
        state = BandState(band_id=0, freq_start=0, freq_end=20e6)
        x = _context_vector(state, 5.0, time_scale=10.0)
        # inf time_since_scan → normalized to 1.0
        assert x[3] == 1.0
        assert x[4] == 1.0

    def test_no_ground_truth_in_context(self):
        """Context is built only from BandState fields (observation-derived)."""
        state = BandState(
            band_id=0, freq_start=0, freq_end=20e6,
            rolling_activity_prob=0.7, activity_ratio=0.6, confidence=0.5,
        )
        x = _context_vector(state, 1.0, time_scale=10.0)
        assert x[1] == 0.7  # rolling_activity_prob
        assert x[5] == 0.5  # confidence


class TestAdaptiveScheduler:
    def test_enforces_revisit_constraint_and_emits_complete_action(self):
        cfg = SchedulerConfig(max_revisit_gap=0.5, min_exploration_fraction=0.0)
        sched = AdaptiveScheduler(rx_cfg(), cfg, num_bands=3)
        states = make_states(3)
        for state in states:
            state.last_scan_time = 0.9
        states[2].last_scan_time = 0.0
        decision = sched.select_band(states, 1.0)
        assert decision.band_id == 2
        assert decision.reason == "hard revisit constraint"
        assert rx_cfg().min_dwell_time <= decision.dwell_time <= rx_cfg().max_dwell_time
        assert decision.recommended_revisit_time is not None
        assert decision.utility_components["coverage"] == 1.0

    def test_selects_valid_bands(self):
        sched = AdaptiveScheduler(rx_cfg(), SchedulerConfig(), num_bands=5)
        states = make_states(5)
        for t in range(20):
            d = sched.select_band(states, float(t))
            assert 0 <= d.band_id < 5
            sched.update(d, make_obs(d.band_id, False))

    def test_name(self):
        assert AdaptiveScheduler(rx_cfg(), SchedulerConfig(), 5).name == "adaptive"

    def test_learns_from_rewards(self):
        """After training, theta should reflect that activity features predict reward."""
        sched = AdaptiveScheduler(rx_cfg(), SchedulerConfig(), num_bands=4)
        states = make_states(4)

        # Simulate: bands with high rolling_activity_prob yield detections.
        # activity_ratio is a derived property, so set the underlying counts.
        for t in range(80):
            states[0].rolling_activity_prob = 0.9
            states[0].hit_count = 9
            states[0].miss_count = 1
            states[0].observation_count = 10
            states[0].confidence = 0.5
            d = sched.select_band(states, float(t))
            # Reward correlates with the activity feature of the chosen band
            detected = states[d.band_id].rolling_activity_prob > 0.5
            obs = make_obs(d.band_id, detected)
            states[d.band_id].last_scan_time = float(t)
            sched.update(d, obs, reward=float(detected))

        theta = sched.get_theta()
        assert len(theta) == AdaptiveScheduler.N_FEATURES
        # The activity-prob weight (index 1) should be positive after learning
        assert theta[1] > 0

    def test_online_update_changes_model(self):
        sched = AdaptiveScheduler(rx_cfg(), SchedulerConfig(), num_bands=3)
        states = make_states(3)
        theta_before = sched.get_theta().copy()
        d = sched.select_band(states, 0.0)
        sched.update(d, make_obs(d.band_id, True), reward=1.0)
        theta_after = sched.get_theta()
        assert not np.allclose(theta_before, theta_after)

    def test_reset(self):
        sched = AdaptiveScheduler(rx_cfg(), SchedulerConfig(), num_bands=3)
        states = make_states(3)
        d = sched.select_band(states, 0.0)
        sched.update(d, make_obs(d.band_id, True), reward=1.0)
        sched.reset()
        assert np.allclose(sched.get_theta(), np.zeros(AdaptiveScheduler.N_FEATURES))

    def test_no_environment_reference(self):
        """Adaptive scheduler must not hold ground-truth access."""
        sched = AdaptiveScheduler(rx_cfg(), SchedulerConfig(), num_bands=3)
        for attr_name in vars(sched):
            attr = getattr(sched, attr_name)
            assert type(attr).__name__ != "RFEnvironment"
