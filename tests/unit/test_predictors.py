"""Tests for statistical activity predictors."""

import pytest

from smartscan.core.models import BandState
from smartscan.prediction.statistical import EWMAPredictor, PeriodicityAwarePredictor


def make_state(band_id=0):
    return BandState(band_id=band_id, freq_start=0, freq_end=20e6)


class TestEWMAPredictor:
    def test_initial_prediction_uses_state(self):
        pred = EWMAPredictor(alpha=0.2)
        state = make_state()
        state.rolling_activity_prob = 0.3
        assert pred.predict_activity_probability(state, 1.0) == 0.3

    def test_updates_toward_detections(self):
        pred = EWMAPredictor(alpha=0.5)
        state = make_state()
        for _ in range(10):
            pred.update(state, detected=True, timestamp=0.0)
        # Should climb toward 1.0
        assert pred.predict_activity_probability(state, 1.0) > 0.9

    def test_updates_toward_misses(self):
        pred = EWMAPredictor(alpha=0.5)
        state = make_state()
        for _ in range(10):
            pred.update(state, detected=False, timestamp=0.0)
        assert pred.predict_activity_probability(state, 1.0) < 0.1

    def test_probability_in_range(self):
        pred = EWMAPredictor()
        state = make_state()
        for detected in [True, False, True, True, False]:
            pred.update(state, detected=detected, timestamp=0.0)
            p = pred.predict_activity_probability(state, 1.0)
            assert 0.0 <= p <= 1.0

    def test_reset(self):
        pred = EWMAPredictor(alpha=0.5)
        state = make_state()
        pred.update(state, detected=True, timestamp=0.0)
        pred.reset()
        # Back to state default
        state.rolling_activity_prob = 0.5
        assert pred.predict_activity_probability(state, 1.0) == 0.5


class TestPeriodicityAwarePredictor:
    def test_baseline_without_period(self):
        pred = PeriodicityAwarePredictor(alpha=0.5)
        state = make_state()
        state.rolling_activity_prob = 0.4
        # No detections yet → baseline
        assert pred.predict_activity_probability(state, 1.0) == pytest.approx(0.4, abs=0.1)

    def test_boost_near_predicted_recurrence(self):
        pred = PeriodicityAwarePredictor(alpha=0.3, phase_tolerance=0.2)
        state = make_state()
        # Feed regular detections every 2 seconds
        for t in [0.0, 2.0, 4.0, 6.0, 8.0]:
            pred.update(state, detected=True, timestamp=t)

        # At t=10 (a predicted recurrence), probability should be boosted
        p_near = pred.predict_activity_probability(state, 10.0)
        # At t=11 (mid-cycle), probability should be lower
        p_mid = pred.predict_activity_probability(state, 11.0)
        assert p_near >= p_mid

    def test_probability_in_range(self):
        pred = PeriodicityAwarePredictor()
        state = make_state()
        for t in [0.0, 1.0, 2.0, 3.0, 4.0]:
            pred.update(state, detected=True, timestamp=t)
        p = pred.predict_activity_probability(state, 5.0)
        assert 0.0 <= p <= 1.0

    def test_reset(self):
        pred = PeriodicityAwarePredictor()
        state = make_state()
        for t in [0.0, 2.0, 4.0]:
            pred.update(state, detected=True, timestamp=t)
        pred.reset()
        assert pred._detection_times == {}
