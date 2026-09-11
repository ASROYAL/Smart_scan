"""Tests for feature extraction and periodicity."""

import numpy as np
import pytest

from smartscan.core.models import BandObservation, BandState
from smartscan.features.extractor import FeatureExtractor
from smartscan.prediction.periodicity import (
    autocorrelation_period,
    estimate_periodicity,
)
from smartscan.state.band_history import BandHistory


def make_obs(t, detected, snr=10.0):
    return BandObservation(
        band_id=0, timestamp=t, detected=detected, confidence=0.5,
        peak_power_db=-50.0, avg_power_db=-60.0,
        noise_floor_db=-100.0, estimated_snr_db=snr,
    )


class TestEstimatePeriodicity:
    def test_too_few_detections(self):
        est = estimate_periodicity([1.0, 2.0], current_time=5.0)
        assert est.estimated_period is None
        assert est.confidence == 0.0

    def test_regular_period(self):
        # Detections every 2 seconds
        times = [0.0, 2.0, 4.0, 6.0, 8.0]
        est = estimate_periodicity(times, current_time=9.0)
        assert est.estimated_period == pytest.approx(2.0, abs=0.1)
        assert est.confidence > 0.9

    def test_predicts_next_activity(self):
        times = [0.0, 2.0, 4.0, 6.0]
        est = estimate_periodicity(times, current_time=7.0)
        # Next activity should be around 8.0
        assert est.predicted_next_activity_time == pytest.approx(8.0, abs=0.5)

    def test_irregular_low_confidence(self):
        times = [0.0, 1.0, 5.0, 5.5, 12.0]
        est = estimate_periodicity(times, current_time=13.0)
        # Irregular → lower confidence
        assert est.confidence < 0.7


class TestAutocorrelationPeriod:
    def test_constant_signal_returns_none(self):
        signal = np.ones(20)
        assert autocorrelation_period(signal, bin_duration=1.0) is None

    def test_periodic_signal(self):
        # Period of 4 bins: [1,1,0,0,1,1,0,0,...]
        signal = np.tile([1, 1, 0, 0], 10).astype(float)
        period = autocorrelation_period(signal, bin_duration=1.0)
        assert period == pytest.approx(4.0, abs=1.0)


class TestFeatureExtractor:
    def test_extract_basic(self):
        extractor = FeatureExtractor(history_window=10)
        state = BandState(
            band_id=0, freq_start=100e6, freq_end=120e6,
            hit_count=3, miss_count=7, observation_count=10,
            avg_snr_db=15.0, rolling_activity_prob=0.4,
        )
        history = BandHistory(band_id=0)
        for i in range(10):
            history.add(make_obs(float(i), i % 3 == 0))

        features = extractor.extract(state, history, current_time=10.0)
        assert features.band_id == 0
        assert features.activity_ratio == pytest.approx(0.3)
        assert features.ewma_activity == pytest.approx(0.4)
        assert features.estimated_snr_db == 15.0

    def test_feature_vector_length(self):
        extractor = FeatureExtractor()
        state = BandState(band_id=0, freq_start=100e6, freq_end=120e6)
        history = BandHistory(band_id=0)
        features = extractor.extract(state, history, current_time=0.0)
        arr = features.to_array()
        assert len(arr) == 14

    def test_features_pass_through_state_period(self):
        """Periodicity is estimated by the state manager and stored on BandState;
        the extractor reads it through rather than recomputing (avoids double work)."""
        extractor = FeatureExtractor(history_window=50)
        state = BandState(
            band_id=0, freq_start=100e6, freq_end=120e6, observation_count=20,
            estimated_period=2.0,   # supplied by the state manager's online estimate
        )
        history = BandHistory(band_id=0)
        for t in np.arange(0, 20, 0.5):
            history.add(make_obs(float(t), (t % 2.0) < 0.5))
        features = extractor.extract(state, history, current_time=20.0)
        assert features.estimated_period == 2.0
