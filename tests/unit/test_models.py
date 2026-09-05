"""Tests for core data models."""

import numpy as np
import pytest
from pydantic import ValidationError

from smartscan.core.models import (
    AcquisitionMeta,
    BandFeatures,
    BandObservation,
    BandState,
    DetectionResult,
    EmitterConfig,
    EmitterType,
    ExperimentResult,
    GroundTruthEvent,
    ScanDecision,
)


class TestScanDecision:
    def test_valid_construction(self):
        d = ScanDecision(
            band_id=3,
            center_frequency=100e6,
            bandwidth=20e6,
            dwell_time=0.01,
            priority_score=0.8,
            reason="high activity",
            timestamp=1.5,
        )
        assert d.band_id == 3
        assert d.center_frequency == 100e6
        assert d.dwell_time == 0.01

    def test_rejects_negative_band_id(self):
        with pytest.raises(ValidationError):
            ScanDecision(
                band_id=-1, center_frequency=100e6, bandwidth=20e6,
                dwell_time=0.01, timestamp=0.0,
            )

    def test_rejects_zero_bandwidth(self):
        with pytest.raises(ValidationError):
            ScanDecision(
                band_id=0, center_frequency=100e6, bandwidth=0,
                dwell_time=0.01, timestamp=0.0,
            )

    def test_rejects_negative_timestamp(self):
        with pytest.raises(ValidationError):
            ScanDecision(
                band_id=0, center_frequency=100e6, bandwidth=20e6,
                dwell_time=0.01, timestamp=-1.0,
            )


class TestDetectionResult:
    def test_valid_construction(self):
        d = DetectionResult(
            detected=True, confidence=0.95, peak_power_db=-30.0,
            avg_power_db=-40.0, noise_floor_db=-90.0, estimated_snr_db=50.0,
            freq_start=90e6, freq_end=110e6, timestamp=2.0,
        )
        assert d.detected is True
        assert d.confidence == 0.95

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            DetectionResult(
                detected=True, confidence=1.5, peak_power_db=-30.0,
                avg_power_db=-40.0, noise_floor_db=-90.0, estimated_snr_db=50.0,
                freq_start=90e6, freq_end=110e6, timestamp=2.0,
            )


class TestBandState:
    def test_defaults(self):
        bs = BandState(band_id=0, freq_start=100e6, freq_end=120e6)
        assert bs.hit_count == 0
        assert bs.miss_count == 0
        assert bs.observation_count == 0
        assert bs.rolling_activity_prob == 0.5

    def test_center_frequency(self):
        bs = BandState(band_id=0, freq_start=100e6, freq_end=120e6)
        assert bs.center_frequency == 110e6

    def test_bandwidth(self):
        bs = BandState(band_id=0, freq_start=100e6, freq_end=120e6)
        assert bs.bandwidth == 20e6

    def test_activity_ratio_zero_obs(self):
        bs = BandState(band_id=0, freq_start=100e6, freq_end=120e6)
        assert bs.activity_ratio == 0.0

    def test_activity_ratio_with_data(self):
        bs = BandState(
            band_id=0, freq_start=100e6, freq_end=120e6,
            hit_count=3, miss_count=7, observation_count=10,
        )
        assert bs.activity_ratio == pytest.approx(0.3)

    def test_time_since_scan_never_scanned(self):
        bs = BandState(band_id=0, freq_start=100e6, freq_end=120e6)
        assert bs.time_since_scan(10.0) == float("inf")

    def test_time_since_scan_with_data(self):
        bs = BandState(
            band_id=0, freq_start=100e6, freq_end=120e6,
            last_scan_time=5.0,
        )
        assert bs.time_since_scan(10.0) == pytest.approx(5.0)

    def test_time_since_detection_never_detected(self):
        bs = BandState(band_id=0, freq_start=100e6, freq_end=120e6)
        assert bs.time_since_detection(10.0) == float("inf")


class TestBandObservation:
    def test_valid(self):
        obs = BandObservation(
            band_id=5, timestamp=3.0, detected=True, confidence=0.8,
            peak_power_db=-40.0, avg_power_db=-50.0,
            noise_floor_db=-90.0, estimated_snr_db=40.0,
        )
        assert obs.band_id == 5


class TestEmitterConfig:
    def test_continuous_emitter(self):
        e = EmitterConfig(
            emitter_id=0, emitter_type=EmitterType.CONTINUOUS,
            center_frequency=150e6, bandwidth=5e6,
            amplitude=1.0, snr_db=20.0,
        )
        assert e.emitter_type == EmitterType.CONTINUOUS

    def test_periodic_emitter(self):
        e = EmitterConfig(
            emitter_id=1, emitter_type=EmitterType.PERIODIC_BURST,
            center_frequency=200e6, bandwidth=5e6,
            amplitude=0.5, period=2.0, duty_cycle=0.3,
        )
        assert e.period == 2.0
        assert e.duty_cycle == 0.3

    def test_rejects_invalid_duty_cycle(self):
        with pytest.raises(ValidationError):
            EmitterConfig(
                emitter_id=1, emitter_type=EmitterType.PERIODIC_BURST,
                center_frequency=200e6, bandwidth=5e6,
                amplitude=0.5, duty_cycle=1.5,
            )


class TestGroundTruthEvent:
    def test_construction(self):
        gt = GroundTruthEvent(
            emitter_id=0, freq_start=100e6, freq_end=105e6,
            time_start=1.0, time_end=2.0, amplitude=1.0, snr_db=20.0,
        )
        assert gt.emitter_id == 0


class TestBandFeatures:
    def test_to_array(self):
        f = BandFeatures(band_id=0, current_power_db=-50.0, estimated_snr_db=10.0)
        arr = f.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert len(arr) == 14
        assert arr[0] == -50.0  # current_power_db
        assert arr[1] == 10.0   # estimated_snr_db

    def test_to_array_inf_clamped(self):
        f = BandFeatures(band_id=0)
        arr = f.to_array()
        assert arr[2] == 1e6  # time_since_scan clamped from inf
        assert arr[3] == 1e6  # time_since_detection clamped

    def test_to_array_none_period(self):
        f = BandFeatures(band_id=0, estimated_period=None)
        arr = f.to_array()
        assert arr[11] == 0.0  # None maps to 0


class TestAcquisitionMeta:
    def test_valid(self):
        m = AcquisitionMeta(
            center_frequency=100e6, sample_rate=20e6, bandwidth=20e6,
            num_samples=200000, timestamp=1.0, dwell_time=0.01,
        )
        assert m.num_samples == 200000

    def test_no_ground_truth_fields(self):
        """AcquisitionMeta must not have any ground-truth fields."""
        field_names = set(AcquisitionMeta.model_fields.keys())
        forbidden = {"active", "emitters", "ground_truth", "activity", "is_active", "label"}
        assert field_names.isdisjoint(forbidden), (
            f"AcquisitionMeta has forbidden ground-truth fields: {field_names & forbidden}"
        )


class TestExperimentResult:
    def test_valid(self):
        r = ExperimentResult(
            scheduler_name="round_robin", scenario_name="sparse",
            seed=42, num_steps=1000, duration=60.0,
            probability_of_detection=0.85, probability_of_false_alarm=0.05,
            scan_hit_rate=0.3, activity_discovery_ratio=0.7,
            avg_discovery_delay=0.5, median_discovery_delay=0.4,
            p95_discovery_delay=1.2, scan_efficiency=0.6,
            band_coverage=0.95, starvation_rate=0.02,
        )
        assert r.probability_of_detection == 0.85
