"""Tests for the matched-filter and cyclostationary detectors + factory."""

import numpy as np

from smartscan.core.config import DetectorConfig
from smartscan.core.models import AcquisitionMeta, DetectorType
from smartscan.dsp.detector import (
    CyclostationaryDetector,
    EnergyDetector,
    MatchedFilterDetector,
    create_detector,
)
from smartscan.simulation.noise import generate_awgn
from smartscan.simulation.waveform import generate_tone


def _meta(n):
    return AcquisitionMeta(center_frequency=100e6, sample_rate=20e6, bandwidth=20e6,
                           num_samples=n, timestamp=0.0, dwell_time=n / 20e6)


def _signal(n, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    from smartscan.simulation.noise import power_for_snr
    noise = generate_awgn(n, -100.0, rng)
    tone = generate_tone(n, 20e6, frequency_offset=2e6,
                         power_dbm=power_for_snr(-100.0, snr_db))
    return noise + tone


def _noise(n, seed=0):
    return generate_awgn(n, -100.0, np.random.default_rng(seed))


class TestMatchedFilter:
    def test_detects_tone(self):
        det = MatchedFilterDetector(DetectorConfig(fft_size=1024))
        r = det.detect(_signal(4096, 20.0, 1), _meta(4096))
        assert r.detected is True
        assert r.estimated_snr_db > 0

    def test_rejects_noise(self):
        det = MatchedFilterDetector(DetectorConfig(fft_size=1024, threshold_margin_db=8.0))
        fa = sum(det.detect(_noise(4096, s), _meta(4096)).detected for s in range(20))
        assert fa <= 3   # low false-alarm rate on pure noise


class TestCyclostationary:
    def test_detects_structured_signal(self):
        det = CyclostationaryDetector(DetectorConfig(autocorr_threshold=0.15))
        r = det.detect(_signal(4096, 15.0, 2), _meta(4096))
        assert r.detected is True

    def test_rejects_white_noise(self):
        det = CyclostationaryDetector(DetectorConfig(autocorr_threshold=0.15))
        fa = sum(det.detect(_noise(4096, s), _meta(4096)).detected for s in range(20))
        assert fa <= 3

    def test_feature_higher_for_signal_than_noise(self):
        det = CyclostationaryDetector(DetectorConfig())
        sig = det.detect(_signal(4096, 20.0, 3), _meta(4096)).confidence
        noi = det.detect(_noise(4096, 3), _meta(4096)).confidence
        assert sig > noi


class TestFactory:
    def test_energy(self):
        assert isinstance(create_detector(DetectorConfig(detector_type=DetectorType.ENERGY)),
                          EnergyDetector)

    def test_matched(self):
        assert isinstance(create_detector(DetectorConfig(detector_type=DetectorType.MATCHED_FILTER)),
                          MatchedFilterDetector)

    def test_cyclo(self):
        assert isinstance(create_detector(DetectorConfig(detector_type=DetectorType.CYCLOSTATIONARY)),
                          CyclostationaryDetector)

    def test_string_type(self):
        assert isinstance(create_detector(DetectorConfig(detector_type="matched_filter")),
                          MatchedFilterDetector)
