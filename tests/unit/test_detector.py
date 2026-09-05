"""Tests for the energy detector — validated at multiple SNR levels."""

import numpy as np
import pytest

from smartscan.core.config import DetectorConfig
from smartscan.core.models import AcquisitionMeta
from smartscan.dsp.detector import EnergyDetector
from smartscan.simulation.noise import generate_awgn, power_for_snr
from smartscan.simulation.waveform import generate_tone


def make_meta(cf=100e6, fs=20e6, bw=20e6, n=8192, t=1.0):
    return AcquisitionMeta(
        center_frequency=cf, sample_rate=fs, bandwidth=bw,
        num_samples=n, timestamp=t, dwell_time=n / fs,
    )


def make_detector(margin_db=6.0):
    return EnergyDetector(DetectorConfig(threshold_margin_db=margin_db, fft_size=1024))


class TestDetectorNoiseOnly:
    def test_no_detection_on_pure_noise(self):
        """Pure noise should rarely trigger detection with adequate margin."""
        det = make_detector(margin_db=10.0)
        rng = np.random.default_rng(42)
        false_alarms = 0
        trials = 30
        for i in range(trials):
            noise = generate_awgn(8192, power_dbm=-100.0, rng=rng)
            result = det.detect(noise, make_meta())
            if result.detected:
                false_alarms += 1
        # With 10 dB margin, false alarm rate should be low
        assert false_alarms < trials * 0.3


class TestDetectorStrongSignal:
    def test_detects_high_snr_signal(self):
        det = make_detector()
        rng = np.random.default_rng(42)
        noise_dbm = -100.0
        signal_dbm = power_for_snr(noise_dbm, 30.0)  # 30 dB SNR

        noise = generate_awgn(8192, power_dbm=noise_dbm, rng=rng)
        signal = generate_tone(8192, sample_rate=20e6, frequency_offset=2e6, power_dbm=signal_dbm)
        samples = noise + signal

        result = det.detect(samples, make_meta())
        assert result.detected is True
        assert result.estimated_snr_db > 15.0

    def test_confidence_high_for_strong_signal(self):
        det = make_detector()
        rng = np.random.default_rng(1)
        signal = generate_tone(8192, sample_rate=20e6, frequency_offset=1e6, power_dbm=-60.0)
        noise = generate_awgn(8192, power_dbm=-100.0, rng=rng)
        result = det.detect(noise + signal, make_meta())
        assert result.confidence >= 0.5


class TestDetectorSNRSweep:
    @pytest.mark.parametrize("snr_db", [40.0, 30.0, 20.0, 10.0])
    def test_high_snr_tone_detected(self, snr_db):
        """A narrowband tone at moderate-to-high integrated SNR is reliably detected.

        Note: a single tone gains ~10*log10(fft_size) dB of processing gain because
        its energy concentrates in one FFT bin while noise spreads across all bins.
        """
        det = make_detector(margin_db=6.0)
        rng = np.random.default_rng(123)
        noise_dbm = -100.0
        signal_dbm = power_for_snr(noise_dbm, snr_db)

        detections = 0
        trials = 15
        for _ in range(trials):
            noise = generate_awgn(8192, power_dbm=noise_dbm, rng=rng)
            signal = generate_tone(
                8192, sample_rate=20e6, frequency_offset=2e6, power_dbm=signal_dbm,
            )
            result = det.detect(noise + signal, make_meta())
            if result.detected:
                detections += 1

        assert detections / trials > 0.8

    def test_deeply_buried_wideband_signal_missed(self):
        """A WIDEBAND signal well below the noise floor gets no processing-gain
        concentration and should be reliably missed."""
        from smartscan.simulation.waveform import generate_bandlimited_noise

        det = make_detector(margin_db=6.0)
        rng = np.random.default_rng(123)
        noise_dbm = -100.0
        signal_dbm = power_for_snr(noise_dbm, -20.0)  # 20 dB below noise

        detections = 0
        trials = 15
        for _ in range(trials):
            noise = generate_awgn(8192, power_dbm=noise_dbm, rng=rng)
            signal = generate_bandlimited_noise(
                8192, sample_rate=20e6, bandwidth=15e6,
                power_dbm=signal_dbm, rng=rng,
            )
            result = det.detect(noise + signal, make_meta())
            if result.detected:
                detections += 1

        assert detections / trials < 0.3


class TestDetectorResult:
    def test_result_fields_populated(self):
        det = make_detector()
        signal = generate_tone(8192, sample_rate=20e6, frequency_offset=1e6, power_dbm=-60.0)
        result = det.detect(signal, make_meta())
        assert result.peak_power_db > result.noise_floor_db
        assert result.timestamp == 1.0
        assert result.freq_start <= result.freq_end

    def test_no_ground_truth_used(self):
        """Detector only receives samples + meta, never ground truth."""
        import inspect
        sig = inspect.signature(EnergyDetector.detect)
        params = set(sig.parameters.keys())
        assert params == {"self", "samples", "meta"}
