"""Tests for the recorded IQ file source."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from smartscan.acquisition.file_source import RecordedIQSource, write_iq_file
from smartscan.core.models import AcquisitionMeta


class TestWriteAndRead:
    def test_roundtrip(self):
        iq = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.c64"
            write_iq_file(iq, path, sample_rate=20e6, center_frequency=100e6)

            src = RecordedIQSource(path)
            assert src.get_sample_rate() == 20e6
            assert src.recorded_center_frequency == 100e6
            assert src.num_samples_available == 1000

    def test_read_samples(self):
        iq = (np.ones(500) + 1j * np.zeros(500)).astype(np.complex64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ones.c64"
            write_iq_file(iq, path, sample_rate=10e6, center_frequency=50e6)
            src = RecordedIQSource(path)
            samples, meta = src.read_samples(50e6, 10e6, 100)
            assert len(samples) == 100
            assert np.iscomplexobj(samples)
            assert isinstance(meta, AcquisitionMeta)

    def test_returns_complex128(self):
        iq = (np.random.randn(200) + 1j * np.random.randn(200)).astype(np.complex64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t.c64"
            write_iq_file(iq, path, sample_rate=10e6, center_frequency=50e6)
            src = RecordedIQSource(path)
            samples, _ = src.read_samples(50e6, 10e6, 50)
            assert samples.dtype == np.complex128

    def test_time_advances(self):
        iq = (np.random.randn(1000) + 1j * np.random.randn(1000)).astype(np.complex64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "t.c64"
            write_iq_file(iq, path, sample_rate=1000.0, center_frequency=50e6)
            src = RecordedIQSource(path)
            assert src.get_time() == 0.0
            src.read_samples(50e6, 500.0, 100)
            # 100 samples / 1000 sps = 0.1 s
            assert src.get_time() == pytest.approx(0.1)

    def test_wraps_around(self):
        iq = (np.arange(100) + 1j * np.zeros(100)).astype(np.complex64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wrap.c64"
            write_iq_file(iq, path, sample_rate=1000.0, center_frequency=50e6)
            src = RecordedIQSource(path)
            # Read more than available across two reads
            s1, _ = src.read_samples(50e6, 500.0, 80)
            s2, _ = src.read_samples(50e6, 500.0, 80)  # wraps
            assert len(s1) == 80
            assert len(s2) == 80


class TestErrors:
    def test_missing_iq_file(self):
        with pytest.raises(FileNotFoundError):
            RecordedIQSource("nonexistent.c64")

    def test_missing_metadata(self):
        iq = np.ones(10, dtype=np.complex64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nometa.c64"
            iq.tofile(path)  # write IQ without sidecar
            with pytest.raises(FileNotFoundError):
                RecordedIQSource(path)


class TestSamePipeline:
    """Recorded IQ must flow through the same detector as simulated IQ."""

    def test_detector_processes_recorded_iq(self):
        from smartscan.core.config import DetectorConfig
        from smartscan.dsp.detector import EnergyDetector
        from smartscan.simulation.waveform import generate_tone
        from smartscan.simulation.noise import generate_awgn

        # Create a recording with a strong tone
        rng = np.random.default_rng(0)
        signal = generate_tone(8192, sample_rate=20e6, frequency_offset=2e6, power_dbm=-60.0)
        noise = generate_awgn(8192, power_dbm=-100.0, rng=rng)
        iq = (signal + noise).astype(np.complex64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "signal.c64"
            write_iq_file(iq, path, sample_rate=20e6, center_frequency=100e6)
            src = RecordedIQSource(path)
            samples, meta = src.read_samples(100e6, 20e6, 8192)

            detector = EnergyDetector(DetectorConfig(threshold_margin_db=6.0, fft_size=1024))
            result = detector.detect(samples, meta)
            assert result.detected is True
