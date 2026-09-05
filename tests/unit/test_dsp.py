"""Tests for FFT and PSD."""

import numpy as np
import pytest

from smartscan.core.models import PSDMethod, WindowFunction
from smartscan.dsp.fft import fft_frequencies, get_window, windowed_fft
from smartscan.dsp.psd import compute_psd
from smartscan.simulation.waveform import generate_tone


class TestWindows:
    @pytest.mark.parametrize("window", list(WindowFunction))
    def test_window_length(self, window):
        w = get_window(window, 256)
        assert len(w) == 256

    @pytest.mark.parametrize("window", list(WindowFunction))
    def test_window_essentially_nonnegative(self, window):
        # Blackman legitimately dips to ~-1e-17 at its edges; allow tiny negatives.
        w = get_window(window, 128)
        assert np.all(w >= -1e-6)

    def test_window_peaks_near_one(self):
        for window in WindowFunction:
            w = get_window(window, 128)
            assert np.max(w) == pytest.approx(1.0, abs=0.05)


class TestWindowedFFT:
    def test_output_length(self):
        samples = np.ones(1024, dtype=complex)
        spectrum = windowed_fft(samples, fft_size=1024)
        assert len(spectrum) == 1024

    def test_tone_peak_location(self):
        fs = 20e6
        n = 1024
        f_offset = 5e6
        tone = generate_tone(n, sample_rate=fs, frequency_offset=f_offset, amplitude=1.0)
        spectrum = windowed_fft(tone, fft_size=n)
        freqs = fft_frequencies(n, fs)
        peak_idx = np.argmax(np.abs(spectrum))
        assert freqs[peak_idx] == pytest.approx(f_offset, abs=2 * fs / n)

    def test_zero_padding(self):
        samples = np.ones(500, dtype=complex)
        spectrum = windowed_fft(samples, fft_size=1024)
        assert len(spectrum) == 1024


class TestFFTFrequencies:
    def test_centered_at_dc(self):
        freqs = fft_frequencies(1024, 20e6)
        assert freqs[512] == pytest.approx(0.0, abs=1e-6)

    def test_center_offset(self):
        freqs = fft_frequencies(1024, 20e6, center_frequency=100e6)
        assert freqs[512] == pytest.approx(100e6, abs=1.0)


class TestComputePSD:
    @pytest.mark.parametrize("method", list(PSDMethod))
    def test_psd_length_matches_fft(self, method):
        fs = 20e6
        samples = generate_tone(4096, sample_rate=fs, frequency_offset=1e6)
        freqs, psd = compute_psd(samples, fs, method=method, fft_size=1024)
        assert len(freqs) == len(psd)
        assert len(psd) == 1024

    def test_tone_appears_as_peak(self):
        fs = 20e6
        f_offset = 3e6
        samples = generate_tone(8192, sample_rate=fs, frequency_offset=f_offset, amplitude=1.0)
        freqs, psd = compute_psd(samples, fs, method=PSDMethod.WELCH, fft_size=1024)
        peak_idx = np.argmax(psd)
        assert freqs[peak_idx] == pytest.approx(f_offset, abs=3 * fs / 1024)

    def test_psd_in_db(self):
        fs = 20e6
        samples = generate_tone(4096, sample_rate=fs, frequency_offset=0, amplitude=1.0)
        _, psd = compute_psd(samples, fs, fft_size=1024)
        # dB values, peak should be much higher than the floor
        assert np.max(psd) - np.median(psd) > 10

    def test_center_frequency_offset(self):
        fs = 20e6
        samples = generate_tone(4096, sample_rate=fs, frequency_offset=0, amplitude=1.0)
        freqs, _ = compute_psd(samples, fs, fft_size=1024, center_frequency=100e6)
        assert np.median(freqs) == pytest.approx(100e6, abs=fs)
