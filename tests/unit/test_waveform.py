"""Tests for waveform generators."""

import numpy as np
import pytest

from smartscan.simulation.waveform import generate_bandlimited_noise, generate_tone


class TestGenerateTone:
    def test_correct_length(self):
        tone = generate_tone(1000, sample_rate=1e6, frequency_offset=100e3)
        assert len(tone) == 1000

    def test_is_complex(self):
        tone = generate_tone(100, sample_rate=1e6, frequency_offset=0)
        assert np.iscomplexobj(tone)

    def test_dc_tone_is_constant_amplitude(self):
        tone = generate_tone(1000, sample_rate=1e6, frequency_offset=0, amplitude=2.0)
        magnitudes = np.abs(tone)
        assert np.allclose(magnitudes, 2.0, atol=1e-10)

    def test_peak_at_correct_frequency(self):
        fs = 1e6
        f_offset = 100e3
        n = 1024
        tone = generate_tone(n, sample_rate=fs, frequency_offset=f_offset, amplitude=1.0)
        spectrum = np.abs(np.fft.fft(tone))
        freqs = np.fft.fftfreq(n, d=1 / fs)
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        assert peak_freq == pytest.approx(f_offset, abs=fs / n)

    def test_negative_frequency_offset(self):
        fs = 1e6
        f_offset = -200e3
        n = 1024
        tone = generate_tone(n, sample_rate=fs, frequency_offset=f_offset)
        spectrum = np.abs(np.fft.fft(tone))
        freqs = np.fft.fftfreq(n, d=1 / fs)
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        assert peak_freq == pytest.approx(f_offset, abs=fs / n)

    def test_power_dbm(self):
        from smartscan.simulation.noise import dbm_to_watts
        target_dbm = -60.0
        tone = generate_tone(10000, sample_rate=1e6, frequency_offset=0, power_dbm=target_dbm)
        measured_power = np.mean(np.abs(tone) ** 2)
        expected_power = dbm_to_watts(target_dbm)
        assert measured_power == pytest.approx(expected_power, rel=0.01)

    def test_cannot_provide_both_amplitude_and_power(self):
        with pytest.raises(ValueError, match="not both"):
            generate_tone(100, 1e6, 0, amplitude=1.0, power_dbm=-60.0)


class TestBandlimitedNoise:
    def test_correct_length(self):
        sig = generate_bandlimited_noise(1000, 1e6, 100e3)
        assert len(sig) == 1000

    def test_is_complex(self):
        sig = generate_bandlimited_noise(100, 1e6, 100e3)
        assert np.iscomplexobj(sig)

    def test_energy_concentrated_in_bandwidth(self):
        fs = 1e6
        bw = 100e3
        n = 4096
        rng = np.random.default_rng(42)
        sig = generate_bandlimited_noise(n, fs, bw, rng=rng)
        spectrum = np.abs(np.fft.fft(sig)) ** 2
        freqs = np.fft.fftfreq(n, d=1 / fs)

        in_band_power = spectrum[np.abs(freqs) <= bw / 2].sum()
        total_power = spectrum.sum()

        # Most energy should be within the bandwidth
        assert in_band_power / total_power > 0.9


class TestChirpAndWaveformDispatch:
    def test_chirp_occupies_bandwidth(self):
        import numpy as np

        from smartscan.simulation.waveform import generate_chirp
        sig = generate_chirp(4096, 20e6, center_offset=0.0, bandwidth=5e6, power_dbm=-40.0)
        assert len(sig) == 4096 and np.iscomplexobj(sig)
        spec = np.abs(np.fft.fftshift(np.fft.fft(sig))) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(4096, 1 / 20e6))
        in_bw = spec[np.abs(freqs) <= 3e6].sum()
        assert in_bw / spec.sum() > 0.8      # energy concentrated in the swept band

    def test_emitter_waveform_dispatch(self):
        import numpy as np

        from smartscan.core.models import EmitterConfig, EmitterType, WaveformType
        from smartscan.simulation.emitters import create_emitter
        for wf in (WaveformType.TONE, WaveformType.CHIRP, WaveformType.PULSED,
                   WaveformType.BANDLIMITED_NOISE):
            cfg = EmitterConfig(emitter_id=0, emitter_type=EmitterType.CONTINUOUS,
                                center_frequency=100e6, bandwidth=5e6, amplitude=1.0,
                                snr_db=25.0, waveform=wf)
            e = create_emitter(cfg, noise_power_dbm=-100.0)
            sig = e.generate_samples(0.0, 2000, 20e6, 100e6, 20e6)
            assert sig is not None and len(sig) == 2000 and np.iscomplexobj(sig)

    def test_default_waveform_is_tone(self):
        from smartscan.core.models import EmitterConfig, EmitterType, WaveformType
        from smartscan.simulation.emitters import create_emitter
        cfg = EmitterConfig(emitter_id=0, emitter_type=EmitterType.RADAR_SCAN,
                            center_frequency=100e6, bandwidth=5e6, amplitude=1.0,
                            snr_db=25.0, scan_period=2.0, beam_dwell=0.2)
        e = create_emitter(cfg, noise_power_dbm=-100.0)
        assert e._effective_waveform() == WaveformType.TONE   # opt-in for chirp

    def test_chirp_opt_in(self):
        from smartscan.core.models import EmitterConfig, EmitterType, WaveformType
        from smartscan.simulation.emitters import create_emitter
        cfg = EmitterConfig(emitter_id=0, emitter_type=EmitterType.RADAR_SCAN,
                            center_frequency=100e6, bandwidth=5e6, amplitude=1.0,
                            snr_db=25.0, scan_period=2.0, beam_dwell=0.2,
                            waveform=WaveformType.CHIRP)
        assert create_emitter(cfg, noise_power_dbm=-100.0)._effective_waveform() == WaveformType.CHIRP
