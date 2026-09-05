"""IQ waveform generators for simulated RF signals."""

from __future__ import annotations

import numpy as np

from smartscan.simulation.noise import dbm_to_watts


def generate_tone(
    num_samples: int,
    sample_rate: float,
    frequency_offset: float,
    amplitude: float | None = None,
    power_dbm: float | None = None,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate a complex sinusoidal tone at a frequency offset from center.

    Provide either amplitude (linear) or power_dbm, not both.

    Args:
        num_samples: Number of complex samples.
        sample_rate: Sample rate in Hz.
        frequency_offset: Offset from center frequency in Hz.
        amplitude: Linear amplitude (mutually exclusive with power_dbm).
        power_dbm: Signal power in dBm (mutually exclusive with amplitude).
        phase: Initial phase in radians.

    Returns:
        Complex128 array representing the tone in baseband.
    """
    if amplitude is not None and power_dbm is not None:
        raise ValueError("Provide either amplitude or power_dbm, not both")
    if amplitude is None and power_dbm is None:
        amplitude = 1.0

    if power_dbm is not None:
        amplitude = np.sqrt(dbm_to_watts(power_dbm))

    t = np.arange(num_samples) / sample_rate
    signal = amplitude * np.exp(1j * (2 * np.pi * frequency_offset * t + phase))
    return signal


def generate_bandlimited_noise(
    num_samples: int,
    sample_rate: float,
    bandwidth: float,
    center_offset: float = 0.0,
    power_dbm: float = -60.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate band-limited noise centered at a frequency offset.

    Models a wideband signal (e.g., digital modulation) occupying a specific
    bandwidth, rather than a single tone.

    Args:
        num_samples: Number of complex samples.
        sample_rate: Sample rate in Hz.
        bandwidth: Signal bandwidth in Hz.
        center_offset: Center frequency offset from baseband in Hz.
        power_dbm: Total signal power in dBm.
        rng: Random generator for reproducibility.

    Returns:
        Complex128 array.
    """
    if rng is None:
        rng = np.random.default_rng()

    power_watts = dbm_to_watts(power_dbm)
    sigma = np.sqrt(power_watts / 2)

    # Generate baseband noise, then filter to bandwidth
    raw = sigma * (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples))

    # Frequency-domain filtering
    spectrum = np.fft.fft(raw)
    freqs = np.fft.fftfreq(num_samples, d=1 / sample_rate)

    # Zero out frequencies outside the desired bandwidth
    mask = np.abs(freqs) <= bandwidth / 2
    spectrum[~mask] = 0.0

    # Scale to preserve target power after filtering
    filtered = np.fft.ifft(spectrum)
    current_power = np.mean(np.abs(filtered) ** 2)
    if current_power > 0:
        filtered *= np.sqrt(power_watts / current_power)

    # Shift to center_offset
    if center_offset != 0.0:
        t = np.arange(num_samples) / sample_rate
        filtered *= np.exp(1j * 2 * np.pi * center_offset * t)

    return filtered
