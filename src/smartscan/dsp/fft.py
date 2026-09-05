"""FFT and windowing utilities."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from smartscan.core.models import WindowFunction


def get_window(window: WindowFunction, size: int) -> np.ndarray:
    """Return a window function array of the given size."""
    match window:
        case WindowFunction.HANN:
            return sp_signal.windows.hann(size, sym=False)
        case WindowFunction.HAMMING:
            return sp_signal.windows.hamming(size, sym=False)
        case WindowFunction.BLACKMAN:
            return sp_signal.windows.blackman(size, sym=False)
        case _:
            raise ValueError(f"Unknown window: {window}")


def windowed_fft(
    samples: np.ndarray,
    window: WindowFunction = WindowFunction.HANN,
    fft_size: int | None = None,
) -> np.ndarray:
    """Compute a windowed FFT of complex IQ samples.

    Args:
        samples: Complex IQ samples.
        window: Window function to apply.
        fft_size: FFT size (defaults to len(samples), truncates or zero-pads).

    Returns:
        Complex FFT output, fftshifted so DC is centered.
    """
    if fft_size is None:
        fft_size = len(samples)

    if len(samples) >= fft_size:
        seg = samples[:fft_size]
    else:
        seg = np.zeros(fft_size, dtype=complex)
        seg[: len(samples)] = samples

    win = get_window(window, fft_size)
    windowed = seg * win
    spectrum = np.fft.fftshift(np.fft.fft(windowed))
    return spectrum


def fft_frequencies(
    fft_size: int,
    sample_rate: float,
    center_frequency: float = 0.0,
) -> np.ndarray:
    """Return the frequency axis for an fftshifted FFT, offset by center frequency."""
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1 / sample_rate))
    return freqs + center_frequency
