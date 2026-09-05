"""Power Spectral Density estimation."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from smartscan.core.models import PSDMethod, WindowFunction


def compute_psd(
    samples: np.ndarray,
    sample_rate: float,
    method: PSDMethod = PSDMethod.WELCH,
    window: WindowFunction = WindowFunction.HANN,
    fft_size: int = 1024,
    nperseg: int | None = None,
    overlap: float = 0.5,
    center_frequency: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the PSD of complex IQ samples.

    Args:
        samples: Complex IQ samples.
        sample_rate: Sample rate in Hz.
        method: PERIODOGRAM or WELCH.
        window: Window function.
        fft_size: FFT size.
        nperseg: Welch segment length (defaults to fft_size).
        overlap: Welch fractional overlap.
        center_frequency: Center frequency for the frequency axis.

    Returns:
        Tuple of (frequencies in Hz, PSD in dB/Hz).
    """
    window_name = window.value

    if method == PSDMethod.PERIODOGRAM:
        freqs, psd = sp_signal.periodogram(
            samples,
            fs=sample_rate,
            window=window_name,
            nfft=fft_size,
            detrend=False,
            return_onesided=False,
            scaling="density",
        )
    elif method == PSDMethod.WELCH:
        if nperseg is None:
            nperseg = min(fft_size, len(samples))
        nperseg = min(nperseg, len(samples))
        noverlap = int(nperseg * overlap)
        freqs, psd = sp_signal.welch(
            samples,
            fs=sample_rate,
            window=window_name,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=fft_size,
            detrend=False,
            return_onesided=False,
            scaling="density",
        )
    else:
        raise ValueError(f"Unknown PSD method: {method}")

    # Shift so DC is centered and apply center frequency offset
    freqs = np.fft.fftshift(freqs) + center_frequency
    psd = np.fft.fftshift(psd)

    # Convert to dB, guarding against log(0)
    psd_db = 10 * np.log10(np.maximum(psd, 1e-30))

    return freqs, psd_db
