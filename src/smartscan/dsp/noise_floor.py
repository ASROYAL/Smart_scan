"""Noise floor estimation from a PSD.

Robust estimators that work even when signals are present, by assuming most
of the spectrum is noise (median/percentile are robust to sparse strong bins).
"""

from __future__ import annotations

import numpy as np

from smartscan.core.models import NoiseEstMethod


def estimate_noise_floor(
    psd_db: np.ndarray,
    method: NoiseEstMethod = NoiseEstMethod.MEDIAN,
    percentile: float = 25.0,
) -> float:
    """Estimate the noise floor (dB) from a PSD.

    Args:
        psd_db: PSD values in dB.
        method: Estimation method.
        percentile: Percentile to use for PERCENTILE method.

    Returns:
        Estimated noise floor in dB.
    """
    match method:
        case NoiseEstMethod.MEDIAN:
            return float(np.median(psd_db))
        case NoiseEstMethod.PERCENTILE:
            return float(np.percentile(psd_db, percentile))
        case NoiseEstMethod.MOVING:
            # Moving estimate: median of a smoothed version
            kernel = min(len(psd_db) // 8 * 2 + 1, 21)
            if kernel < 3:
                return float(np.median(psd_db))
            smoothed = _moving_median(psd_db, kernel)
            return float(np.median(smoothed))
        case _:
            raise ValueError(f"Unknown noise estimation method: {method}")


def _moving_median(x: np.ndarray, kernel: int) -> np.ndarray:
    """Compute a moving median with edge padding."""
    pad = kernel // 2
    padded = np.pad(x, pad, mode="edge")
    result = np.empty_like(x)
    for i in range(len(x)):
        result[i] = np.median(padded[i : i + kernel])
    return result


def compute_threshold(
    noise_floor_db: float,
    margin_db: float = 6.0,
) -> float:
    """Compute the detection threshold above the noise floor."""
    return noise_floor_db + margin_db
