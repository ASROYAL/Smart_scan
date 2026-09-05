"""Additive White Gaussian Noise (AWGN) generator for IQ samples."""

from __future__ import annotations

import numpy as np


def generate_awgn(
    num_samples: int,
    power_dbm: float = -100.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate complex AWGN with specified power.

    Args:
        num_samples: Number of complex samples to generate.
        power_dbm: Noise power in dBm.
        rng: Numpy random generator for reproducibility.

    Returns:
        Complex128 array of noise samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    power_watts = dbm_to_watts(power_dbm)
    # Split power equally between I and Q channels
    sigma = np.sqrt(power_watts / 2)
    noise = sigma * (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples))
    return noise


def dbm_to_watts(dbm: float) -> float:
    """Convert dBm to Watts."""
    return 10 ** ((dbm - 30) / 10)


def watts_to_dbm(watts: float) -> float:
    """Convert Watts to dBm."""
    if watts <= 0:
        return -np.inf
    return 10 * np.log10(watts) + 30


def power_for_snr(noise_power_dbm: float, snr_db: float) -> float:
    """Calculate signal power in dBm to achieve target SNR above noise."""
    return noise_power_dbm + snr_db
