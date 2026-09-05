"""Periodicity estimation from detection timestamps.

Estimates whether a band's activity recurs on a regular period, using both
inter-arrival statistics and autocorrelation of a binned activity signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PeriodicityEstimate:
    estimated_period: float | None
    confidence: float
    predicted_next_activity_time: float | None


def estimate_periodicity(
    detection_times: list[float],
    current_time: float,
    min_detections: int = 4,
) -> PeriodicityEstimate:
    """Estimate the activity period from detection timestamps.

    Args:
        detection_times: Timestamps (seconds) at which activity was detected.
        current_time: Current simulation time.
        min_detections: Minimum detections needed to estimate a period.

    Returns:
        PeriodicityEstimate with period, confidence, and next-activity prediction.
    """
    if len(detection_times) < min_detections:
        return PeriodicityEstimate(None, 0.0, None)

    times = np.sort(np.array(detection_times))
    intervals = np.diff(times)

    # Filter out sub-resolution intervals (consecutive detections within one burst)
    positive = intervals[intervals > 1e-6]
    if len(positive) < min_detections - 1:
        return PeriodicityEstimate(None, 0.0, None)

    # Robust period estimate: median of inter-arrival intervals
    median_period = float(np.median(positive))

    # Confidence: low coefficient of variation → high confidence in periodicity
    mean_interval = float(np.mean(positive))
    std_interval = float(np.std(positive))
    if mean_interval <= 0:
        return PeriodicityEstimate(None, 0.0, None)

    cv = std_interval / mean_interval
    confidence = float(np.clip(1.0 - cv, 0.0, 1.0))

    # Predict next activity time
    last_detection = float(times[-1])
    if median_period > 0:
        cycles_elapsed = int((current_time - last_detection) / median_period)
        next_time = last_detection + (cycles_elapsed + 1) * median_period
    else:
        next_time = None

    return PeriodicityEstimate(median_period, confidence, next_time)


def autocorrelation_period(
    activity_signal: np.ndarray,
    bin_duration: float,
    max_lag: int | None = None,
) -> float | None:
    """Estimate period via autocorrelation of a binned binary activity signal.

    Args:
        activity_signal: Binary array (1=active, 0=inactive) sampled at bin_duration.
        bin_duration: Time per bin in seconds.
        max_lag: Maximum lag to consider.

    Returns:
        Estimated period in seconds, or None if no clear periodicity.
    """
    x = np.asarray(activity_signal, dtype=float)
    if len(x) < 4 or np.all(x == x[0]):
        return None

    x = x - np.mean(x)
    autocorr = np.correlate(x, x, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]  # keep non-negative lags

    if autocorr[0] <= 0:
        return None
    autocorr = autocorr / autocorr[0]

    if max_lag is None:
        max_lag = len(autocorr) - 1
    max_lag = min(max_lag, len(autocorr) - 1)

    # Find first peak after the zero-lag, skipping the initial descent
    peak_lag = None
    for lag in range(2, max_lag):
        if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] >= autocorr[lag + 1]:
            if autocorr[lag] > 0.3:  # require a reasonably strong peak
                peak_lag = lag
                break

    if peak_lag is None:
        return None
    return peak_lag * bin_duration
