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


def group_into_episodes(detection_times: list[float], episode_gap: float) -> list[float]:
    """Cluster detection timestamps into activity episodes; return episode START times.

    Consecutive detections closer than ``episode_gap`` belong to the same episode
    (one burst / illumination that the receiver caught several times); a larger gap
    starts a new episode. Periodicity is then judged from the *episode* cadence,
    not from every positive scan — so a scheduler that camps on a band and detects
    it many times within one ON-window does not fake a tiny period.
    """
    if not detection_times:
        return []
    times = sorted(detection_times)
    starts = [times[0]]
    prev = times[0]
    for t in times[1:]:
        if t - prev > episode_gap:
            starts.append(t)
        prev = t
    return starts


def estimate_periodicity(
    detection_times: list[float],
    current_time: float,
    min_detections: int = 4,
    episode_gap: float = 0.05,
) -> PeriodicityEstimate:
    """Estimate the activity period from the cadence of activity EPISODES.

    Args:
        detection_times: Timestamps (seconds) at which activity was detected.
        current_time: Current simulation time.
        min_detections: Minimum detections before attempting an estimate.
        episode_gap: Detections closer than this (seconds) are one episode.

    Returns:
        PeriodicityEstimate with period, confidence, and next-activity prediction.
    """
    if len(detection_times) < min_detections:
        return PeriodicityEstimate(None, 0.0, None)

    episode_starts = group_into_episodes(detection_times, episode_gap)
    # Need at least 3 episodes (2 intervals) to judge a repeating period
    if len(episode_starts) < 3:
        return PeriodicityEstimate(None, 0.0, None)

    intervals = np.diff(np.array(episode_starts))
    positive = intervals[intervals > 1e-6]
    if len(positive) < 2:
        return PeriodicityEstimate(None, 0.0, None)

    median_period = float(np.median(positive))
    mean_interval = float(np.mean(positive))
    std_interval = float(np.std(positive))
    if mean_interval <= 0:
        return PeriodicityEstimate(None, 0.0, None)

    cv = std_interval / mean_interval
    confidence = float(np.clip(1.0 - cv, 0.0, 1.0))

    last_episode = float(episode_starts[-1])
    if median_period > 0:
        cycles_elapsed = int((current_time - last_episode) / median_period)
        next_time = last_episode + (cycles_elapsed + 1) * median_period
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
        if (autocorr[lag] > autocorr[lag - 1] and autocorr[lag] >= autocorr[lag + 1]
                and autocorr[lag] > 0.3):  # require a reasonably strong peak
            peak_lag = lag
            break

    if peak_lag is None:
        return None
    return peak_lag * bin_duration
