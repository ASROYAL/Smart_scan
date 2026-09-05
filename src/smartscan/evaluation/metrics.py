"""Performance metric calculations.

These functions operate on records collected during a run. Detection metrics
compare detector output against ground truth (evaluation only). Scheduler
metrics measure coverage, efficiency, and delay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScanRecord:
    """One scan's outcome, annotated with ground truth for evaluation."""
    scan_number: int
    timestamp: float
    band_id: int
    detected: bool           # what the detector said
    truly_active: bool       # ground truth (evaluation only)
    reward: float


def probability_of_detection(records: list[ScanRecord]) -> float:
    """PD = correctly detected active opportunities / total active opportunities."""
    active = [r for r in records if r.truly_active]
    if not active:
        return 0.0
    correct = sum(1 for r in active if r.detected)
    return correct / len(active)


def probability_of_false_alarm(records: list[ScanRecord]) -> float:
    """PFA = false detections / inactive opportunities."""
    inactive = [r for r in records if not r.truly_active]
    if not inactive:
        return 0.0
    false_alarms = sum(1 for r in inactive if r.detected)
    return false_alarms / len(inactive)


def scan_hit_rate(records: list[ScanRecord]) -> float:
    """Fraction of scans that detected activity."""
    if not records:
        return 0.0
    hits = sum(1 for r in records if r.detected)
    return hits / len(records)


def activity_discovery_ratio(
    records: list[ScanRecord],
    total_activity_events: int,
) -> float:
    """Fraction of distinct activity events that were discovered."""
    if total_activity_events == 0:
        return 0.0
    discovered = sum(1 for r in records if r.truly_active and r.detected)
    return min(1.0, discovered / total_activity_events)


def discovery_delays(
    event_start_times: list[float],
    detection_times_by_event: dict[int, float],
) -> list[float]:
    """Compute discovery delay for each detected event.

    Args:
        event_start_times: Start time of each ground-truth event.
        detection_times_by_event: Map from event index to first detection time.

    Returns:
        List of delays (detection_time - event_start) for discovered events.
    """
    delays = []
    for idx, start in enumerate(event_start_times):
        if idx in detection_times_by_event:
            delay = detection_times_by_event[idx] - start
            if delay >= 0:
                delays.append(delay)
    return delays


def delay_statistics(delays: list[float]) -> tuple[float, float, float]:
    """Return (mean, median, p95) of discovery delays."""
    if not delays:
        return 0.0, 0.0, 0.0
    arr = np.array(delays)
    return float(np.mean(arr)), float(np.median(arr)), float(np.percentile(arr, 95))


def scan_efficiency(records: list[ScanRecord]) -> float:
    """Useful observations (true detections) / total observations."""
    if not records:
        return 0.0
    useful = sum(1 for r in records if r.detected and r.truly_active)
    return useful / len(records)


def band_coverage(records: list[ScanRecord], num_bands: int) -> float:
    """Fraction of bands visited at least once."""
    if num_bands == 0:
        return 0.0
    visited = {r.band_id for r in records}
    return len(visited) / num_bands


def starvation_rate(
    records: list[ScanRecord],
    num_bands: int,
    duration: float,
    starvation_threshold: float,
) -> float:
    """Fraction of bands whose maximum revisit gap exceeded the starvation threshold."""
    if num_bands == 0:
        return 0.0

    visits_by_band: dict[int, list[float]] = {}
    for r in records:
        visits_by_band.setdefault(r.band_id, []).append(r.timestamp)

    starved = 0
    for band_id in range(num_bands):
        times = sorted(visits_by_band.get(band_id, []))
        # Include start (0) and end (duration) as boundaries
        boundaries = [0.0] + times + [duration]
        max_gap = max(boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1))
        if max_gap > starvation_threshold:
            starved += 1

    return starved / num_bands


def average_reward(records: list[ScanRecord]) -> float:
    if not records:
        return 0.0
    return float(np.mean([r.reward for r in records]))


def precision_recall_f1(records: list[ScanRecord]) -> tuple[float, float, float]:
    """Precision, recall, F1 of the detector treating truly_active as ground truth."""
    tp = sum(1 for r in records if r.detected and r.truly_active)
    fp = sum(1 for r in records if r.detected and not r.truly_active)
    fn = sum(1 for r in records if not r.detected and r.truly_active)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
