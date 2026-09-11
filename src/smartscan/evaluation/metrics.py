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
    # the ACTUAL frequency window the receiver observed this scan (from acquisition
    # metadata), which may be narrower than the logical band. Truth and event
    # matching use this window, not the whole BandState.
    freq_start: float = 0.0
    freq_end: float = 0.0
    predicted_prob: float = 0.5     # pre-scan predicted activity probability
    predicted_active: bool = False  # predicted_prob >= threshold


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
    num_events_discovered: int,
    total_activity_events: int,
) -> float:
    """Fraction of DISTINCT ground-truth events discovered at least once.

    Pass the count of distinct events that were intercepted (from the
    event→first-detection matching), NOT the number of true-positive scans:
    counting every TP scan and capping at 1 inflates the ratio when one event
    is re-detected many times.
    """
    if total_activity_events == 0:
        return 0.0
    return min(1.0, num_events_discovered / total_activity_events)


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
        boundaries = [0.0, *times, duration]
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


# ---------------------------------------------------------------------------
# EW-vocabulary figures of merit (problem-statement naming)
# ---------------------------------------------------------------------------

def percentage_correct_predictions(records: list[ScanRecord]) -> float:
    """Fraction of scans whose pre-scan activity prediction matched ground truth.

    Before each observation the predictor gives a probability the chosen band is
    active; thresholded at 0.5 it becomes ``predicted_active``. This scores that
    against what was truly there — the "percentage of correct predictions".
    """
    if not records:
        return 0.0
    correct = sum(1 for r in records if r.predicted_active == r.truly_active)
    return correct / len(records)


def brier_score(records: list[ScanRecord]) -> float:
    """Mean squared error of the predicted activity probability (0 = perfect).

    Brier = mean( (predicted_prob - truly_active)^2 ). A proper scoring rule for
    probability quality — lower is better.
    """
    if not records:
        return 0.0
    return float(np.mean([(r.predicted_prob - (1.0 if r.truly_active else 0.0)) ** 2
                          for r in records]))


def log_loss(records: list[ScanRecord], eps: float = 1e-6) -> float:
    """Mean binary cross-entropy of the predicted probability (lower is better)."""
    if not records:
        return 0.0
    losses = []
    for r in records:
        p = min(max(r.predicted_prob, eps), 1.0 - eps)
        y = 1.0 if r.truly_active else 0.0
        losses.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean(losses))


def average_intercept_time_error(
    predicted_next_times: dict[int, float],
    actual_next_times: dict[int, float],
) -> float:
    """Mean |predicted - actual| next-activity time over bands with both values.

    Uses periodicity-based predictions (last detection + estimated period) versus
    the true next event start — the problem statement's "average intercept time
    error". Returns 0.0 when no band has a comparable prediction.
    """
    errors = []
    for band_id, predicted in predicted_next_times.items():
        actual = actual_next_times.get(band_id)
        if actual is not None:
            errors.append(abs(predicted - actual))
    if not errors:
        return 0.0
    return float(np.mean(errors))


def intercept_rate(records: list[ScanRecord]) -> float:
    """Average intercept rate = successful intercepts per scan (alias of scan hit rate,
    in EW terminology)."""
    return scan_hit_rate(records)


def sensitivity_curve(
    detector,
    make_meta,
    snr_values: list[float],
    noise_power_dbm: float = -100.0,
    num_samples: int = 4096,
    trials: int = 20,
    sample_rate: float = 20e6,
    seed: int = 0,
) -> list[tuple[float, float, float]]:
    """Detector sensitivity sweep: PD (and PFA) versus SNR.

    For each SNR, injects a tone at that SNR into fresh noise ``trials`` times and
    measures the detection rate (PD); also measures the false-alarm rate on
    noise-only frames (PFA). Characterises receiver **sensitivity** — the FoM the
    problem statement asks for.

    Args:
        detector: an object with ``detect(samples, meta) -> DetectionResult``.
        make_meta: callable(num_samples) -> AcquisitionMeta for the frames.
        snr_values: SNRs (dB) to sweep.
        noise_power_dbm / num_samples / trials / sample_rate / seed: sweep params.

    Returns:
        List of (snr_db, pd, pfa) tuples.
    """
    from smartscan.simulation.noise import generate_awgn, power_for_snr
    from smartscan.simulation.waveform import generate_tone

    rng = np.random.default_rng(seed)
    curve = []
    # PFA is SNR-independent; measure once on noise-only frames.
    fa = 0
    for _ in range(trials):
        noise = generate_awgn(num_samples, noise_power_dbm, rng)
        if detector.detect(noise, make_meta(num_samples)).detected:
            fa += 1
    pfa = fa / trials if trials else 0.0

    for snr in snr_values:
        hits = 0
        power_dbm = power_for_snr(noise_power_dbm, snr)
        for _ in range(trials):
            noise = generate_awgn(num_samples, noise_power_dbm, rng)
            tone = generate_tone(num_samples, sample_rate, frequency_offset=0.0,
                                 power_dbm=power_dbm)
            if detector.detect(noise + tone, make_meta(num_samples)).detected:
                hits += 1
        curve.append((snr, hits / trials if trials else 0.0, pfa))
    return curve
