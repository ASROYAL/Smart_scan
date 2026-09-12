"""Tests for performance metrics."""

import pytest

from smartscan.core.models import GroundTruthEvent
from smartscan.evaluation import metrics
from smartscan.evaluation.metrics import ScanRecord


def test_censored_intercept_penalizes_undiscovered_events():
    mean, missed = metrics.censored_intercept_statistics(
        [0.0, 1.0], {0: 0.2}, mission_end=3.0,
    )
    assert mean == pytest.approx(1.1)
    assert missed == pytest.approx(0.5)


def rec(scan=0, t=0.0, band=0, detected=False, active=False, reward=0.0):
    return ScanRecord(
        scan_number=scan, timestamp=t, band_id=band,
        detected=detected, truly_active=active, reward=reward,
    )


def test_emitter_metrics_use_dwell_overlap_and_ignore_other_emitters():
    target = GroundTruthEvent(
        emitter_id=0, freq_start=100.0, freq_end=110.0,
        time_start=1.0, time_end=1.2, amplitude=1.0, snr_db=-2.0,
    )
    background = GroundTruthEvent(
        emitter_id=1, freq_start=200.0, freq_end=210.0,
        time_start=0.0, time_end=3.0, amplitude=1.0, snr_db=10.0,
    )
    records = [
        ScanRecord(0, 0.9, 0, True, True, 1.0, 100.0, 110.0, dwell_time=0.2),
        ScanRecord(1, 1.1, 1, True, True, 1.0, 200.0, 210.0, dwell_time=0.1),
    ]

    result = metrics.emitter_intercept_metrics(
        records, [target, background], emitter_id=0, mission_end=3.0
    )

    assert result.opportunities == 1
    assert result.scan_probability_of_detection == 1.0
    assert result.discovered_events == 1


class TestPD:
    def test_perfect_detection(self):
        records = [rec(detected=True, active=True) for _ in range(5)]
        assert metrics.probability_of_detection(records) == 1.0

    def test_half_detection(self):
        records = [
            rec(detected=True, active=True),
            rec(detected=False, active=True),
        ]
        assert metrics.probability_of_detection(records) == 0.5

    def test_no_active_opportunities(self):
        records = [rec(detected=False, active=False)]
        assert metrics.probability_of_detection(records) == 0.0


class TestPFA:
    def test_no_false_alarms(self):
        records = [rec(detected=False, active=False) for _ in range(5)]
        assert metrics.probability_of_false_alarm(records) == 0.0

    def test_all_false_alarms(self):
        records = [rec(detected=True, active=False) for _ in range(5)]
        assert metrics.probability_of_false_alarm(records) == 1.0


class TestScanHitRate:
    def test_half_hits(self):
        records = [rec(detected=True), rec(detected=False)]
        assert metrics.scan_hit_rate(records) == 0.5


class TestDiscoveryDelays:
    def test_basic_delay(self):
        event_starts = [1.0, 5.0]
        detections = {0: 1.5, 1: 6.0}
        delays = metrics.discovery_delays(event_starts, detections)
        assert delays == [0.5, 1.0]

    def test_undetected_event_excluded(self):
        event_starts = [1.0, 5.0]
        detections = {0: 1.5}  # event 1 never detected
        delays = metrics.discovery_delays(event_starts, detections)
        assert delays == [0.5]

    def test_delay_statistics(self):
        delays = [0.5, 1.0, 1.5, 2.0]
        mean, median, _p95 = metrics.delay_statistics(delays)
        assert mean == pytest.approx(1.25)
        assert median == pytest.approx(1.25)

    def test_empty_delays(self):
        assert metrics.delay_statistics([]) == (0.0, 0.0, 0.0)


class TestCoverage:
    def test_full_coverage(self):
        records = [rec(band=i) for i in range(5)]
        assert metrics.band_coverage(records, 5) == 1.0

    def test_partial_coverage(self):
        records = [rec(band=0), rec(band=1)]
        assert metrics.band_coverage(records, 4) == 0.5


class TestStarvationRate:
    def test_no_starvation(self):
        # Visit all 2 bands frequently
        records = [rec(t=float(t), band=t % 2) for t in range(10)]
        rate = metrics.starvation_rate(records, num_bands=2, duration=10.0,
                                        starvation_threshold=5.0)
        assert rate == 0.0

    def test_starved_band(self):
        # Band 0 visited once at t=0, band 1 never
        records = [rec(t=0.0, band=0)]
        rate = metrics.starvation_rate(records, num_bands=2, duration=100.0,
                                        starvation_threshold=5.0)
        assert rate == 1.0  # both bands have large gaps


class TestScanEfficiency:
    def test_efficiency(self):
        records = [
            rec(detected=True, active=True),   # useful
            rec(detected=True, active=False),  # false alarm
            rec(detected=False, active=False),
            rec(detected=False, active=True),  # miss
        ]
        assert metrics.scan_efficiency(records) == 0.25


class TestPrecisionRecallF1:
    def test_perfect(self):
        records = [
            rec(detected=True, active=True),
            rec(detected=False, active=False),
        ]
        p, r, f1 = metrics.precision_recall_f1(records)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_with_errors(self):
        records = [
            rec(detected=True, active=True),   # TP
            rec(detected=True, active=False),  # FP
            rec(detected=False, active=True),  # FN
        ]
        p, r, f1 = metrics.precision_recall_f1(records)
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)


class TestActivityDiscoveryRatio:
    def test_ratio(self):
        # 2 distinct events discovered out of 4 total = 0.5 (counts distinct events,
        # not per-scan true positives)
        assert metrics.activity_discovery_ratio(2, total_activity_events=4) == 0.5
        # re-detecting the same events cannot inflate past 1.0
        assert metrics.activity_discovery_ratio(9, total_activity_events=4) == 1.0

    def test_zero_events(self):
        assert metrics.activity_discovery_ratio(0, 0) == 0.0


class TestAverageReward:
    def test_average(self):
        records = [rec(reward=1.0), rec(reward=0.0), rec(reward=2.0)]
        assert metrics.average_reward(records) == pytest.approx(1.0)
