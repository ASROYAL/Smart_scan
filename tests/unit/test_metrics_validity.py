"""Validity tests: distinct-event discovery, emitter-intercept ratio, and
scoring against the receiver's actual observation window."""

import pytest

from smartscan.core.models import GroundTruthEvent
from smartscan.evaluation.evaluator import Evaluator
from smartscan.evaluation.metrics import ScanRecord, activity_discovery_ratio


def _ev(eid, fstart, fend, tstart, tend):
    return GroundTruthEvent(emitter_id=eid, freq_start=fstart, freq_end=fend,
                            time_start=tstart, time_end=tend, amplitude=1.0, snr_db=20.0)


def _rec(t, fstart, fend, detected=True):
    return ScanRecord(scan_number=0, timestamp=t, band_id=0, detected=detected,
                      truly_active=True, reward=0.0, freq_start=fstart, freq_end=fend)


def _match(gt_events, records):
    ev = Evaluator(environment=None)
    return ev._match_detections_to_events(records, gt_events)


class TestDistinctDiscovery:
    def test_one_event_detected_repeatedly_is_not_inflated(self):
        gt = [_ev(0, 90e6, 110e6, 1.0, 2.0)]
        # three scans all catch the SAME single event
        recs = [_rec(1.1, 90e6, 110e6), _rec(1.4, 90e6, 110e6), _rec(1.8, 90e6, 110e6)]
        matched = _match(gt, recs)
        assert len(matched) == 1
        assert activity_discovery_ratio(len(matched), len(gt)) == 1.0  # not >1

    def test_several_events_only_one_detected(self):
        gt = [_ev(0, 90e6, 110e6, 1.0, 2.0),
              _ev(1, 190e6, 210e6, 1.0, 2.0),
              _ev(2, 290e6, 310e6, 1.0, 2.0)]
        recs = [_rec(1.5, 90e6, 110e6)]  # only overlaps event 0
        matched = _match(gt, recs)
        assert len(matched) == 1
        assert activity_discovery_ratio(len(matched), len(gt)) == pytest.approx(1 / 3)

    def test_multiple_events_same_emitter(self):
        # emitter 0 has two events (two illuminations); emitter 1 has one
        gt = [_ev(0, 90e6, 110e6, 1.0, 1.2),
              _ev(0, 90e6, 110e6, 3.0, 3.2),
              _ev(1, 290e6, 310e6, 1.0, 2.0)]
        recs = [_rec(1.1, 90e6, 110e6)]  # catches one event of emitter 0 only
        matched = _match(gt, recs)
        ev = Evaluator(environment=None)
        assert activity_discovery_ratio(len(matched), len(gt)) == pytest.approx(1 / 3)
        # but 1 of 2 distinct emitters intercepted
        assert ev._emitter_intercept_ratio(gt, matched) == pytest.approx(0.5)

    def test_no_events(self):
        assert activity_discovery_ratio(0, 0) == 0.0
        ev = Evaluator(environment=None)
        assert ev._emitter_intercept_ratio([], {}) == 0.0


class TestObservationWindowMatching:
    def test_out_of_window_event_not_matched(self):
        # event at 90 MHz; a scan whose 20 MHz window is 40-60 MHz must NOT match it
        gt = [_ev(0, 88e6, 92e6, 1.0, 2.0)]
        in_window = [_rec(1.5, 40e6, 60e6)]     # 20 MHz window far from 90 MHz
        assert len(_match(gt, in_window)) == 0
        on_window = [_rec(1.5, 82e6, 102e6)]    # 20 MHz window covering 90 MHz
        assert len(_match(gt, on_window)) == 1


class TestRunnerUsesReceiverWindow:
    def test_record_window_is_receiver_bandwidth_not_logical_band(self):
        # 10 bands over 1000 MHz -> 100 MHz logical bands, but the receiver only
        # observes its 20 MHz instantaneous bandwidth; records must reflect 20 MHz.
        from smartscan.core.config import SmartScanConfig
        from smartscan.evaluation.experiment import build_and_run
        cfg = SmartScanConfig()
        cfg.environment.num_bands = 10           # 100 MHz logical bands
        cfg.simulation.num_steps = 60
        out = build_and_run(cfg, "round_robin", "sparse", seed=1)
        r = out.artifacts.records[0]
        width = r.freq_end - r.freq_start
        assert width == pytest.approx(cfg.receiver.instantaneous_bandwidth)   # 20 MHz
        assert width < 100e6                     # not the 100 MHz logical band
