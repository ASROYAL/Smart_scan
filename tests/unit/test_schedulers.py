"""Tests for baseline schedulers."""

import pytest

from smartscan.core.config import ReceiverConfig, SchedulerConfig
from smartscan.core.models import BandObservation, BandState
from smartscan.schedulers.priority_scan import PriorityScanScheduler
from smartscan.schedulers.random_scan import RandomScanScheduler
from smartscan.schedulers.round_robin import RoundRobinScheduler


def make_states(n=5):
    states = []
    for i in range(n):
        fstart = i * 20e6
        states.append(BandState(band_id=i, freq_start=fstart, freq_end=fstart + 20e6))
    return states


def rx_config():
    return ReceiverConfig(instantaneous_bandwidth=20e6, dwell_time=0.01)


class TestRoundRobin:
    def test_cycles_sequentially(self):
        sched = RoundRobinScheduler(rx_config())
        states = make_states(3)
        ids = [sched.select_band(states, float(t)).band_id for t in range(6)]
        assert ids == [0, 1, 2, 0, 1, 2]

    def test_covers_all_bands(self):
        sched = RoundRobinScheduler(rx_config())
        states = make_states(5)
        ids = {sched.select_band(states, float(t)).band_id for t in range(5)}
        assert ids == {0, 1, 2, 3, 4}

    def test_name(self):
        assert RoundRobinScheduler(rx_config()).name == "round_robin"

    def test_reset(self):
        sched = RoundRobinScheduler(rx_config())
        states = make_states(3)
        sched.select_band(states, 0.0)
        sched.reset()
        assert sched.select_band(states, 0.0).band_id == 0


class TestRandomScan:
    def test_selects_valid_bands(self):
        sched = RandomScanScheduler(rx_config(), seed=42)
        states = make_states(5)
        for t in range(20):
            d = sched.select_band(states, float(t))
            assert 0 <= d.band_id < 5

    def test_reproducible(self):
        s1 = RandomScanScheduler(rx_config(), seed=42)
        s2 = RandomScanScheduler(rx_config(), seed=42)
        states = make_states(10)
        ids1 = [s1.select_band(states, float(t)).band_id for t in range(20)]
        ids2 = [s2.select_band(states, float(t)).band_id for t in range(20)]
        assert ids1 == ids2

    def test_covers_bands_eventually(self):
        sched = RandomScanScheduler(rx_config(), seed=42)
        states = make_states(5)
        ids = {sched.select_band(states, float(t)).band_id for t in range(100)}
        assert ids == {0, 1, 2, 3, 4}


class TestPriorityScan:
    def test_prefers_high_activity(self):
        cfg = SchedulerConfig(
            exploration_weight=10.0, recency_weight=0.1, uncertainty_weight=0.1,
            starvation_threshold=1000.0,
        )
        sched = PriorityScanScheduler(rx_config(), cfg)
        states = make_states(3)
        # Make band 2 high activity, and give all bands a recent scan
        for s in states:
            s.last_scan_time = 0.0
            s.observation_count = 5
            s.confidence = 0.5
        states[2].rolling_activity_prob = 0.95
        states[0].rolling_activity_prob = 0.1
        states[1].rolling_activity_prob = 0.1

        decision = sched.select_band(states, current_time=0.5)
        assert decision.band_id == 2

    def test_anti_starvation(self):
        cfg = SchedulerConfig(starvation_threshold=5.0)
        sched = PriorityScanScheduler(rx_config(), cfg)
        states = make_states(3)
        # Band 1 last scanned long ago → starved
        states[0].last_scan_time = 9.0
        states[1].last_scan_time = 1.0  # 9 sec ago at t=10 → starved
        states[2].last_scan_time = 9.5
        for s in states:
            s.observation_count = 5

        decision = sched.select_band(states, current_time=10.0)
        assert decision.band_id == 1
        assert "starvation" in decision.reason

    def test_no_band_starved_over_time(self):
        """Over many steps, every band should be visited."""
        cfg = SchedulerConfig(starvation_threshold=2.0)
        sched = PriorityScanScheduler(rx_config(), cfg)
        states = make_states(5)
        visited = set()
        t = 0.0
        for _ in range(50):
            d = sched.select_band(states, t)
            visited.add(d.band_id)
            # Simulate the scan updating last_scan_time
            states[d.band_id].last_scan_time = t
            states[d.band_id].observation_count += 1
            t += 1.0
        assert visited == {0, 1, 2, 3, 4}
