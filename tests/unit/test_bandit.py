"""Tests for bandit schedulers."""

import numpy as np
import pytest

from smartscan.core.config import ReceiverConfig, SchedulerConfig
from smartscan.core.models import BandObservation, BandState, ScanDecision
from smartscan.schedulers.bandit import ThompsonSamplingScheduler, UCB1BanditScheduler


def make_states(n=5):
    return [
        BandState(band_id=i, freq_start=i * 20e6, freq_end=(i + 1) * 20e6)
        for i in range(n)
    ]


def make_obs(band_id, detected):
    return BandObservation(
        band_id=band_id, timestamp=0.0, detected=detected, confidence=0.8,
        peak_power_db=-50.0, avg_power_db=-60.0,
        noise_floor_db=-100.0, estimated_snr_db=20.0,
    )


def rx_cfg():
    return ReceiverConfig(instantaneous_bandwidth=20e6, dwell_time=0.01)


class TestUCB1:
    def test_tries_all_arms_first(self):
        sched = UCB1BanditScheduler(rx_cfg(), SchedulerConfig(), num_bands=5)
        states = make_states(5)
        visited = []
        for t in range(5):
            d = sched.select_band(states, float(t))
            visited.append(d.band_id)
            sched.update(d, make_obs(d.band_id, False))
        assert set(visited) == {0, 1, 2, 3, 4}

    def test_exploits_high_reward_arm(self):
        sched = UCB1BanditScheduler(
            rx_cfg(), SchedulerConfig(ucb_c=0.5), num_bands=3,
        )
        states = make_states(3)
        # Train: band 1 always detects, others never
        for t in range(60):
            d = sched.select_band(states, float(t))
            detected = (d.band_id == 1)
            sched.update(d, make_obs(d.band_id, detected))

        # Now band 1 should be selected most often in a fresh batch
        counts = {0: 0, 1: 0, 2: 0}
        for t in range(30):
            d = sched.select_band(states, float(t))
            counts[d.band_id] += 1
            sched.update(d, make_obs(d.band_id, d.band_id == 1))
        assert counts[1] > counts[0]
        assert counts[1] > counts[2]

    def test_reset(self):
        sched = UCB1BanditScheduler(rx_cfg(), SchedulerConfig(), num_bands=3)
        states = make_states(3)
        d = sched.select_band(states, 0.0)
        sched.update(d, make_obs(d.band_id, True))
        sched.reset()
        assert sched._total == 0
        assert np.all(sched._counts == 0)

    def test_name(self):
        assert UCB1BanditScheduler(rx_cfg(), SchedulerConfig(), 3).name == "bandit_ucb"


class TestThompsonSampling:
    def test_selects_valid_bands(self):
        sched = ThompsonSamplingScheduler(rx_cfg(), num_bands=5, seed=42)
        states = make_states(5)
        for t in range(20):
            d = sched.select_band(states, float(t))
            assert 0 <= d.band_id < 5
            sched.update(d, make_obs(d.band_id, False))

    def test_learns_best_arm(self):
        sched = ThompsonSamplingScheduler(rx_cfg(), num_bands=3, seed=42)
        states = make_states(3)
        # Band 2 always active
        for t in range(100):
            d = sched.select_band(states, float(t))
            sched.update(d, make_obs(d.band_id, d.band_id == 2))

        # Posterior for band 2 should have high alpha
        assert sched._alpha[2] > sched._alpha[0]
        assert sched._alpha[2] > sched._alpha[1]

    def test_reproducible(self):
        s1 = ThompsonSamplingScheduler(rx_cfg(), num_bands=5, seed=7)
        s2 = ThompsonSamplingScheduler(rx_cfg(), num_bands=5, seed=7)
        states = make_states(5)
        ids1, ids2 = [], []
        for t in range(20):
            d1 = s1.select_band(states, float(t))
            d2 = s2.select_band(states, float(t))
            ids1.append(d1.band_id)
            ids2.append(d2.band_id)
            s1.update(d1, make_obs(d1.band_id, False))
            s2.update(d2, make_obs(d2.band_id, False))
        assert ids1 == ids2
