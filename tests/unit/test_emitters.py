"""Tests for emitter activity models."""

import numpy as np
import pytest

from smartscan.core.models import EmitterConfig, EmitterType
from smartscan.simulation.emitters import (
    ContinuousEmitter,
    FrequencyHoppingEmitter,
    PeriodicBurstEmitter,
    RandomBurstEmitter,
    ScanLikeEmitter,
    create_emitter,
)


def make_config(**kwargs) -> EmitterConfig:
    defaults = {
        "emitter_id": 0, "emitter_type": EmitterType.CONTINUOUS,
        "center_frequency": 100e6, "bandwidth": 5e6, "amplitude": 1.0, "snr_db": 20.0,
    }
    defaults.update(kwargs)
    return EmitterConfig(**defaults)


class TestContinuousEmitter:
    def test_always_active(self):
        cfg = make_config(start_time=0.0, end_time=None)
        em = ContinuousEmitter(cfg, noise_power_dbm=-100.0)
        assert em.is_active(0.0)
        assert em.is_active(100.0)
        assert em.is_active(1e6)

    def test_respects_start_time(self):
        cfg = make_config(start_time=5.0)
        em = ContinuousEmitter(cfg, noise_power_dbm=-100.0)
        assert not em.is_active(4.9)
        assert em.is_active(5.0)
        assert em.is_active(10.0)

    def test_respects_end_time(self):
        cfg = make_config(start_time=0.0, end_time=10.0)
        em = ContinuousEmitter(cfg, noise_power_dbm=-100.0)
        assert em.is_active(5.0)
        assert not em.is_active(10.1)

    def test_frequency_is_constant(self):
        cfg = make_config(center_frequency=150e6)
        em = ContinuousEmitter(cfg, noise_power_dbm=-100.0)
        assert em.get_frequency(0.0) == 150e6
        assert em.get_frequency(100.0) == 150e6


class TestPeriodicBurstEmitter:
    def test_periodic_on_off(self):
        cfg = make_config(
            emitter_type=EmitterType.PERIODIC_BURST,
            period=2.0, duty_cycle=0.5,
        )
        em = PeriodicBurstEmitter(cfg, noise_power_dbm=-100.0)
        # ON during first half of period
        assert em.is_active(0.0)
        assert em.is_active(0.5)
        assert em.is_active(0.99)
        # OFF during second half
        assert not em.is_active(1.0)
        assert not em.is_active(1.5)
        # ON again in next cycle
        assert em.is_active(2.0)
        assert em.is_active(2.5)

    def test_duty_cycle_25_percent(self):
        cfg = make_config(
            emitter_type=EmitterType.PERIODIC_BURST,
            period=4.0, duty_cycle=0.25,
        )
        em = PeriodicBurstEmitter(cfg, noise_power_dbm=-100.0)
        assert em.is_active(0.0)
        assert em.is_active(0.99)
        assert not em.is_active(1.0)
        assert not em.is_active(3.5)

    def test_long_period(self):
        cfg = make_config(
            emitter_type=EmitterType.PERIODIC_BURST,
            period=100.0, duty_cycle=0.1,
        )
        em = PeriodicBurstEmitter(cfg, noise_power_dbm=-100.0)
        # ON for first 10 seconds
        assert em.is_active(5.0)
        # OFF for remaining 90
        assert not em.is_active(50.0)


class TestRandomBurstEmitter:
    def test_has_some_bursts(self):
        cfg = make_config(
            emitter_type=EmitterType.RANDOM_BURST,
            burst_rate=10.0, burst_duration=0.1,
        )
        em = RandomBurstEmitter(cfg, noise_power_dbm=-100.0, seed=42)
        # Check many time points — should find at least some active
        active_count = sum(1 for t in np.linspace(0, 10, 1000) if em.is_active(t))
        assert active_count > 0
        assert active_count < 1000  # not always active

    def test_reproducible(self):
        cfg = make_config(
            emitter_type=EmitterType.RANDOM_BURST,
            burst_rate=5.0, burst_duration=0.05,
        )
        em1 = RandomBurstEmitter(cfg, noise_power_dbm=-100.0, seed=42)
        em2 = RandomBurstEmitter(cfg, noise_power_dbm=-100.0, seed=42)
        times = np.linspace(0, 5, 500)
        active1 = [em1.is_active(t) for t in times]
        active2 = [em2.is_active(t) for t in times]
        assert active1 == active2

    def test_respects_time_window(self):
        cfg = make_config(
            emitter_type=EmitterType.RANDOM_BURST,
            burst_rate=5.0, burst_duration=0.05,
            start_time=5.0,
        )
        em = RandomBurstEmitter(cfg, noise_power_dbm=-100.0)
        assert not em.is_active(0.0)
        assert not em.is_active(4.9)


class TestFrequencyHoppingEmitter:
    def test_hops_between_frequencies(self):
        freqs = [100e6, 200e6, 300e6]
        cfg = make_config(
            emitter_type=EmitterType.FREQUENCY_HOPPING,
            hop_frequencies=freqs, hop_interval=1.0,
        )
        em = FrequencyHoppingEmitter(cfg, noise_power_dbm=-100.0)
        assert em.get_frequency(0.0) == 100e6
        assert em.get_frequency(1.0) == 200e6
        assert em.get_frequency(2.0) == 300e6
        assert em.get_frequency(3.0) == 100e6  # wraps

    def test_always_active(self):
        cfg = make_config(
            emitter_type=EmitterType.FREQUENCY_HOPPING,
            hop_frequencies=[100e6, 200e6], hop_interval=1.0,
        )
        em = FrequencyHoppingEmitter(cfg, noise_power_dbm=-100.0)
        assert em.is_active(0.5)
        assert em.is_active(5.5)


class TestScanLikeEmitter:
    def test_sweeps_with_duty_cycle(self):
        freqs = [100e6, 200e6]
        cfg = make_config(
            emitter_type=EmitterType.SCAN_LIKE,
            hop_frequencies=freqs, hop_interval=2.0, duty_cycle=0.5,
        )
        em = ScanLikeEmitter(cfg, noise_power_dbm=-100.0)
        # t=0: freq 100MHz, ON (phase 0 < 1.0)
        assert em.is_active(0.0)
        assert em.get_frequency(0.0) == 100e6
        # t=1.5: freq 100MHz, OFF (phase 1.5 > 1.0)
        assert not em.is_active(1.5)
        # t=2.0: freq 200MHz, ON
        assert em.is_active(2.0)
        assert em.get_frequency(2.0) == 200e6


class TestCreateEmitter:
    @pytest.mark.parametrize("emitter_type,cls", [
        (EmitterType.CONTINUOUS, ContinuousEmitter),
        (EmitterType.PERIODIC_BURST, PeriodicBurstEmitter),
        (EmitterType.RANDOM_BURST, RandomBurstEmitter),
        (EmitterType.FREQUENCY_HOPPING, FrequencyHoppingEmitter),
        (EmitterType.SCAN_LIKE, ScanLikeEmitter),
    ])
    def test_factory(self, emitter_type, cls):
        cfg = make_config(
            emitter_type=emitter_type,
            hop_frequencies=[100e6, 200e6], hop_interval=1.0,
            period=1.0, duty_cycle=0.5,
            burst_rate=1.0, burst_duration=0.1,
        )
        em = create_emitter(cfg, noise_power_dbm=-100.0)
        assert isinstance(em, cls)


class TestEmitterGenerateSamples:
    def test_returns_none_when_inactive(self):
        cfg = make_config(start_time=10.0)
        em = ContinuousEmitter(cfg, noise_power_dbm=-100.0)
        result = em.generate_samples(
            time=0.0, num_samples=100, sample_rate=1e6,
            center_frequency=100e6, bandwidth=20e6,
        )
        assert result is None

    def test_returns_none_when_out_of_band(self):
        cfg = make_config(center_frequency=500e6)
        em = ContinuousEmitter(cfg, noise_power_dbm=-100.0)
        result = em.generate_samples(
            time=0.0, num_samples=100, sample_rate=1e6,
            center_frequency=100e6, bandwidth=20e6,
        )
        assert result is None

    def test_returns_samples_when_active_and_in_band(self):
        cfg = make_config(center_frequency=105e6)
        em = ContinuousEmitter(cfg, noise_power_dbm=-100.0)
        result = em.generate_samples(
            time=0.0, num_samples=1000, sample_rate=20e6,
            center_frequency=100e6, bandwidth=20e6,
        )
        assert result is not None
        assert len(result) == 1000
        assert np.iscomplexobj(result)
        assert np.mean(np.abs(result) ** 2) > 0
