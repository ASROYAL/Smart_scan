"""Tests for the receiver model."""

import numpy as np
import pytest

from smartscan.acquisition.simulator_source import SimulatedRFSource
from smartscan.core.config import ReceiverConfig
from smartscan.core.models import AcquisitionMeta, EmitterConfig, EmitterType
from smartscan.receiver.receiver import Receiver
from smartscan.simulation.environment import RFEnvironment


def make_receiver(**cfg_kwargs):
    emitters = [
        EmitterConfig(
            emitter_id=0, emitter_type=EmitterType.CONTINUOUS,
            center_frequency=100e6, bandwidth=5e6, amplitude=1.0, snr_db=20.0,
        ),
    ]
    env = RFEnvironment(emitters, noise_power_dbm=-100.0, sample_rate=20e6, seed=42)
    src = SimulatedRFSource(env)
    defaults = dict(
        instantaneous_bandwidth=20e6, sample_rate=20e6,
        dwell_time=0.001, tuning_delay=0.0005,
    )
    defaults.update(cfg_kwargs)
    cfg = ReceiverConfig(**defaults)
    return Receiver(src, cfg), env


class TestReceiverObserve:
    def test_returns_samples_and_meta(self):
        rx, _ = make_receiver()
        samples, meta = rx.observe(100e6)
        assert np.iscomplexobj(samples)
        assert isinstance(meta, AcquisitionMeta)

    def test_num_samples_from_dwell(self):
        rx, _ = make_receiver(sample_rate=20e6, dwell_time=0.001)
        samples, _ = rx.observe(100e6)
        # 20e6 * 0.001 = 20000 samples
        assert len(samples) == 20000

    def test_bandwidth_clamped_to_instantaneous(self):
        rx, _ = make_receiver(instantaneous_bandwidth=20e6)
        # Request 100 MHz but receiver can only do 20 MHz
        _, meta = rx.observe(100e6, bandwidth=100e6)
        assert meta.bandwidth == 20e6

    def test_scan_count_increments(self):
        rx, _ = make_receiver()
        assert rx.scan_count == 0
        rx.observe(100e6)
        assert rx.scan_count == 1
        rx.observe(200e6)
        assert rx.scan_count == 2


class TestReceiverTiming:
    def test_dwell_time_accumulates(self):
        rx, _ = make_receiver(dwell_time=0.001)
        rx.observe(100e6)
        rx.observe(100e6)
        assert rx.total_dwell_time == pytest.approx(0.002)

    def test_tuning_delay_on_frequency_change(self):
        rx, _ = make_receiver(tuning_delay=0.0005)
        rx.observe(100e6)  # first tune
        rx.observe(200e6)  # retune
        assert rx.total_tuning_time == pytest.approx(0.001)  # two tunes

    def test_no_retune_same_frequency(self):
        rx, _ = make_receiver(tuning_delay=0.0005)
        rx.observe(100e6)  # first tune
        rx.observe(100e6)  # same freq, no retune
        assert rx.total_tuning_time == pytest.approx(0.0005)  # one tune only

    def test_clock_advances(self):
        rx, env = make_receiver(dwell_time=0.001, tuning_delay=0.0005)
        assert env.time == 0.0
        rx.observe(100e6)
        # time = tuning + dwell = 0.0005 + 0.001 = 0.0015
        assert env.time == pytest.approx(0.0015)


class TestReceiverConstraint:
    def test_only_observes_tuned_window(self):
        """Receiver at 500 MHz should see noise; emitter is at 100 MHz."""
        rx, _ = make_receiver()
        samples, _ = rx.observe(500e6)
        power = np.mean(np.abs(samples) ** 2)
        # Just noise — should be low
        from smartscan.simulation.noise import watts_to_dbm
        assert watts_to_dbm(power) == pytest.approx(-100.0, abs=3.0)

    def test_observes_signal_when_tuned_correctly(self):
        rx, _ = make_receiver()
        samples, _ = rx.observe(100e6)
        power = np.mean(np.abs(samples) ** 2)
        from smartscan.simulation.noise import watts_to_dbm
        # Signal present — should be well above noise
        assert watts_to_dbm(power) > -95.0
