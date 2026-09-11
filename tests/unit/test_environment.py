"""Tests for the RF environment."""

import numpy as np
import pytest

from smartscan.core.models import EmitterConfig, EmitterType
from smartscan.simulation.environment import RFEnvironment
from smartscan.simulation.noise import watts_to_dbm


def make_env(emitters=None, **kwargs):
    if emitters is None:
        emitters = [
            EmitterConfig(
                emitter_id=0, emitter_type=EmitterType.CONTINUOUS,
                center_frequency=100e6, bandwidth=5e6, amplitude=1.0, snr_db=20.0,
            )
        ]
    defaults = {"noise_power_dbm": -100.0, "sample_rate": 20e6, "seed": 42}
    defaults.update(kwargs)
    return RFEnvironment(emitters, **defaults)


class TestEnvironmentBasic:
    def test_generate_returns_complex(self):
        env = make_env()
        samples = env.generate_samples(100e6, 20e6, 1000)
        assert np.iscomplexobj(samples)
        assert len(samples) == 1000

    def test_noise_only(self):
        env = make_env(emitters=[])
        samples = env.generate_samples(100e6, 20e6, 10000)
        power = np.mean(np.abs(samples) ** 2)
        power_dbm = watts_to_dbm(power)
        # Should be close to noise floor
        assert power_dbm == pytest.approx(-100.0, abs=2.0)

    def test_signal_above_noise(self):
        env = make_env()
        # Observe at the emitter frequency
        signal_samples = env.generate_samples(100e6, 20e6, 10000)
        signal_power = np.mean(np.abs(signal_samples) ** 2)

        # Observe away from emitter
        env2 = make_env(emitters=[])
        noise_samples = env2.generate_samples(100e6, 20e6, 10000)
        noise_power = np.mean(np.abs(noise_samples) ** 2)

        assert signal_power > noise_power * 5  # signal should be much stronger

    def test_out_of_band_observation(self):
        env = make_env()  # emitter at 100 MHz
        # Observe at 500 MHz — no signal there
        samples = env.generate_samples(500e6, 20e6, 10000)
        power = np.mean(np.abs(samples) ** 2)
        power_dbm = watts_to_dbm(power)
        # Should be just noise
        assert power_dbm == pytest.approx(-100.0, abs=3.0)


class TestEnvironmentGroundTruth:
    def test_dwell_interval_detects_burst_that_starts_after_scan_start(self):
        emitter = EmitterConfig(
            emitter_id=0, emitter_type=EmitterType.PERIODIC_BURST,
            center_frequency=100e6, bandwidth=5e6, amplitude=1.0,
            snr_db=20.0, period=1.0, duty_cycle=0.1,
        )
        env = make_env(emitters=[emitter])
        assert not env.is_any_active_in_band(0.95, 90e6, 110e6)
        assert env.is_any_active_in_band(0.95, 90e6, 110e6, dwell=0.1)
        samples = env.generate_samples(100e6, 20e6, 2000, time=0.99995)
        assert np.mean(np.abs(samples[-500:]) ** 2) > np.mean(np.abs(samples[:500]) ** 2)

    def test_active_emitters(self):
        env = make_env()
        active = env.get_active_emitters(0.0)
        assert len(active) == 1
        assert active[0].emitter_id == 0

    def test_ground_truth_at_frequency(self):
        env = make_env()
        events = env.get_ground_truth_at(0.0, 90e6, 110e6)
        assert len(events) == 1
        assert events[0].emitter_id == 0

    def test_no_ground_truth_out_of_band(self):
        env = make_env()
        events = env.get_ground_truth_at(0.0, 400e6, 420e6)
        assert len(events) == 0

    def test_is_any_active_in_band(self):
        env = make_env()
        assert env.is_any_active_in_band(0.0, 90e6, 110e6)
        assert not env.is_any_active_in_band(0.0, 400e6, 420e6)


class TestEnvironmentTime:
    def test_time_starts_at_zero(self):
        env = make_env()
        assert env.time == 0.0

    def test_advance_time(self):
        env = make_env()
        env.advance_time(1.5)
        assert env.time == pytest.approx(1.5)

    def test_set_time(self):
        env = make_env()
        env.set_time(10.0)
        assert env.time == 10.0


class TestEnvironmentMultipleEmitters:
    def test_two_emitters_different_freqs(self):
        emitters = [
            EmitterConfig(
                emitter_id=0, emitter_type=EmitterType.CONTINUOUS,
                center_frequency=100e6, bandwidth=5e6, amplitude=1.0, snr_db=20.0,
            ),
            EmitterConfig(
                emitter_id=1, emitter_type=EmitterType.CONTINUOUS,
                center_frequency=200e6, bandwidth=5e6, amplitude=1.0, snr_db=20.0,
            ),
        ]
        env = make_env(emitters=emitters)

        # Both should be visible in ground truth
        events = env.get_ground_truth_at(0.0, 50e6, 250e6)
        assert len(events) == 2

        # Only one in narrower window
        events_100 = env.get_ground_truth_at(0.0, 90e6, 110e6)
        assert len(events_100) == 1
        assert events_100[0].emitter_id == 0

    def test_periodic_emitter_in_environment(self):
        emitters = [
            EmitterConfig(
                emitter_id=0, emitter_type=EmitterType.PERIODIC_BURST,
                center_frequency=100e6, bandwidth=5e6, amplitude=1.0,
                snr_db=20.0, period=2.0, duty_cycle=0.5,
            ),
        ]
        env = make_env(emitters=emitters)

        # Active during ON phase
        assert env.is_any_active_in_band(0.0, 90e6, 110e6)
        # Inactive during OFF phase
        assert not env.is_any_active_in_band(1.5, 90e6, 110e6)


class TestEnvironmentReproducibility:
    def test_same_seed_same_noise(self):
        env1 = make_env(seed=42)
        env2 = make_env(seed=42)
        s1 = env1.generate_samples(500e6, 20e6, 100)
        s2 = env2.generate_samples(500e6, 20e6, 100)
        np.testing.assert_array_equal(s1, s2)
