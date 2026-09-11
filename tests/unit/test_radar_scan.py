"""Tests for the rotating-antenna (spatially scanning) radar emitter."""

import numpy as np

from smartscan.core.models import EmitterConfig, EmitterType
from smartscan.simulation.emitters import RadarScanEmitter, create_emitter
from smartscan.simulation.scenarios import ALL_SCENARIOS, get_scenario


def _cfg(**kw):
    base = {
        "emitter_id": 0, "emitter_type": EmitterType.RADAR_SCAN,
        "center_frequency": 100e6, "bandwidth": 5e6, "amplitude": 1.0, "snr_db": 25.0,
        "scan_period": 2.0, "beam_dwell": 0.2,
    }
    base.update(kw)
    return EmitterConfig(**base)


class TestRadarScanActivity:
    def test_illuminated_at_start_of_rotation(self):
        e = RadarScanEmitter(_cfg(), noise_power_dbm=-100.0)
        assert e.is_active(0.0) is True        # main beam on
        assert e.is_active(0.1) is True        # still within beam_dwell (0.2)

    def test_dark_between_rotations(self):
        e = RadarScanEmitter(_cfg(), noise_power_dbm=-100.0)
        assert e.is_active(0.5) is False       # beam elsewhere
        assert e.is_active(1.9) is False

    def test_reilluminates_next_rotation(self):
        e = RadarScanEmitter(_cfg(), noise_power_dbm=-100.0)
        # scan_period = 2.0, so beam points back at us at t=2.0
        assert e.is_active(2.0) is True
        assert e.is_active(2.15) is True
        assert e.is_active(2.3) is False

    def test_low_duty_cycle(self):
        """Over one rotation the radar is visible only beam_dwell/scan_period of the time."""
        e = RadarScanEmitter(_cfg(scan_period=4.0, beam_dwell=0.2), noise_power_dbm=-100.0)
        ts = np.linspace(0, 4.0, 4001, endpoint=False)
        frac = np.mean([e.is_active(float(t)) for t in ts])
        assert abs(frac - 0.05) < 0.01        # 0.2 / 4.0

    def test_respects_start_time(self):
        e = RadarScanEmitter(_cfg(start_time=5.0), noise_power_dbm=-100.0)
        assert e.is_active(0.0) is False
        assert e.is_active(5.0) is True


class TestRadarScanFrequencyAgility:
    def test_fixed_frequency_without_hops(self):
        e = RadarScanEmitter(_cfg(), noise_power_dbm=-100.0)
        assert e.get_frequency(0.0) == 100e6
        assert e.get_frequency(2.0) == 100e6

    def test_hops_per_rotation(self):
        e = RadarScanEmitter(
            _cfg(hop_frequencies=[100e6, 200e6], scan_period=2.0),
            noise_power_dbm=-100.0,
        )
        assert e.get_frequency(0.0) == 100e6     # rotation 0
        assert e.get_frequency(2.0) == 200e6     # rotation 1
        assert e.get_frequency(4.0) == 100e6     # rotation 2 wraps


class TestRadarScanFactory:
    def test_factory_builds_radar(self):
        e = create_emitter(_cfg(), noise_power_dbm=-100.0)
        assert isinstance(e, RadarScanEmitter)

    def test_generates_iq_when_illuminated(self):
        e = create_emitter(_cfg(), noise_power_dbm=-100.0)
        sig = e.generate_samples(
            time=0.0, num_samples=1000, sample_rate=20e6,
            center_frequency=100e6, bandwidth=20e6,
        )
        assert sig is not None and len(sig) == 1000

    def test_no_iq_when_dark(self):
        e = create_emitter(_cfg(), noise_power_dbm=-100.0)
        sig = e.generate_samples(
            time=0.5, num_samples=1000, sample_rate=20e6,
            center_frequency=100e6, bandwidth=20e6,
        )
        assert sig is None


class TestRadarScenario:
    def test_registered(self):
        assert "radar_scan" in ALL_SCENARIOS

    def test_builds(self):
        s = get_scenario("radar_scan", seed=1, num_bands=50, total_bw=1000e6, center=500e6)
        assert len(s.emitters) == 4
        assert all(e.emitter_type == EmitterType.RADAR_SCAN for e in s.emitters)

    def test_has_ground_truth_events(self):
        from smartscan.simulation.environment import RFEnvironment
        s = get_scenario("radar_scan", seed=1)
        env = RFEnvironment(s.emitters, noise_power_dbm=-100.0, sample_rate=20e6, seed=1)
        events = env.get_all_ground_truth(0.0, 10.0)
        assert len(events) > 0   # radars do illuminate within the window
