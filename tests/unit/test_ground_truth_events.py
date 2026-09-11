"""Frequency-aware, analytic ground-truth event generation."""

import pytest

from smartscan.core.models import EmitterConfig, EmitterType
from smartscan.simulation.emitters import create_emitter


def _emit(**kw):
    base = {"emitter_id": 0, "center_frequency": 100e6, "bandwidth": 5e6, "amplitude": 1.0, "snr_db": 25.0}
    base.update(kw)
    return create_emitter(EmitterConfig(**base), noise_power_dbm=-100.0)


class TestPeriodic:
    def test_one_event_per_cycle(self):
        e = _emit(emitter_type=EmitterType.PERIODIC_BURST, period=1.0, duty_cycle=0.3)
        evs = e.get_ground_truth_events(0.0, 5.0)
        assert len(evs) == 5
        assert evs[0].time_end - evs[0].time_start == pytest.approx(0.3, abs=1e-6)
        assert all(ev.freq_start == 100e6 - 2.5e6 for ev in evs)


class TestFrequencyHopping:
    def test_each_hop_is_its_own_event_with_its_own_frequency(self):
        e = _emit(emitter_type=EmitterType.FREQUENCY_HOPPING,
                  hop_frequencies=[100e6, 300e6], hop_interval=1.0)
        evs = e.get_ground_truth_events(0.0, 4.0)
        assert len(evs) == 4
        centers = [(ev.freq_start + ev.freq_end) / 2 for ev in evs]
        assert centers == [100e6, 300e6, 100e6, 300e6]      # alternating carriers
        assert evs[0].time_start == pytest.approx(0.0)
        assert evs[1].time_start == pytest.approx(1.0)


class TestRadarScan:
    def test_each_illumination_is_an_event(self):
        e = _emit(emitter_type=EmitterType.RADAR_SCAN, scan_period=2.0, beam_dwell=0.2)
        evs = e.get_ground_truth_events(0.0, 6.0)
        assert len(evs) == 3       # illuminations at t=0, 2, 4
        assert all(ev.time_end - ev.time_start == pytest.approx(0.2, abs=1e-6) for ev in evs)

    def test_agile_radar_hops_carrier_per_rotation(self):
        e = _emit(emitter_type=EmitterType.RADAR_SCAN, scan_period=2.0, beam_dwell=0.2,
                  hop_frequencies=[100e6, 400e6])
        evs = e.get_ground_truth_events(0.0, 6.0)
        centers = [(ev.freq_start + ev.freq_end) / 2 for ev in evs]
        assert centers == [100e6, 400e6, 100e6]


class TestContinuousAndWindow:
    def test_continuous_single_event(self):
        e = _emit(emitter_type=EmitterType.CONTINUOUS, end_time=3.0)
        evs = e.get_ground_truth_events(0.0, 10.0)
        assert len(evs) == 1
        assert evs[0].time_end == pytest.approx(3.0)

    def test_respects_start_time(self):
        e = _emit(emitter_type=EmitterType.CONTINUOUS, start_time=4.0)
        evs = e.get_ground_truth_events(0.0, 6.0)
        assert evs[0].time_start == pytest.approx(4.0)


class TestAnalyticMatchesSampledActivity:
    def test_intervals_agree_with_is_active(self):
        # analytic intervals should cover the times is_active() is True, away from
        # exact episode boundaries (where float rounding makes the edge ambiguous)
        e = _emit(emitter_type=EmitterType.PERIODIC_BURST, period=1.3, duty_cycle=0.4)
        ivs = e.activity_intervals(0.0, 5.0)
        edges = {round(x, 6) for a, b, _ in ivs for x in (a, b)}
        for t in [x * 0.01 for x in range(500)]:
            if any(abs(t - edge) < 6e-3 for edge in edges):
                continue  # skip boundary-adjacent samples
            inside = any(a <= t < b for a, b, _ in ivs)
            assert inside == e.is_active(t), f"mismatch at t={t}"
