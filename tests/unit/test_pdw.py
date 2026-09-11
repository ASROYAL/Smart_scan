"""Tests for the PDW (radar ESM) environment, loaders, and source."""

import numpy as np
import pytest

from smartscan.acquisition.pdw_source import PDWRFSource
from smartscan.simulation.pdw import (
    PDWEnvironment,
    PulseDescriptorWord,
    generate_synthetic_pdws,
    load_pdws_csv,
    save_pdws_csv,
)


def _pdw(toa, freq, eid=0, pw=1e-6, snr=25.0):
    return PulseDescriptorWord(toa=toa, frequency=freq, pulse_width=pw,
                               amplitude=1.0, emitter_id=eid, snr_db=snr)


class TestPDWModel:
    def test_toe(self):
        p = _pdw(1.0, 100e6, pw=2e-6)
        assert p.toe == pytest.approx(1.0 + 2e-6)


class TestSyntheticGeneration:
    def test_generates_pulses(self):
        pdws = generate_synthetic_pdws(num_emitters=4, duration=10.0, seed=1)
        assert len(pdws) > 0
        assert {p.emitter_id for p in pdws} == {0, 1, 2, 3}

    def test_reproducible(self):
        a = generate_synthetic_pdws(num_emitters=3, duration=5.0, seed=7)
        b = generate_synthetic_pdws(num_emitters=3, duration=5.0, seed=7)
        assert len(a) == len(b)
        assert a[0].toa == b[0].toa and a[-1].frequency == b[-1].frequency

    def test_within_band(self):
        pdws = generate_synthetic_pdws(num_emitters=5, duration=5.0,
                                       total_bw=1000e6, center=500e6, seed=2)
        for p in pdws:
            assert 0 < p.frequency < 1000e6


class TestCSVRoundTrip:
    def test_save_and_load(self, tmp_path):
        pdws = generate_synthetic_pdws(num_emitters=3, duration=4.0, seed=3)
        path = tmp_path / "pdws.csv"
        save_pdws_csv(pdws, path)
        loaded = load_pdws_csv(path)
        assert len(loaded) == len(pdws)
        assert loaded[0].emitter_id == pdws[0].emitter_id
        assert loaded[0].frequency == pytest.approx(pdws[0].frequency)

    def test_load_with_aliased_columns(self, tmp_path):
        path = tmp_path / "turing.csv"
        path.write_text("time,rf,pw,pa,label\n0.0,1.5e8,1e-6,1.0,3\n0.001,1.5e8,1e-6,1.0,3\n")
        pdws = load_pdws_csv(path)
        assert len(pdws) == 2
        assert pdws[0].frequency == 1.5e8
        assert pdws[0].emitter_id == 3

    def test_missing_required_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("foo,bar\n1,2\n")
        with pytest.raises(ValueError):
            load_pdws_csv(path)


class TestPDWEnvironment:
    def test_active_during_pulse(self):
        env = PDWEnvironment([_pdw(1.0, 100e6, eid=0)], dwell_tolerance=0.02)
        assert env.is_any_active_in_band(1.0, 98e6, 102e6) is True

    def test_inactive_outside_pulse(self):
        env = PDWEnvironment([_pdw(1.0, 100e6)], dwell_tolerance=0.02)
        assert env.is_any_active_in_band(5.0, 98e6, 102e6) is False

    def test_inactive_wrong_band(self):
        env = PDWEnvironment([_pdw(1.0, 100e6)], dwell_tolerance=0.02)
        assert env.is_any_active_in_band(1.0, 200e6, 210e6) is False

    def test_generate_samples_shape_and_type(self):
        env = PDWEnvironment([_pdw(1.0, 100e6)])
        env.set_time(1.0)
        s = env.generate_samples(100e6, 20e6, 2000)
        assert len(s) == 2000 and np.iscomplexobj(s)

    def test_signal_present_when_active(self):
        env = PDWEnvironment([_pdw(1.0, 100e6, snr=40.0)], noise_power_dbm=-100.0)
        env.set_time(1.0)
        on = env.generate_samples(100e6, 20e6, 4000)
        env.set_time(5.0)  # no pulse here
        off = env.generate_samples(100e6, 20e6, 4000)
        assert np.mean(np.abs(on) ** 2) > np.mean(np.abs(off) ** 2) * 5

    def test_clock(self):
        env = PDWEnvironment([_pdw(0.0, 100e6)])
        env.advance_time(0.5)
        assert env.time == pytest.approx(0.5)

    def test_ground_truth_events_grouped(self):
        # a burst of pulses close together = one illumination event
        pdws = [_pdw(1.0 + i * 1e-3, 100e6, eid=0) for i in range(10)]
        env = PDWEnvironment(pdws)
        events = env.get_all_ground_truth(0.0, 10.0)
        assert len(events) == 1
        assert events[0].emitter_id == 0

    def test_ground_truth_separates_distant_bursts(self):
        pdws = [_pdw(1.0, 100e6, eid=0), _pdw(3.0, 100e6, eid=0)]  # 2 s apart
        env = PDWEnvironment(pdws)
        events = env.get_all_ground_truth(0.0, 10.0)
        assert len(events) == 2


class TestPDWSource:
    def test_source_returns_samples_no_labels(self):
        env = PDWEnvironment([_pdw(1.0, 100e6)])
        src = PDWRFSource(env)
        samples, meta = src.read_samples(100e6, 20e6, 1000)
        assert len(samples) == 1000
        assert meta.center_frequency == 100e6
        # metadata must not carry ground-truth fields
        assert not any(k in type(meta).model_fields for k in ("emitter_id", "label", "active"))


class TestIntervalOverlap:
    def test_dwell_overlap_not_fixed_tolerance(self):
        # pulse at t=1.0 (1 us wide). A 2 ms dwell starting at 0.9990 overlaps it;
        # a dwell starting at 1.0020 (just after) does not.
        env = PDWEnvironment([_pdw(1.0, 100e6)], dwell_tolerance=0.0)
        assert env.is_any_active_in_band(0.9990, 98e6, 102e6, dwell=0.002) is True
        assert env.is_any_active_in_band(1.0020, 98e6, 102e6, dwell=0.002) is False

    def test_point_query_uses_min_window(self):
        env = PDWEnvironment([_pdw(1.0, 100e6)], dwell_tolerance=0.02)
        # dwell=0 -> falls back to the documented minimum integration window
        assert env.is_any_active_in_band(1.0, 98e6, 102e6, dwell=0.0) is True
