"""Tests for band state management and history."""

import pytest

from smartscan.core.models import BandObservation
from smartscan.state.band_history import BandHistory
from smartscan.state.spectrum_state import SpectrumStateManager


def make_obs(band_id=0, t=0.0, detected=False, snr=0.0, power=-90.0):
    return BandObservation(
        band_id=band_id, timestamp=t, detected=detected, confidence=0.5,
        peak_power_db=power + 10, avg_power_db=power,
        noise_floor_db=-100.0, estimated_snr_db=snr,
    )


class TestBandHistory:
    def test_add_and_length(self):
        h = BandHistory(band_id=0)
        h.add(make_obs(detected=True, t=1.0))
        assert len(h) == 1

    def test_detection_times_tracked(self):
        h = BandHistory(band_id=0)
        h.add(make_obs(detected=True, t=1.0))
        h.add(make_obs(detected=False, t=2.0))
        h.add(make_obs(detected=True, t=3.0))
        assert h.detection_times == [1.0, 3.0]

    def test_recent_hits(self):
        h = BandHistory(band_id=0)
        for i in range(10):
            h.add(make_obs(detected=(i % 2 == 0), t=float(i)))
        assert h.recent_hits(10) == 5

    def test_max_records_enforced(self):
        h = BandHistory(band_id=0, max_records=5)
        for i in range(10):
            h.add(make_obs(t=float(i)))
        assert len(h) == 5

    def test_inter_detection_intervals(self):
        h = BandHistory(band_id=0)
        h.add(make_obs(detected=True, t=1.0))
        h.add(make_obs(detected=True, t=3.0))
        h.add(make_obs(detected=True, t=6.0))
        assert h.inter_detection_intervals() == [2.0, 3.0]


class TestSpectrumStateManager:
    def test_creates_num_bands(self):
        sm = SpectrumStateManager(
            num_bands=50, total_bandwidth=1000e6, center_frequency=500e6,
        )
        assert len(sm.all_states()) == 50

    def test_band_widths(self):
        sm = SpectrumStateManager(
            num_bands=50, total_bandwidth=1000e6, center_frequency=500e6,
        )
        assert sm.band_width == pytest.approx(20e6)

    def test_bands_span_spectrum(self):
        sm = SpectrumStateManager(
            num_bands=10, total_bandwidth=1000e6, center_frequency=500e6,
        )
        states = sm.all_states()
        assert states[0].freq_start == pytest.approx(0.0)
        assert states[-1].freq_end == pytest.approx(1000e6)

    def test_band_id_for_frequency(self):
        sm = SpectrumStateManager(
            num_bands=10, total_bandwidth=1000e6, center_frequency=500e6,
        )
        # Band 0: 0-100 MHz, band 5: 500-600 MHz
        assert sm.band_id_for_frequency(50e6) == 0
        assert sm.band_id_for_frequency(550e6) == 5

    def test_frequency_clamping(self):
        sm = SpectrumStateManager(
            num_bands=10, total_bandwidth=1000e6, center_frequency=500e6,
        )
        assert sm.band_id_for_frequency(-100e6) == 0
        assert sm.band_id_for_frequency(2000e6) == 9

    def test_update_increments_counts(self):
        sm = SpectrumStateManager(
            num_bands=10, total_bandwidth=1000e6, center_frequency=500e6,
        )
        sm.update(0, make_obs(band_id=0, detected=True, t=1.0))
        state = sm.get_state(0)
        assert state.hit_count == 1
        assert state.observation_count == 1
        assert state.last_detection_time == 1.0

    def test_update_miss(self):
        sm = SpectrumStateManager(
            num_bands=10, total_bandwidth=1000e6, center_frequency=500e6,
        )
        sm.update(0, make_obs(band_id=0, detected=False, t=1.0))
        state = sm.get_state(0)
        assert state.miss_count == 1
        assert state.last_detection_time < 0  # never detected

    def test_ewma_activity_updates(self):
        sm = SpectrumStateManager(
            num_bands=10, total_bandwidth=1000e6, center_frequency=500e6,
            ewma_alpha=0.5,
        )
        state = sm.get_state(0)
        assert state.rolling_activity_prob == pytest.approx(0.5)  # starts neutral
        sm.update(0, make_obs(band_id=0, detected=True, t=1.0))
        # 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        assert sm.get_state(0).rolling_activity_prob == pytest.approx(0.75)

    def test_bands_in_window(self):
        sm = SpectrumStateManager(
            num_bands=10, total_bandwidth=1000e6, center_frequency=500e6,
        )
        # 20 MHz window at 550 MHz covers band 5 (500-600)
        ids = sm.bands_in_window(550e6, 20e6)
        assert 5 in ids

    def test_reset(self):
        sm = SpectrumStateManager(
            num_bands=10, total_bandwidth=1000e6, center_frequency=500e6,
        )
        sm.update(0, make_obs(band_id=0, detected=True, t=1.0))
        sm.reset()
        assert sm.get_state(0).hit_count == 0
