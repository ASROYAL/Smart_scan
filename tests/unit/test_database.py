"""Tests for SQLite persistence."""

import pytest

from smartscan.core.models import BandObservation, BandState
from smartscan.state.database import ScanDatabase


class TestScanDatabase:
    def test_in_memory_creation(self):
        with ScanDatabase(":memory:") as db:
            assert db.observation_count() == 0

    def test_save_and_load_band_state(self):
        with ScanDatabase(":memory:") as db:
            state = BandState(
                band_id=5, freq_start=100e6, freq_end=120e6,
                hit_count=3, miss_count=7, observation_count=10,
                rolling_activity_prob=0.4, avg_snr_db=15.0,
            )
            db.save_band_state(state)
            loaded = db.load_band_state(5)
            assert loaded is not None
            assert loaded.band_id == 5
            assert loaded.hit_count == 3
            assert loaded.rolling_activity_prob == pytest.approx(0.4)

    def test_load_missing_band_state(self):
        with ScanDatabase(":memory:") as db:
            assert db.load_band_state(99) is None

    def test_upsert_band_state(self):
        with ScanDatabase(":memory:") as db:
            state = BandState(band_id=0, freq_start=0, freq_end=20e6, hit_count=1)
            db.save_band_state(state)
            state.hit_count = 5
            db.save_band_state(state)
            loaded = db.load_band_state(0)
            assert loaded.hit_count == 5

    def test_save_and_load_observations(self):
        with ScanDatabase(":memory:") as db:
            for i in range(5):
                obs = BandObservation(
                    band_id=1, timestamp=float(i), detected=(i % 2 == 0),
                    confidence=0.5, peak_power_db=-50.0, avg_power_db=-60.0,
                    noise_floor_db=-100.0, estimated_snr_db=10.0,
                )
                db.save_observation(obs)
            obs_list = db.load_observations(1)
            assert len(obs_list) == 5
            assert obs_list[0].timestamp == 0.0
            assert db.observation_count() == 5

    def test_observations_ordered_by_time(self):
        with ScanDatabase(":memory:") as db:
            for t in [3.0, 1.0, 2.0]:
                obs = BandObservation(
                    band_id=0, timestamp=t, detected=True, confidence=0.5,
                    peak_power_db=-50.0, avg_power_db=-60.0,
                    noise_floor_db=-100.0, estimated_snr_db=10.0,
                )
                db.save_observation(obs)
            obs_list = db.load_observations(0)
            times = [o.timestamp for o in obs_list]
            assert times == [1.0, 2.0, 3.0]

    def test_file_backend(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = ScanDatabase(db_path)
        state = BandState(band_id=0, freq_start=0, freq_end=20e6, hit_count=2)
        db.save_band_state(state)
        db.close()

        # Reopen and verify persistence
        db2 = ScanDatabase(db_path)
        loaded = db2.load_band_state(0)
        assert loaded.hit_count == 2
        db2.close()
