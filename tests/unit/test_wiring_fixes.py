"""Tests for the runtime-wiring and metric-correctness fixes."""

import numpy as np
import pytest

from smartscan.core.config import SmartScanConfig
from smartscan.core.models import BandObservation
from smartscan.evaluation.experiment import (
    build_and_run,
    build_and_run_recorded,
    compute_num_steps,
)


def _cfg(steps=250, bands=30):
    c = SmartScanConfig()
    c.simulation.num_steps = steps
    c.environment.num_bands = bands
    c.simulation.seed = 1
    return c


class TestDiscoveryRatioDistinct:
    def test_ratio_stays_bounded_and_valid(self):
        # a continuous emitter is re-detected many times; ratio must not inflate
        r = build_and_run(_cfg(steps=400), "adaptive", "sparse", seed=1).result
        assert 0.0 <= r.activity_discovery_ratio <= 1.0


class TestSharedRewardDirection:
    def _obs(self, detected=True, conf=0.9):
        return BandObservation(band_id=0, timestamp=0.0, detected=detected, confidence=conf,
                               peak_power_db=-40, avg_power_db=-60, noise_floor_db=-100,
                               estimated_snr_db=20)

    def test_corrected_penalty_direction(self):
        """Revisiting a JUST-seen band is penalised; returning to a long-neglected
        band earns a small bonus (opposite of the old long-gap penalty)."""
        from smartscan.core.config import SchedulerConfig
        from smartscan.rewards import compute_scan_reward
        cfg = SchedulerConfig()  # repeat/starvation scales default 0.02, threshold 10
        obs = self._obs()
        r_just = compute_scan_reward(cfg, obs, time_since_prev_scan=0.0, starvation_threshold=10)
        r_never = compute_scan_reward(cfg, obs, time_since_prev_scan=float("inf"), starvation_threshold=10)
        r_neglected = compute_scan_reward(cfg, obs, time_since_prev_scan=10.0, starvation_threshold=10)
        assert r_just < r_never < r_neglected

    def test_shared_reward_flows_into_schedulers(self):
        """The runner passes the shared reward into scheduler.update()."""
        from smartscan.core.config import ReceiverConfig, SchedulerConfig
        from smartscan.core.models import ScanDecision
        from smartscan.schedulers.bandit import UCB1BanditScheduler
        s = UCB1BanditScheduler(ReceiverConfig(), SchedulerConfig(), num_bands=3)
        d = ScanDecision(band_id=0, center_frequency=100e6, bandwidth=20e6,
                         dwell_time=0.01, timestamp=0.0)
        s.update(d, self._obs(), reward=0.9)
        assert s._mean_reward[0] == pytest.approx(0.9)   # used the passed reward


class TestFeatureExtractorConnected:
    def test_band_features_populated(self):
        out = build_and_run(_cfg(steps=200), "adaptive", "mixed", seed=1)
        assert len(out.artifacts.band_features) > 0
        from smartscan.core.models import BandFeatures
        any_feat = next(iter(out.artifacts.band_features.values()))
        assert isinstance(any_feat, BandFeatures)


class TestMLPredictorConnected:
    def test_runs_with_ml_predictor(self):
        r = build_and_run(_cfg(steps=300), "adaptive", "periodic", seed=1,
                          use_ml_predictor=True).result
        assert 0.0 <= r.prediction_accuracy <= 1.0
        assert r.num_steps == 300


class TestSqlitePersistence:
    def test_writes_sqlite_file(self, tmp_path):
        from smartscan.state.database import ScanDatabase
        cfg = _cfg(steps=150)
        cfg.database.backend = "sqlite"
        cfg.database.sqlite_path = str(tmp_path / "scan.db")
        build_and_run(cfg, "adaptive", "sparse", seed=1)
        assert (tmp_path / "scan.db").exists()
        with ScanDatabase(str(tmp_path / "scan.db")) as db:
            assert db.observation_count() > 0
            assert db.load_band_state(0) is not None


class TestRecordedEntryPoint:
    def test_recorded_run(self, tmp_path):
        from smartscan.acquisition.file_source import write_iq_file
        from smartscan.simulation.noise import generate_awgn, power_for_snr
        from smartscan.simulation.waveform import generate_tone
        rng = np.random.default_rng(0)
        iq = generate_awgn(400_000, -100.0, rng) + generate_tone(
            400_000, 20e6, frequency_offset=1e6, power_dbm=power_for_snr(-100.0, 25.0))
        path = tmp_path / "rec.c64"
        write_iq_file(iq, path, sample_rate=20e6, center_frequency=500e6)

        cfg = _cfg(steps=100, bands=20)
        out = build_and_run_recorded(cfg, "adaptive", path, seed=1)
        r = out.result
        assert r.num_steps == 100
        assert r.scenario_name == "recorded"
        assert 0.0 <= r.scan_hit_rate <= 1.0          # detection-side metric is real
        # no ground truth for a real recording
        assert r.activity_discovery_ratio == 0.0


class TestSimulationDurationDrivesSteps:
    def test_duration_used_when_set(self):
        cfg = SmartScanConfig()
        cfg.simulation.num_steps = None
        cfg.simulation.duration = 2.0
        per_scan = cfg.receiver.dwell_time + cfg.receiver.tuning_delay
        assert compute_num_steps(cfg, scenario_duration=10.0) == int(2.0 / per_scan)

    def test_scenario_duration_used_when_duration_none(self):
        cfg = SmartScanConfig()
        cfg.simulation.num_steps = None
        cfg.simulation.duration = None
        per_scan = cfg.receiver.dwell_time + cfg.receiver.tuning_delay
        assert compute_num_steps(cfg, scenario_duration=10.0) == int(10.0 / per_scan)
