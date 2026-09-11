"""Tests for the Q-learning scheduler, ML predictor, and periodicity wiring."""
import pytest

from smartscan.core.config import ReceiverConfig, SchedulerConfig, SmartScanConfig
from smartscan.core.models import BandObservation, BandState, SchedulerType
from smartscan.evaluation.experiment import build_and_run
from smartscan.prediction.ml_predictor import MLActivityPredictor
from smartscan.schedulers.factory import ALL_SCHEDULER_TYPES, create_scheduler
from smartscan.schedulers.qlearning import QLearningScheduler
from smartscan.state.spectrum_state import SpectrumStateManager


def _obs(bid, t, detected, snr=20.0, conf=0.9):
    return BandObservation(band_id=bid, timestamp=t, detected=detected, confidence=conf,
                           peak_power_db=-40.0, avg_power_db=-60.0,
                           noise_floor_db=-100.0, estimated_snr_db=snr)


class TestQLearningScheduler:
    def test_registered(self):
        assert SchedulerType.Q_LEARNING in ALL_SCHEDULER_TYPES

    def test_factory(self):
        s = create_scheduler(SchedulerType.Q_LEARNING, ReceiverConfig(),
                             SchedulerConfig(), num_bands=10, seed=1)
        assert isinstance(s, QLearningScheduler)
        assert s.name == "q_learning"

    def test_selects_valid_band(self):
        s = QLearningScheduler(ReceiverConfig(), SchedulerConfig(), num_bands=5, seed=1)
        states = [BandState(band_id=i, freq_start=i*1e6+1e6, freq_end=i*1e6+2e6) for i in range(5)]
        d = s.select_band(states, 0.0)
        assert 0 <= d.band_id < 5

    def test_learns_to_value_active_context(self):
        """After learning, the active-looking context is valued above the idle one."""
        s = QLearningScheduler(ReceiverConfig(), SchedulerConfig(q_epsilon=0.3),
                               num_bands=2, seed=1)
        active = BandState(band_id=0, freq_start=1e6, freq_end=2e6,
                           rolling_activity_prob=0.9, hit_count=8, observation_count=10)
        idle = BandState(band_id=1, freq_start=2e6, freq_end=3e6,
                         rolling_activity_prob=0.0, hit_count=0, observation_count=10)
        for t in range(400):
            d = s.select_band([active, idle], float(t))
            detected = (d.band_id == 0)   # only band 0 is truly active
            s.update(d, _obs(d.band_id, float(t), detected), reward=float(detected))
        from smartscan.schedulers.qlearning import _context_key
        ka = _context_key(active, 100.0, s._time_scale)
        ki = _context_key(idle, 100.0, s._time_scale)
        assert s._q.get(ka, 0.0) > s._q.get(ki, 0.0)

    def test_runs_end_to_end(self):
        cfg = SmartScanConfig()
        cfg.simulation.num_steps = 300
        cfg.environment.num_bands = 30
        r = build_and_run(cfg, "q_learning", "periodic", seed=1).result
        assert r.num_steps == 300
        assert 0.0 <= r.activity_discovery_ratio <= 1.0


class TestMLPredictor:
    def test_cold_start_uses_rolling(self):
        p = MLActivityPredictor()
        bs = BandState(band_id=0, freq_start=1e6, freq_end=2e6, rolling_activity_prob=0.7)
        assert p.predict_activity_probability(bs, 1.0) == 0.7

    def test_learns_active_vs_idle(self):
        p = MLActivityPredictor()
        active = BandState(band_id=0, freq_start=1e6, freq_end=2e6,
                           rolling_activity_prob=0.9, activity_ratio=0.9,
                           avg_snr_db=25.0, confidence=0.9)
        idle = BandState(band_id=1, freq_start=2e6, freq_end=3e6,
                         rolling_activity_prob=0.05, activity_ratio=0.05,
                         avg_snr_db=0.0, confidence=0.9)
        for t in range(80):
            p.update(active, True, float(t))
            p.update(idle, False, float(t))
        pa = p.predict_activity_probability(active, 100.0)
        pi = p.predict_activity_probability(idle, 100.0)
        assert pa > pi   # learned to separate active from idle


class TestPeriodicityWiring:
    def test_estimated_period_populated(self):
        """Regular detections should make the state manager estimate a period."""
        sm = SpectrumStateManager(num_bands=4, total_bandwidth=100e6, center_frequency=50e6)
        # feed detections every 1.0 s on band 0
        for k in range(8):
            sm.update(0, _obs(0, k * 1.0, detected=True))
        state = sm.get_state(0)
        assert state.estimated_period is not None
        assert abs(state.estimated_period - 1.0) < 0.2


class TestEpisodePeriodicity:
    def test_camping_burst_is_one_episode(self):
        """Many detections a few ms apart (a scheduler camping during one ON-window)
        must NOT be read as a tiny period."""
        from smartscan.prediction.periodicity import estimate_periodicity, group_into_episodes
        # three ON-windows at t=1,3,5; each caught 5 times ~10 ms apart
        times = []
        for base in (1.0, 3.0, 5.0):
            times += [base + i * 0.01 for i in range(5)]
        assert len(group_into_episodes(times, episode_gap=0.05)) == 3
        est = estimate_periodicity(times, current_time=6.0, episode_gap=0.05)
        assert est.estimated_period == pytest.approx(2.0, abs=0.05)

    def test_single_burst_gives_no_period(self):
        from smartscan.prediction.periodicity import estimate_periodicity
        times = [1.0 + i * 0.01 for i in range(8)]   # one episode only
        est = estimate_periodicity(times, current_time=2.0, episode_gap=0.05)
        assert est.estimated_period is None
