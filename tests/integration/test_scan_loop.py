"""Integration test — full scan loop from simulator to evaluator."""

import pytest

from smartscan.acquisition.simulator_source import SimulatedRFSource
from smartscan.core.config import DetectorConfig, ReceiverConfig, SchedulerConfig
from smartscan.core.models import EmitterConfig, EmitterType
from smartscan.dsp.detector import EnergyDetector
from smartscan.evaluation.evaluator import Evaluator, ExperimentRunner
from smartscan.features.extractor import FeatureExtractor
from smartscan.receiver.receiver import Receiver
from smartscan.schedulers.round_robin import RoundRobinScheduler
from smartscan.simulation.environment import RFEnvironment
from smartscan.state.spectrum_state import SpectrumStateManager


def build_system(num_bands=5, emitters=None, seed=42):
    if emitters is None:
        emitters = [
            EmitterConfig(
                emitter_id=0, emitter_type=EmitterType.CONTINUOUS,
                center_frequency=50e6, bandwidth=5e6, amplitude=1.0, snr_db=30.0,
            ),
        ]
    total_bw = 100e6
    center = 50e6
    env = RFEnvironment(emitters, noise_power_dbm=-100.0, sample_rate=20e6, seed=seed)
    source = SimulatedRFSource(env)
    rx_cfg = ReceiverConfig(
        instantaneous_bandwidth=20e6, sample_rate=20e6,
        dwell_time=0.001, tuning_delay=0.0001,
    )
    receiver = Receiver(source, rx_cfg)
    detector = EnergyDetector(DetectorConfig(threshold_margin_db=6.0, fft_size=512))
    state = SpectrumStateManager(num_bands, total_bw, center)
    features = FeatureExtractor()
    sched_cfg = SchedulerConfig()
    scheduler = RoundRobinScheduler(rx_cfg)

    runner = ExperimentRunner(env, receiver, detector, state, features, scheduler, sched_cfg)
    return runner, env, state


class TestFullScanLoop:
    def test_runs_without_error(self):
        runner, env, state = build_system()
        artifacts = runner.run(num_steps=25)
        assert len(artifacts.records) == 25

    def test_detects_strong_continuous_signal(self):
        """A 30 dB continuous emitter at 50 MHz should be detected when scanned."""
        runner, env, state = build_system()
        artifacts = runner.run(num_steps=25)

        # Band containing 50 MHz should have detections
        target_band = state.band_id_for_frequency(50e6)
        target_records = [r for r in artifacts.records if r.band_id == target_band]
        detections = [r for r in target_records if r.detected]
        assert len(detections) > 0

    def test_evaluator_produces_metrics(self):
        runner, env, state = build_system()
        artifacts = runner.run(num_steps=50)

        evaluator = Evaluator(env)
        evaluator.set_state_manager(state)
        result = evaluator.evaluate(
            artifacts, scheduler_name="round_robin", scenario_name="test",
            seed=42, duration=env.time, num_bands=5, starvation_threshold=10.0,
        )
        assert 0.0 <= result.probability_of_detection <= 1.0
        assert 0.0 <= result.probability_of_false_alarm <= 1.0
        assert result.band_coverage > 0.0

    def test_round_robin_full_coverage(self):
        runner, env, state = build_system(num_bands=5)
        artifacts = runner.run(num_steps=10)
        visited = set(artifacts.band_visit_counts.keys())
        assert visited == {0, 1, 2, 3, 4}

    def test_timing_instrumentation_populated(self):
        runner, env, state = build_system()
        artifacts = runner.run(num_steps=10)
        assert artifacts.timing.get_stats("acquisition").count == 10
        assert artifacts.timing.get_stats("dsp_detection").count == 10
        assert artifacts.timing.get_stats("scheduler_decision").count == 10


class TestScanLoopWithPeriodicSource:
    def test_periodic_source_detected_sometimes(self):
        emitters = [
            EmitterConfig(
                emitter_id=0, emitter_type=EmitterType.PERIODIC_BURST,
                center_frequency=50e6, bandwidth=5e6, amplitude=1.0,
                snr_db=30.0, period=0.01, duty_cycle=0.5,
            ),
        ]
        runner, env, state = build_system(emitters=emitters)
        artifacts = runner.run(num_steps=50)

        target_band = state.band_id_for_frequency(50e6)
        target_records = [r for r in artifacts.records if r.band_id == target_band]
        # Should detect during ON phases at least sometimes
        assert any(r.detected for r in target_records)
        # And truly_active annotation should vary (sometimes on, sometimes off)
        active_flags = [r.truly_active for r in target_records]
        assert any(active_flags)


class TestReproducibility:
    def test_same_seed_same_results(self):
        r1, e1, s1 = build_system(seed=42)
        a1 = r1.run(num_steps=30)

        r2, e2, s2 = build_system(seed=42)
        a2 = r2.run(num_steps=30)

        detections1 = [r.detected for r in a1.records]
        detections2 = [r.detected for r in a2.records]
        assert detections1 == detections2
