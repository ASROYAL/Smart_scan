"""Experiment builder — wires the full system from a config + scenario.

Single source of truth for assembling environment → source → receiver → detector
→ state → features → scheduler → runner → evaluator. Used by scripts, the
benchmark harness, and the dashboard so wiring stays consistent everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from smartscan.acquisition.simulator_source import SimulatedRFSource
from smartscan.core.config import SmartScanConfig
from smartscan.core.models import ExperimentResult, SchedulerType
from smartscan.dsp.detector import create_detector
from smartscan.evaluation.evaluator import (
    Evaluator,
    ExperimentRunner,
    GroundTruthSource,
    RunArtifacts,
)
from smartscan.features.extractor import FeatureExtractor
from smartscan.receiver.receiver import Receiver
from smartscan.schedulers.factory import create_scheduler
from smartscan.simulation.environment import RFEnvironment
from smartscan.simulation.scenarios import Scenario, get_scenario
from smartscan.state.spectrum_state import SpectrumStateManager


@dataclass
class ExperimentOutcome:
    result: ExperimentResult
    artifacts: RunArtifacts
    state_manager: SpectrumStateManager
    environment: GroundTruthSource  # RFEnvironment | PDWEnvironment | _NullGroundTruth


def compute_num_steps(config: SmartScanConfig, scenario_duration: float) -> int:
    """Determine how many scan steps to run.

    If not explicitly configured, derive from duration and per-scan time cost so
    the run spans roughly the scenario duration.
    """
    if config.simulation.num_steps is not None:
        return config.simulation.num_steps
    per_scan = config.receiver.dwell_time + config.receiver.tuning_delay
    # Prefer an explicit simulation.duration if the user set one; otherwise fall
    # back to the selected scenario's own duration.
    duration = (config.simulation.duration
                if config.simulation.duration is not None else scenario_duration)
    return max(1, int(duration / max(per_scan, 1e-9)))


def _persist_run(state_manager, path: str) -> None:
    """Write final band states and their observation history to a SQLite file."""
    from smartscan.state.database import ScanDatabase
    with ScanDatabase(path) as db:
        for s in state_manager.all_states():
            db.save_band_state(s)
            for obs in state_manager.get_history(s.band_id).observations:
                db.save_observation(obs)


def _make_predictor(use_ml_predictor: bool):
    if not use_ml_predictor:
        return None
    from smartscan.prediction.ml_predictor import MLActivityPredictor
    return MLActivityPredictor()


def build_and_run(
    config: SmartScanConfig,
    scheduler_type: SchedulerType | str,
    scenario_name: str,
    seed: int | None = None,
    use_ml_predictor: bool = False,
    scenario_override: Scenario | None = None,
) -> ExperimentOutcome:
    """Build the full system for one (scheduler, scenario) pair and run it."""
    if seed is None:
        seed = config.simulation.seed

    num_bands = config.environment.num_bands
    total_bw = config.environment.total_bandwidth
    center = config.environment.center_frequency

    # Scenario defines the emitters
    scenario: Scenario = scenario_override or get_scenario(
        scenario_name, seed=seed, num_bands=num_bands,
        total_bw=total_bw, center=center,
    )

    # Environment + source (ground truth lives here, walled off from scheduler)
    env = RFEnvironment(
        emitter_configs=scenario.emitters,
        noise_power_dbm=scenario.noise_power_dbm,
        sample_rate=config.environment.sample_rate,
        seed=seed,
        noise_drift_db=config.environment.noise_drift_db,
        impulsive_noise_probability=config.environment.impulsive_noise_probability,
        impulsive_noise_gain_db=config.environment.impulsive_noise_gain_db,
    )
    source = SimulatedRFSource(env)

    # Receiver
    receiver = Receiver(source, config.receiver)

    # DSP
    detector = create_detector(config.detector)

    # State + features (scheduler's ONLY view)
    state = SpectrumStateManager(
        num_bands=num_bands, total_bandwidth=total_bw, center_frequency=center,
        ewma_alpha=config.scheduler.ewma_alpha,
        episode_gap=max(5 * config.receiver.dwell_time, 0.03),
    )
    features = FeatureExtractor()

    # Scheduler
    scheduler = create_scheduler(
        scheduler_type, config.receiver, config.scheduler, num_bands, seed=seed,
    )

    # Runner (optionally with a pluggable online predictor)
    runner = ExperimentRunner(
        env, receiver, detector, state, features, scheduler, config.scheduler,
        predictor=_make_predictor(use_ml_predictor),
    )

    num_steps = compute_num_steps(config, scenario.duration)
    artifacts = runner.run(num_steps)

    # Optional durable persistence (default backend is in-memory only)
    if config.database.backend == "sqlite":
        _persist_run(state, config.database.sqlite_path)

    # Evaluate against ground truth
    evaluator = Evaluator(env)
    evaluator.set_state_manager(state)
    result = evaluator.evaluate(
        artifacts,
        scheduler_name=scheduler.name,
        scenario_name=scenario_name,
        seed=seed,
        duration=env.time,
        num_bands=num_bands,
        starvation_threshold=config.scheduler.starvation_threshold,
        wall_clock_seconds=getattr(runner, "_wall_seconds", 0.0),
        config={
            "scheduler": scheduler.name,
            "scenario": scenario_name,
            "num_bands": num_bands,
            "num_steps": num_steps,
            "seed": seed,
            "dwell_time": config.receiver.dwell_time,
            "tuning_delay": config.receiver.tuning_delay,
            "instantaneous_bandwidth": config.receiver.instantaneous_bandwidth,
            "detector_margin_db": config.detector.threshold_margin_db,
        },
    )

    return ExperimentOutcome(
        result=result, artifacts=artifacts,
        state_manager=state, environment=env,
    )


def build_and_run_pdw(
    config: SmartScanConfig,
    scheduler_type: SchedulerType | str,
    pdws,
    seed: int | None = None,
    scenario_name: str = "pdw",
) -> ExperimentOutcome:
    """Same online loop as build_and_run, but the environment is a PDW stream
    (e.g. the Turing Synthetic Radar dataset, Scan Mode). All downstream
    components — receiver, DSP, scheduler, evaluator — are identical."""
    from smartscan.acquisition.pdw_source import PDWRFSource
    from smartscan.simulation.pdw import PDWEnvironment

    if seed is None:
        seed = config.simulation.seed
    num_bands = config.environment.num_bands
    total_bw = config.environment.total_bandwidth
    center = config.environment.center_frequency

    env = PDWEnvironment(
        pdws, noise_power_dbm=config.environment.noise_power_dbm,
        sample_rate=config.environment.sample_rate, seed=seed,
    )
    source = PDWRFSource(env)
    receiver = Receiver(source, config.receiver)
    detector = create_detector(config.detector)
    state = SpectrumStateManager(
        num_bands=num_bands, total_bandwidth=total_bw, center_frequency=center,
        ewma_alpha=config.scheduler.ewma_alpha,
        episode_gap=max(5 * config.receiver.dwell_time, 0.03),
    )
    features = FeatureExtractor()
    scheduler = create_scheduler(
        scheduler_type, config.receiver, config.scheduler, num_bands, seed=seed,
    )
    runner = ExperimentRunner(
        env, receiver, detector, state, features, scheduler, config.scheduler,
    )
    num_steps = config.simulation.num_steps or 800
    artifacts = runner.run(num_steps)

    evaluator = Evaluator(env)
    evaluator.set_state_manager(state)
    result = evaluator.evaluate(
        artifacts, scheduler_name=scheduler.name, scenario_name=scenario_name,
        seed=seed, duration=env.time, num_bands=num_bands,
        starvation_threshold=config.scheduler.starvation_threshold,
        wall_clock_seconds=getattr(runner, "_wall_seconds", 0.0),
        config={"scheduler": scheduler.name, "scenario": scenario_name,
                "num_bands": num_bands, "num_steps": num_steps, "seed": seed,
                "source": "pdw"},
    )
    return ExperimentOutcome(
        result=result, artifacts=artifacts, state_manager=state, environment=env,
    )


class _NullGroundTruth:
    """Environment stand-in for real recordings, which carry no ground truth.

    Real recorded IQ has no labels, so truth-based metrics (PD / PFA / discovery
    ratio / discovery delay) are undefined and report 0; detection-side metrics
    (scan hit rate, coverage, reward, prediction accuracy) remain meaningful.
    """

    def __init__(self) -> None:
        self._time = 0.0

    @property
    def time(self) -> float:
        return self._time

    def advance_time(self, dt: float) -> None:
        self._time += dt

    def set_time(self, t: float) -> None:
        self._time = t

    def is_any_active_in_band(self, time, freq_start, freq_end, dwell=0.0):
        return False

    def get_ground_truth_at(self, time, freq_start, freq_end):
        return []

    def get_all_ground_truth(self, start_time, end_time):
        return []


def build_and_run_recorded(
    config: SmartScanConfig,
    scheduler_type: SchedulerType | str,
    iq_path,
    meta_path=None,
    seed: int | None = None,
    use_ml_predictor: bool = False,
) -> ExperimentOutcome:
    """Run the full online loop against a recorded complex64 IQ file.

    Mirrors build_and_run / build_and_run_pdw but the samples come from
    RecordedIQSource. Truth-based metrics are unavailable for real recordings
    (reported as 0); detection-side metrics are real.
    """
    from smartscan.acquisition.file_source import RecordedIQSource

    if seed is None:
        seed = config.simulation.seed
    num_bands = config.environment.num_bands
    total_bw = config.environment.total_bandwidth
    center = config.environment.center_frequency

    source = RecordedIQSource(iq_path, meta_path)
    env = _NullGroundTruth()
    receiver = Receiver(source, config.receiver)
    detector = create_detector(config.detector)
    state = SpectrumStateManager(
        num_bands=num_bands, total_bandwidth=total_bw, center_frequency=center,
        ewma_alpha=config.scheduler.ewma_alpha,
        episode_gap=max(5 * config.receiver.dwell_time, 0.03),
    )
    features = FeatureExtractor()
    scheduler = create_scheduler(
        scheduler_type, config.receiver, config.scheduler, num_bands, seed=seed,
    )
    runner = ExperimentRunner(
        env, receiver, detector, state, features, scheduler, config.scheduler,
        predictor=_make_predictor(use_ml_predictor),
    )
    num_steps = config.simulation.num_steps or 800
    artifacts = runner.run(num_steps)

    if config.database.backend == "sqlite":
        _persist_run(state, config.database.sqlite_path)

    evaluator = Evaluator(env)
    evaluator.set_state_manager(state)
    result = evaluator.evaluate(
        artifacts, scheduler_name=scheduler.name, scenario_name="recorded",
        seed=seed, duration=source.get_time(), num_bands=num_bands,
        starvation_threshold=config.scheduler.starvation_threshold,
        wall_clock_seconds=getattr(runner, "_wall_seconds", 0.0),
        config={"scheduler": scheduler.name, "scenario": "recorded",
                "source": str(iq_path), "num_steps": num_steps, "seed": seed},
    )
    return ExperimentOutcome(
        result=result, artifacts=artifacts, state_manager=state, environment=env,
    )
