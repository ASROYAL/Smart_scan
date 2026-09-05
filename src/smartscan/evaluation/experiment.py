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
from smartscan.dsp.detector import EnergyDetector
from smartscan.evaluation.evaluator import Evaluator, ExperimentRunner, RunArtifacts
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
    environment: RFEnvironment


def compute_num_steps(config: SmartScanConfig, scenario_duration: float) -> int:
    """Determine how many scan steps to run.

    If not explicitly configured, derive from duration and per-scan time cost so
    the run spans roughly the scenario duration.
    """
    if config.simulation.num_steps is not None:
        return config.simulation.num_steps
    per_scan = config.receiver.dwell_time + config.receiver.tuning_delay
    return max(1, int(scenario_duration / max(per_scan, 1e-9)))


def build_and_run(
    config: SmartScanConfig,
    scheduler_type: SchedulerType | str,
    scenario_name: str,
    seed: int | None = None,
) -> ExperimentOutcome:
    """Build the full system for one (scheduler, scenario) pair and run it."""
    if seed is None:
        seed = config.simulation.seed

    num_bands = config.environment.num_bands
    total_bw = config.environment.total_bandwidth
    center = config.environment.center_frequency

    # Scenario defines the emitters
    scenario: Scenario = get_scenario(
        scenario_name, seed=seed, num_bands=num_bands,
        total_bw=total_bw, center=center,
    )

    # Environment + source (ground truth lives here, walled off from scheduler)
    env = RFEnvironment(
        emitter_configs=scenario.emitters,
        noise_power_dbm=scenario.noise_power_dbm,
        sample_rate=config.environment.sample_rate,
        seed=seed,
    )
    source = SimulatedRFSource(env)

    # Receiver
    receiver = Receiver(source, config.receiver)

    # DSP
    detector = EnergyDetector(config.detector)

    # State + features (scheduler's ONLY view)
    state = SpectrumStateManager(
        num_bands=num_bands, total_bandwidth=total_bw, center_frequency=center,
        ewma_alpha=config.scheduler.ewma_alpha,
    )
    features = FeatureExtractor()

    # Scheduler
    scheduler = create_scheduler(
        scheduler_type, config.receiver, config.scheduler, num_bands, seed=seed,
    )

    # Runner
    runner = ExperimentRunner(
        env, receiver, detector, state, features, scheduler, config.scheduler,
    )

    num_steps = compute_num_steps(config, scenario.duration)
    artifacts = runner.run(num_steps)

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
