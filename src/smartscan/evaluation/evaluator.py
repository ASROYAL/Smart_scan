"""Experiment runner and evaluator.

The ExperimentRunner drives the online scan loop: scheduler → receiver → DSP →
detector → state → scheduler.update. It is ALSO the only component permitted to
query ground truth, which it does SEPARATELY from the scheduler's data path,
purely to annotate scan records for scoring.
"""

from __future__ import annotations

import time as wallclock
from dataclasses import dataclass, field

import numpy as np

from smartscan.core.config import SchedulerConfig
from smartscan.core.models import BandObservation, ExperimentResult
from smartscan.dsp.detector import EnergyDetector
from smartscan.evaluation import metrics
from smartscan.evaluation.metrics import ScanRecord
from smartscan.features.extractor import FeatureExtractor
from smartscan.receiver.receiver import Receiver
from smartscan.schedulers.base import BaseScheduler
from smartscan.simulation.environment import RFEnvironment
from smartscan.state.spectrum_state import SpectrumStateManager
from smartscan.utils.logging import ScanLogEntry, ScanLogger
from smartscan.utils.timing import TimingInstrument


@dataclass
class RunArtifacts:
    """Everything produced by a run, for scoring and visualization."""
    records: list[ScanRecord] = field(default_factory=list)
    logger: ScanLogger = field(default_factory=ScanLogger)
    timing: TimingInstrument = field(default_factory=TimingInstrument)
    band_visit_counts: dict[int, int] = field(default_factory=dict)


class ExperimentRunner:
    """Runs one scheduler against one environment for a number of steps."""

    def __init__(
        self,
        environment: RFEnvironment,
        receiver: Receiver,
        detector: EnergyDetector,
        state_manager: SpectrumStateManager,
        feature_extractor: FeatureExtractor,
        scheduler: BaseScheduler,
        scheduler_config: SchedulerConfig,
    ) -> None:
        self.env = environment
        self.rx = receiver
        self.detector = detector
        self.state = state_manager
        self.features = feature_extractor
        self.scheduler = scheduler
        self.sched_cfg = scheduler_config

    def run(self, num_steps: int) -> RunArtifacts:
        """Execute the online scan loop for num_steps decisions."""
        artifacts = RunArtifacts()
        wall_start = wallclock.perf_counter()

        for step in range(num_steps):
            current_time = self.rx.get_time()

            # 1. Scheduler decides (using observation-derived state ONLY)
            with artifacts.timing.measure("scheduler_decision"):
                states = self.state.all_states()
                decision = self.scheduler.select_band(states, current_time)

            # 2. Receiver observes (returns IQ + meta, NO ground truth)
            with artifacts.timing.measure("acquisition"):
                samples, meta = self.rx.observe(
                    center_frequency=decision.center_frequency,
                    bandwidth=decision.bandwidth,
                    dwell_time=decision.dwell_time,
                )

            # 3. DSP + detection
            with artifacts.timing.measure("dsp_detection"):
                result = self.detector.detect(samples, meta)

            # 4. Build observation and update state
            obs = BandObservation(
                band_id=decision.band_id,
                timestamp=meta.timestamp,
                detected=result.detected,
                confidence=result.confidence,
                peak_power_db=result.peak_power_db,
                avg_power_db=result.avg_power_db,
                noise_floor_db=result.noise_floor_db,
                estimated_snr_db=result.estimated_snr_db,
            )
            with artifacts.timing.measure("state_update"):
                self.state.update(decision.band_id, obs)

            # 5. Reward + scheduler learning
            reward = self._compute_reward(decision, obs, current_time)
            with artifacts.timing.measure("scheduler_update"):
                self.scheduler.update(decision, obs)

            # 6. GROUND TRUTH ANNOTATION — evaluation only, separate from scheduler
            band_state = self.state.get_state(decision.band_id)
            truly_active = self.env.is_any_active_in_band(
                time=meta.timestamp,
                freq_start=band_state.freq_start,
                freq_end=band_state.freq_end,
            )

            artifacts.records.append(ScanRecord(
                scan_number=step,
                timestamp=meta.timestamp,
                band_id=decision.band_id,
                detected=result.detected,
                truly_active=truly_active,
                reward=reward,
            ))
            artifacts.band_visit_counts[decision.band_id] = (
                artifacts.band_visit_counts.get(decision.band_id, 0) + 1
            )

            # 7. Structured log
            artifacts.logger.log(ScanLogEntry(
                scan_number=step,
                timestamp=meta.timestamp,
                band_id=decision.band_id,
                center_frequency=decision.center_frequency,
                dwell_time=decision.dwell_time,
                scheduler_score=decision.priority_score,
                detected=result.detected,
                estimated_snr_db=result.estimated_snr_db,
                hit=result.detected,
                reward=reward,
                processing_latency_ms=artifacts.timing.get_stats("dsp_detection").max_ms,
                reason=decision.reason,
            ))

        self._wall_seconds = wallclock.perf_counter() - wall_start
        return artifacts

    def _compute_reward(self, decision, obs: BandObservation, current_time: float) -> float:
        """Reward from detection outcome (NOT ground truth).

        Positive for a detection, small time cost otherwise, plus a penalty for
        long revisit delays to discourage starvation.
        """
        cfg = self.sched_cfg
        if obs.detected:
            reward = cfg.reward_hit
        else:
            reward = cfg.reward_miss

        # Penalty proportional to how long since this band was last visited
        state = self.state.get_state(decision.band_id)
        tss = state.time_since_scan(current_time)
        if not np.isinf(tss):
            reward -= cfg.reward_revisit_penalty_scale * tss

        return reward


class Evaluator:
    """Scores a run's artifacts against ground truth to produce ExperimentResult."""

    def __init__(self, environment: RFEnvironment) -> None:
        self.env = environment

    def evaluate(
        self,
        artifacts: RunArtifacts,
        scheduler_name: str,
        scenario_name: str,
        seed: int,
        duration: float,
        num_bands: int,
        starvation_threshold: float,
        wall_clock_seconds: float = 0.0,
        config: dict | None = None,
    ) -> ExperimentResult:
        records = artifacts.records

        # Ground-truth events over the run window (evaluation only)
        gt_events = self.env.get_all_ground_truth(0.0, duration)
        event_starts = [e.time_start for e in gt_events]

        # Match each event to its first detection
        detection_times_by_event = self._match_detections_to_events(records, gt_events)
        delays = metrics.discovery_delays(event_starts, detection_times_by_event)
        mean_delay, median_delay, p95_delay = metrics.delay_statistics(delays)

        pd = metrics.probability_of_detection(records)
        pfa = metrics.probability_of_false_alarm(records)
        hit_rate = metrics.scan_hit_rate(records)
        discovery_ratio = metrics.activity_discovery_ratio(records, len(gt_events))
        efficiency = metrics.scan_efficiency(records)
        coverage = metrics.band_coverage(records, num_bands)
        starve = metrics.starvation_rate(records, num_bands, duration, starvation_threshold)
        avg_reward = metrics.average_reward(records)

        return ExperimentResult(
            scheduler_name=scheduler_name,
            scenario_name=scenario_name,
            seed=seed,
            num_steps=len(records),
            duration=duration,
            probability_of_detection=pd,
            probability_of_false_alarm=pfa,
            scan_hit_rate=hit_rate,
            activity_discovery_ratio=discovery_ratio,
            avg_discovery_delay=mean_delay,
            median_discovery_delay=median_delay,
            p95_discovery_delay=p95_delay,
            scan_efficiency=efficiency,
            band_coverage=coverage,
            starvation_rate=starve,
            avg_reward=avg_reward,
            wall_clock_seconds=wall_clock_seconds,
            config=config or {},
        )

    def _match_detections_to_events(
        self, records: list[ScanRecord], gt_events: list,
    ) -> dict[int, float]:
        """For each ground-truth event, find the first scan that detected it.

        A record discovers event i if it detected activity, overlaps the event's
        frequency band, and falls within the event's active time window.
        """
        detection_times: dict[int, float] = {}

        for idx, event in enumerate(gt_events):
            for r in records:
                if not r.detected:
                    continue
                band_state = self._band_for_record(r)
                if band_state is None:
                    continue
                # Frequency overlap with the event
                freq_overlap = (
                    band_state.freq_end > event.freq_start
                    and band_state.freq_start < event.freq_end
                )
                # Time within the event window
                time_within = event.time_start <= r.timestamp <= event.time_end
                if freq_overlap and time_within:
                    if idx not in detection_times or r.timestamp < detection_times[idx]:
                        detection_times[idx] = r.timestamp
        return detection_times

    def _band_for_record(self, record: ScanRecord):
        """Look up the band state for a record via the environment's band layout."""
        # The evaluator holds a reference to state through the runner; but to keep
        # it decoupled, reconstruct band bounds from the environment is not possible.
        # Instead we rely on band_id being stable and stored separately.
        return getattr(self, "_state_manager", None) and self._state_manager.get_state(
            record.band_id
        )

    def set_state_manager(self, state_manager: SpectrumStateManager) -> None:
        """Provide band layout for frequency-overlap checks during scoring."""
        self._state_manager = state_manager
