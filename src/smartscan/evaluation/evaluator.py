"""Experiment runner and evaluator.

The ExperimentRunner drives the online scan loop: scheduler → receiver → DSP →
detector → state → scheduler.update. It is ALSO the only component permitted to
query ground truth, which it does SEPARATELY from the scheduler's data path,
purely to annotate scan records for scoring.
"""

from __future__ import annotations

import time as wallclock
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from smartscan.core.config import SchedulerConfig
from smartscan.core.models import BandObservation, ExperimentResult, GroundTruthEvent
from smartscan.dsp.detector import BaseDetector
from smartscan.evaluation import metrics
from smartscan.evaluation.metrics import ScanRecord
from smartscan.features.extractor import FeatureExtractor
from smartscan.receiver.receiver import Receiver
from smartscan.rewards import compute_scan_reward
from smartscan.schedulers.base import BaseScheduler
from smartscan.state.spectrum_state import SpectrumStateManager
from smartscan.utils.logging import ScanLogEntry, ScanLogger
from smartscan.utils.timing import TimingInstrument


@runtime_checkable
class GroundTruthSource(Protocol):
    """The evaluation-only interface the runner needs from an environment.

    RFEnvironment, PDWEnvironment and the recorded-IQ _NullGroundTruth all satisfy
    this, so the runner is typed against the interface — not a concrete class — and
    stays honest about consuming ground truth ONLY through these two queries.
    """

    def is_any_active_in_band(
        self, time: float, freq_start: float, freq_end: float, dwell: float = 0.0,
    ) -> bool: ...

    def get_all_ground_truth(
        self, start_time: float, end_time: float,
    ) -> list[GroundTruthEvent]: ...


@dataclass
class RunArtifacts:
    """Everything produced by a run, for scoring and visualization."""
    records: list[ScanRecord] = field(default_factory=list)
    logger: ScanLogger = field(default_factory=ScanLogger)
    timing: TimingInstrument = field(default_factory=TimingInstrument)
    band_visit_counts: dict[int, int] = field(default_factory=dict)
    # latest extracted feature vector per band (populated by the FeatureExtractor)
    band_features: dict = field(default_factory=dict)


class ExperimentRunner:
    """Runs one scheduler against one environment for a number of steps."""

    def __init__(
        self,
        environment: GroundTruthSource,
        receiver: Receiver,
        detector: BaseDetector,
        state_manager: SpectrumStateManager,
        feature_extractor: FeatureExtractor,
        scheduler: BaseScheduler,
        scheduler_config: SchedulerConfig,
        predictor=None,
    ) -> None:
        self.env = environment
        self.rx = receiver
        self.detector = detector
        self.state = state_manager
        self.features = feature_extractor
        self.scheduler = scheduler
        self.sched_cfg = scheduler_config
        self.predictor = predictor  # optional BasePredictor for pre-scan belief

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

            # Feature extraction for the selected band (observation-derived only)
            pre_state = self.state.get_state(decision.band_id)
            history = self.state.get_history(decision.band_id)
            feats = self.features.extract(pre_state, history, current_time)
            artifacts.band_features[decision.band_id] = feats

            # Pre-scan prediction: does the system believe this band is active?
            # Use the pluggable predictor if provided, else the band's EWMA belief.
            if self.predictor is not None:
                prob = self.predictor.predict_activity_probability(pre_state, current_time)
            else:
                prob = feats.ewma_activity
            predicted_active = prob >= 0.5

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
            # Train the online predictor on (pre-update features -> detected)
            if self.predictor is not None:
                self.predictor.update(pre_state, result.detected, meta.timestamp)

            # Capture the revisit gap BEFORE the update overwrites last_scan_time,
            # so the revisit penalty reflects the gap since the PREVIOUS visit.
            prev_time_since_scan = self.state.get_state(decision.band_id).time_since_scan(current_time)

            with artifacts.timing.measure("state_update"):
                self.state.update(decision.band_id, obs)

            # 5. Shared reward -> both the reported metric AND the scheduler's learning
            reward = compute_scan_reward(
                self.sched_cfg, obs, prev_time_since_scan,
                self.sched_cfg.starvation_threshold)
            with artifacts.timing.measure("scheduler_update"):
                self.scheduler.update(decision, obs, reward)

            # 6. GROUND TRUTH ANNOTATION — evaluation only, separate from scheduler.
            # Score against the ACTUAL observed window (from acquisition metadata),
            # which the receiver clamps to its instantaneous bandwidth — not the
            # whole logical band. An emitter outside this window must not count.
            win_start = meta.center_frequency - meta.bandwidth / 2
            win_end = meta.center_frequency + meta.bandwidth / 2
            truly_active = self.env.is_any_active_in_band(
                time=meta.timestamp, freq_start=win_start, freq_end=win_end,
                dwell=meta.dwell_time,
            )

            artifacts.records.append(ScanRecord(
                scan_number=step,
                timestamp=meta.timestamp,
                band_id=decision.band_id,
                detected=result.detected,
                truly_active=truly_active,
                reward=reward,
                freq_start=win_start,
                freq_end=win_end,
                predicted_prob=float(prob),
                predicted_active=predicted_active,
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



class Evaluator:
    """Scores a run's artifacts against ground truth to produce ExperimentResult."""

    def __init__(self, environment: GroundTruthSource) -> None:
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
        censored_delay, missed_event_rate = metrics.censored_intercept_statistics(
            event_starts, detection_times_by_event, duration,
        )

        pd = metrics.probability_of_detection(records)
        pfa = metrics.probability_of_false_alarm(records)
        hit_rate = metrics.scan_hit_rate(records)
        # DISTINCT events intercepted (from the same event→first-detection matching
        # used for discovery delay) — not per-scan true positives.
        discovery_ratio = metrics.activity_discovery_ratio(
            len(detection_times_by_event), len(gt_events))
        emitter_ratio = self._emitter_intercept_ratio(gt_events, detection_times_by_event)
        efficiency = metrics.scan_efficiency(records)
        coverage = metrics.band_coverage(records, num_bands)
        starve = metrics.starvation_rate(records, num_bands, duration, starvation_threshold)
        avg_reward = metrics.average_reward(records)

        # EW / prediction figures of merit
        pred_accuracy = metrics.percentage_correct_predictions(records)
        brier = metrics.brier_score(records)
        logloss = metrics.log_loss(records)
        intercept_err = self._intercept_time_error(gt_events, duration)

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
            emitter_intercept_ratio=emitter_ratio,
            avg_discovery_delay=mean_delay,
            median_discovery_delay=median_delay,
            p95_discovery_delay=p95_delay,
            scan_efficiency=efficiency,
            band_coverage=coverage,
            starvation_rate=starve,
            avg_reward=avg_reward,
            prediction_accuracy=pred_accuracy,
            brier_score=brier,
            log_loss=logloss,
            avg_intercept_time_error=intercept_err,
            missed_event_rate=missed_event_rate,
            censored_avg_intercept_time=censored_delay,
            wall_clock_seconds=wall_clock_seconds,
            config=config or {},
        )

    def _intercept_time_error(self, gt_events: list, duration: float) -> float:
        """Average |predicted - actual| next-activity time via periodicity.

        For each band that the scheduler estimated a period for and detected at
        least once, predict its next activity as last_detection + estimated_period,
        then compare to the true next event start in that band. Uses final band
        states (scheduler-side estimates) and ground-truth events (evaluation-side).
        """
        state_mgr = getattr(self, "_state_manager", None)
        if state_mgr is None:
            return 0.0

        predicted: dict[int, float] = {}
        for bs in state_mgr.all_states():
            if bs.estimated_period and bs.last_detection_time >= 0:
                predicted[bs.band_id] = bs.last_detection_time + bs.estimated_period

        if not predicted:
            return 0.0

        actual: dict[int, float] = {}
        for band_id in predicted:
            bs = state_mgr.get_state(band_id)
            anchor = bs.last_detection_time
            # first true event in this band that starts after the last detection
            future = [
                e.time_start for e in gt_events
                if e.freq_end > bs.freq_start and e.freq_start < bs.freq_end
                and e.time_start > anchor
            ]
            if future:
                actual[band_id] = min(future)

        return metrics.average_intercept_time_error(predicted, actual)

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
                # Frequency overlap between the event and the scan's ACTUAL window
                freq_overlap = (r.freq_end > event.freq_start
                                and r.freq_start < event.freq_end)
                # Time within the event window
                time_within = event.time_start <= r.timestamp <= event.time_end
                if freq_overlap and time_within and (
                    idx not in detection_times or r.timestamp < detection_times[idx]
                ):
                    detection_times[idx] = r.timestamp
        return detection_times

    @staticmethod
    def _emitter_intercept_ratio(gt_events: list, detection_times_by_event: dict) -> float:
        """Fraction of distinct EMITTERS intercepted at least once.

        Distinct from the event discovery ratio: several ground-truth events can
        belong to one emitter (e.g. each radar illumination or each hop).
        """
        all_emitters = {e.emitter_id for e in gt_events}
        if not all_emitters:
            return 0.0
        hit_emitters = {gt_events[idx].emitter_id for idx in detection_times_by_event}
        return len(hit_emitters) / len(all_emitters)

    def set_state_manager(self, state_manager: SpectrumStateManager) -> None:
        """Provide band layout for the periodicity-based intercept-time-error metric."""
        self._state_manager = state_manager
