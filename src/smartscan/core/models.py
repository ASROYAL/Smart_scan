"""Core data models for the SmartScan system.

All shared data contracts used across modules. Pydantic models for validation,
dataclasses where immutability and speed matter more than validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EmitterType(str, Enum):
    CONTINUOUS = "continuous"
    PERIODIC_BURST = "periodic_burst"
    RANDOM_BURST = "random_burst"
    FREQUENCY_HOPPING = "frequency_hopping"
    SCAN_LIKE = "scan_like"
    RADAR_SCAN = "radar_scan"  # rotating-antenna radar: illuminates receiver briefly each rotation


class WindowFunction(str, Enum):
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"


class PSDMethod(str, Enum):
    PERIODOGRAM = "periodogram"
    WELCH = "welch"


class NoiseEstMethod(str, Enum):
    MEDIAN = "median"
    PERCENTILE = "percentile"
    MOVING = "moving"


class SchedulerType(str, Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    PRIORITY = "priority"
    BANDIT_UCB = "bandit_ucb"
    BANDIT_THOMPSON = "bandit_thompson"
    ADAPTIVE = "adaptive"
    Q_LEARNING = "q_learning"


class DetectorType(str, Enum):
    ENERGY = "energy"
    MATCHED_FILTER = "matched_filter"
    CYCLOSTATIONARY = "cyclostationary"


class WaveformType(str, Enum):
    TONE = "tone"                      # continuous-wave single carrier
    PULSED = "pulsed"                  # gated carrier (radar-like pulse train)
    BANDLIMITED_NOISE = "noise"        # occupies `bandwidth` (wideband emission)
    DIGITAL = "digital"               # simple digital modulation (≈ band-limited)
    CHIRP = "chirp"                    # linear-FM sweep across `bandwidth` (radar)


# ---------------------------------------------------------------------------
# Scan Decision — what the scheduler outputs
# ---------------------------------------------------------------------------

class ScanDecision(BaseModel):
    """A scheduler's decision about what to scan next."""
    band_id: int = Field(ge=0)
    center_frequency: float = Field(gt=0, description="Hz")
    bandwidth: float = Field(gt=0, description="Hz")
    dwell_time: float = Field(gt=0, description="seconds")
    priority_score: float = Field(default=0.0)
    reason: str = Field(default="")
    timestamp: float = Field(ge=0, description="simulation time in seconds")
    receiver_id: int = Field(default=0, ge=0)
    recommended_revisit_time: float | None = Field(default=None, ge=0)
    utility_components: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detection Result — what the DSP pipeline outputs
# ---------------------------------------------------------------------------

class DetectionResult(BaseModel):
    """Output of the activity detector for one observation."""
    detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    peak_power_db: float
    avg_power_db: float
    noise_floor_db: float
    estimated_snr_db: float
    freq_start: float = Field(description="Hz")
    freq_end: float = Field(description="Hz")
    timestamp: float = Field(ge=0)
    num_detections: int = Field(default=0, ge=0, description="Number of sub-band detections")


# ---------------------------------------------------------------------------
# Band Observation — combined detection + context for state tracking
# ---------------------------------------------------------------------------

class BandObservation(BaseModel):
    """A single observation record for a frequency band."""
    band_id: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    peak_power_db: float
    avg_power_db: float
    noise_floor_db: float
    estimated_snr_db: float


# ---------------------------------------------------------------------------
# Band State — maintained per frequency band
# ---------------------------------------------------------------------------

class BandState(BaseModel):
    """Persistent state for one frequency band."""
    band_id: int = Field(ge=0)
    freq_start: float = Field(description="Hz")
    freq_end: float = Field(description="Hz")
    last_scan_time: float = Field(default=-1.0, description="negative means never scanned")
    last_detection_time: float = Field(default=-1.0, description="negative means never detected")
    hit_count: int = Field(default=0, ge=0)
    miss_count: int = Field(default=0, ge=0)
    observation_count: int = Field(default=0, ge=0)
    rolling_activity_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_period: float | None = Field(default=None, description="seconds, None if unknown")
    avg_power_db: float = Field(default=-100.0)
    avg_snr_db: float = Field(default=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_score: float = Field(default=1.0, ge=0.0, le=1.0)
    track_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    threat_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def center_frequency(self) -> float:
        return (self.freq_start + self.freq_end) / 2

    @property
    def bandwidth(self) -> float:
        return self.freq_end - self.freq_start

    @property
    def activity_ratio(self) -> float:
        if self.observation_count == 0:
            return 0.0
        return self.hit_count / self.observation_count

    def time_since_scan(self, current_time: float) -> float:
        if self.last_scan_time < 0:
            return float("inf")
        return current_time - self.last_scan_time

    def time_since_detection(self, current_time: float) -> float:
        if self.last_detection_time < 0:
            return float("inf")
        return current_time - self.last_detection_time


# ---------------------------------------------------------------------------
# Emitter configuration — used by simulator
# ---------------------------------------------------------------------------

class EmitterConfig(BaseModel):
    """Configuration for a single signal emitter in the simulation."""
    emitter_id: int = Field(ge=0)
    emitter_type: EmitterType
    center_frequency: float = Field(gt=0, description="Hz")
    bandwidth: float = Field(gt=0, description="Hz occupied bandwidth (shapes non-tone waveforms)")
    amplitude: float = Field(gt=0, description="DEPRECATED — power is controlled by snr_db; kept for back-compat")
    snr_db: float = Field(default=20.0, description="signal power as SNR (dB) above the noise floor — the single power control")
    waveform: WaveformType | None = Field(
        default=None, description="waveform type; None defaults to a CW tone. Set to chirp/pulsed/noise/digital to opt into wider waveforms")

    # Timing (type-dependent)
    start_time: float = Field(default=0.0, ge=0, description="seconds")
    end_time: float | None = Field(default=None, description="None = runs until sim ends")
    period: float | None = Field(default=None, description="seconds, for periodic types")
    duty_cycle: float | None = Field(default=None, ge=0, le=1, description="fraction ON per period")
    burst_rate: float | None = Field(default=None, description="bursts/second for random burst")
    burst_duration: float | None = Field(default=None, description="seconds per burst")

    # Frequency hopping
    hop_frequencies: list[float] | None = Field(default=None, description="list of center freqs")
    hop_interval: float | None = Field(default=None, description="seconds between hops")

    # Radar antenna scan (rotating-beam illumination)
    scan_period: float | None = Field(
        default=None, gt=0, description="antenna rotation period in seconds")
    beam_dwell: float | None = Field(
        default=None, gt=0, description="main-beam illumination time on the receiver per rotation, s")


# ---------------------------------------------------------------------------
# Ground Truth Event — known only to simulator and evaluator
# ---------------------------------------------------------------------------

class GroundTruthEvent(BaseModel):
    """A ground-truth activity event. NEVER visible to the scheduler."""
    emitter_id: int = Field(ge=0)
    freq_start: float
    freq_end: float
    time_start: float = Field(ge=0)
    time_end: float = Field(ge=0)
    amplitude: float = Field(gt=0)
    snr_db: float


# ---------------------------------------------------------------------------
# Feature vector — for ML schedulers
# ---------------------------------------------------------------------------

class BandFeatures(BaseModel):
    """Feature vector for one band at a point in time, used by ML schedulers."""
    band_id: int = Field(ge=0)
    current_power_db: float = Field(default=-100.0)
    estimated_snr_db: float = Field(default=0.0)
    time_since_scan: float = Field(default=float("inf"), ge=0)
    time_since_detection: float = Field(default=float("inf"), ge=0)
    recent_hits: int = Field(default=0, ge=0)
    recent_misses: int = Field(default=0, ge=0)
    activity_ratio: float = Field(default=0.0, ge=0, le=1)
    ewma_activity: float = Field(default=0.5, ge=0, le=1)
    avg_active_duration: float = Field(default=0.0, ge=0)
    avg_inactive_duration: float = Field(default=0.0, ge=0)
    revisit_interval: float = Field(default=0.0, ge=0)
    estimated_period: float | None = Field(default=None)
    confidence: float = Field(default=0.0, ge=0, le=1)
    observation_count: int = Field(default=0, ge=0)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        return np.array([
            self.current_power_db,
            self.estimated_snr_db,
            min(self.time_since_scan, 1e6),
            min(self.time_since_detection, 1e6),
            self.recent_hits,
            self.recent_misses,
            self.activity_ratio,
            self.ewma_activity,
            self.avg_active_duration,
            self.avg_inactive_duration,
            self.revisit_interval,
            self.estimated_period if self.estimated_period is not None else 0.0,
            self.confidence,
            self.observation_count,
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Acquisition metadata — returned alongside IQ samples
# ---------------------------------------------------------------------------

class AcquisitionMeta(BaseModel):
    """Metadata accompanying acquired IQ samples. No ground-truth fields."""
    center_frequency: float = Field(gt=0, description="Hz")
    sample_rate: float = Field(gt=0, description="samples/sec")
    bandwidth: float = Field(gt=0, description="Hz")
    num_samples: int = Field(gt=0)
    timestamp: float = Field(ge=0, description="acquisition start time in seconds")
    dwell_time: float = Field(gt=0, description="seconds")


# ---------------------------------------------------------------------------
# Experiment result summary
# ---------------------------------------------------------------------------

class ExperimentResult(BaseModel):
    """Summary metrics from one experiment run."""
    scheduler_name: str
    scenario_name: str
    seed: int
    num_steps: int = Field(gt=0)
    duration: float = Field(gt=0, description="simulated seconds")

    # Detection metrics
    probability_of_detection: float = Field(ge=0, le=1)
    probability_of_false_alarm: float = Field(ge=0, le=1)
    scan_hit_rate: float = Field(ge=0, le=1)
    activity_discovery_ratio: float = Field(ge=0, le=1, description="distinct events intercepted")
    emitter_intercept_ratio: float = Field(default=0.0, ge=0, le=1,
        description="distinct emitters intercepted / total emitters")

    # Delay metrics
    avg_discovery_delay: float = Field(ge=0, description="seconds")
    median_discovery_delay: float = Field(ge=0, description="seconds")
    p95_discovery_delay: float = Field(ge=0, description="seconds")

    # Efficiency metrics
    scan_efficiency: float = Field(ge=0, le=1)
    band_coverage: float = Field(ge=0, le=1)
    starvation_rate: float = Field(ge=0, le=1)
    avg_reward: float = Field(default=0.0)

    # EW figures of merit (problem-statement vocabulary)
    prediction_accuracy: float = Field(default=0.0, ge=0, le=1,
        description="percentage of correct pre-scan activity predictions")
    brier_score: float = Field(default=0.0, ge=0, le=1,
        description="mean squared error of predicted activity probability (lower better)")
    log_loss: float = Field(default=0.0, ge=0,
        description="binary cross-entropy of predicted probability (lower better)")
    avg_intercept_time_error: float = Field(default=0.0, ge=0,
        description="mean |predicted - actual| next-activity time, seconds")
    missed_event_rate: float = Field(default=0.0, ge=0, le=1)
    censored_avg_intercept_time: float = Field(default=0.0, ge=0)

    # Extra
    wall_clock_seconds: float = Field(default=0.0, ge=0)
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def intercept_ratio(self) -> float:
        """EW alias for activity discovery ratio (fraction of emitters intercepted)."""
        return self.activity_discovery_ratio

    @property
    def avg_intercept_delay(self) -> float:
        """EW alias for average discovery delay (average intercept time)."""
        return self.avg_discovery_delay
