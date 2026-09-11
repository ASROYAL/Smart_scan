"""Configuration system — Pydantic models loaded from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from smartscan.core.models import (
    DetectorType,
    NoiseEstMethod,
    PSDMethod,
    SchedulerType,
    WindowFunction,
)


class EnvironmentConfig(BaseModel):
    total_bandwidth: float = Field(default=1_000e6, gt=0, description="Hz")
    center_frequency: float = Field(default=500e6, gt=0, description="Hz")
    num_bands: int = Field(default=50, gt=0)
    noise_power_dbm: float = Field(default=-100.0, description="dBm")
    sample_rate: float = Field(default=20e6, gt=0, description="samples/sec")


class ReceiverConfig(BaseModel):
    instantaneous_bandwidth: float = Field(default=20e6, gt=0, description="Hz")
    sample_rate: float = Field(default=20e6, gt=0, description="samples/sec")
    dwell_time: float = Field(default=0.01, gt=0, description="seconds")
    tuning_delay: float = Field(default=0.001, ge=0, description="seconds")
    scan_step: float | None = Field(default=None, description="Hz, defaults to bandwidth")


class DetectorConfig(BaseModel):
    window: WindowFunction = Field(default=WindowFunction.HANN)
    detector_type: DetectorType = Field(default=DetectorType.ENERGY)
    psd_method: PSDMethod = Field(default=PSDMethod.WELCH)
    noise_method: NoiseEstMethod = Field(default=NoiseEstMethod.MEDIAN)
    threshold_margin_db: float = Field(default=6.0, description="dB above noise floor")
    fft_size: int = Field(default=1024, gt=0)
    welch_nperseg: int | None = Field(default=None, description="defaults to fft_size")
    welch_overlap: float = Field(default=0.5, ge=0, lt=1)
    percentile: float = Field(default=25.0, ge=0, le=100, description="for percentile noise est")
    # cyclostationary/matched-filter detectors
    autocorr_lags: int = Field(default=32, gt=0, description="lags for cyclostationary feature")
    autocorr_threshold: float = Field(default=0.15, ge=0, le=1,
        description="normalised cyclic-autocorrelation detection threshold")


class SchedulerConfig(BaseModel):
    type: SchedulerType = Field(default=SchedulerType.ROUND_ROBIN)
    exploration_weight: float = Field(default=1.0, ge=0)
    recency_weight: float = Field(default=1.0, ge=0)
    uncertainty_weight: float = Field(default=1.0, ge=0)
    starvation_threshold: float = Field(default=10.0, gt=0, description="seconds without visit")
    ewma_alpha: float = Field(default=0.1, gt=0, lt=1, description="EWMA smoothing factor")
    ucb_c: float = Field(default=2.0, gt=0, description="UCB exploration constant")
    reward_hit: float = Field(default=1.0)
    reward_miss: float = Field(default=0.0)
    reward_revisit_penalty_scale: float = Field(default=0.01, ge=0,
        description="deprecated (old long-gap penalty); superseded by the terms below")
    reward_tuning_cost: float = Field(default=0.0, ge=0, description="cost per retune")
    reward_dwell_cost: float = Field(default=0.0, ge=0, description="cost per dwell")
    reward_repeat_penalty_scale: float = Field(default=0.02, ge=0,
        description="penalty for revisiting a just-seen band (over-camping)")
    reward_starvation_bonus_scale: float = Field(default=0.02, ge=0,
        description="bonus for returning to a long-neglected band (coverage recovery)")
    # Q-learning (reinforcement-learning scheduler)
    q_alpha: float = Field(default=0.2, gt=0, le=1, description="TD learning rate")
    q_gamma: float = Field(default=0.9, ge=0, lt=1, description="discount (multi-step lookahead)")
    q_epsilon: float = Field(default=0.2, ge=0, le=1, description="initial exploration rate")
    q_epsilon_decay: float = Field(default=0.999, gt=0, le=1, description="epsilon decay per step")


class SimulationConfig(BaseModel):
    duration: float | None = Field(
        default=None, gt=0,
        description="seconds; when set (and num_steps is not), drives the step count. "
                    "If None, the selected scenario's own duration is used.")
    seed: int = Field(default=42)
    num_steps: int | None = Field(default=None, description="overrides duration-based step count")


class DatabaseConfig(BaseModel):
    backend: str = Field(default="memory", pattern=r"^(memory|sqlite)$")
    sqlite_path: str = Field(default="data/results/scan_history.db")


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    log_dir: str = Field(default="data/results")
    export_csv: bool = Field(default=True)
    export_json: bool = Field(default=True)


class SmartScanConfig(BaseModel):
    """Top-level configuration aggregating all subsystem configs."""
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    receiver: ReceiverConfig = Field(default_factory=ReceiverConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> SmartScanConfig:
    """Load configuration from a YAML file, applying defaults for missing fields."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return SmartScanConfig(**raw)


def load_config_with_overrides(
    base_path: str | Path,
    override_path: str | Path | None = None,
    **kwargs: Any,
) -> SmartScanConfig:
    """Load base config, optionally merge an override file, then apply kwargs."""
    base = load_config(base_path)
    base_dict = base.model_dump()

    if override_path is not None:
        override_path = Path(override_path)
        if override_path.exists():
            with open(override_path) as f:
                overrides: dict[str, Any] = yaml.safe_load(f) or {}
            _deep_merge(base_dict, overrides)

    _deep_merge(base_dict, kwargs)
    return SmartScanConfig(**base_dict)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base, mutating base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
