"""Receiver model — observes a limited slice of spectrum with tuning delay.

The receiver's instantaneous bandwidth is much smaller than the total monitored
spectrum, so it can only observe one window at a time. Tuning to a new frequency
costs a tuning delay, and dwelling costs dwell time — both advance the clock.
"""

from __future__ import annotations

import numpy as np

from smartscan.acquisition.base import RFSource
from smartscan.core.config import ReceiverConfig
from smartscan.core.models import AcquisitionMeta


class Receiver:
    """Models a limited-bandwidth RF receiver.

    Wraps an RFSource and enforces the bandwidth constraint plus tuning/dwell
    timing. Returns only IQ samples and metadata — never ground truth.
    """

    def __init__(self, source: RFSource, config: ReceiverConfig) -> None:
        self._source = source
        self._config = config
        self._current_frequency: float | None = None
        self._total_dwell_time = 0.0
        self._total_tuning_time = 0.0
        self._scan_count = 0

    @property
    def config(self) -> ReceiverConfig:
        return self._config

    @property
    def scan_count(self) -> int:
        return self._scan_count

    @property
    def total_dwell_time(self) -> float:
        return self._total_dwell_time

    @property
    def total_tuning_time(self) -> float:
        return self._total_tuning_time

    @property
    def current_frequency(self) -> float | None:
        return self._current_frequency

    def observe(
        self,
        center_frequency: float,
        bandwidth: float | None = None,
        dwell_time: float | None = None,
    ) -> tuple[np.ndarray, AcquisitionMeta]:
        """Observe a frequency window for a dwell period.

        Args:
            center_frequency: Center frequency to tune to (Hz).
            bandwidth: Observation bandwidth (defaults to instantaneous_bandwidth).
                       Clamped to the receiver's maximum instantaneous bandwidth.
            dwell_time: How long to observe (defaults to config dwell_time).

        Returns:
            Tuple of (IQ samples, acquisition metadata). NO ground truth.
        """
        if bandwidth is None:
            bandwidth = self._config.instantaneous_bandwidth
        else:
            # Enforce the physical constraint: cannot observe more than instantaneous BW
            bandwidth = min(bandwidth, self._config.instantaneous_bandwidth)

        if dwell_time is None:
            dwell_time = self._config.dwell_time

        # Tuning delay applies when changing frequency
        actual_frequency = center_frequency + self._config.frequency_error_hz
        needs_tuning = (
            self._current_frequency is None
            or abs(self._current_frequency - actual_frequency) > 1e-3
        )
        if needs_tuning:
            self._total_tuning_time += self._config.tuning_delay
            self._advance_source_clock(self._config.tuning_delay)
            self._source.tune(actual_frequency)
            self._current_frequency = actual_frequency

        # Number of samples determined by sample rate and dwell time
        num_samples = int(self._config.sample_rate * dwell_time)
        num_samples = max(num_samples, 1)

        samples, meta = self._source.read_samples(
            center_frequency=actual_frequency,
            bandwidth=bandwidth,
            num_samples=num_samples,
        )

        if self._config.gain_error_db:
            samples = samples * (10.0 ** (self._config.gain_error_db / 20.0))
        if self._config.adc_bits is not None and len(samples):
            levels = 2 ** (self._config.adc_bits - 1) - 1
            real_part = np.real(samples)
            imag_part = np.imag(samples)
            peak = max(float(np.max(np.abs(real_part))), float(np.max(np.abs(imag_part))), 1e-15)
            real = np.round(real_part / peak * levels) / levels
            imag = np.round(imag_part / peak * levels) / levels
            samples = (real + 1j * imag) * peak

        self._total_dwell_time += dwell_time
        self._advance_source_clock(dwell_time)
        self._scan_count += 1

        return samples, meta

    def _advance_source_clock(self, dt: float) -> None:
        """Advance the underlying environment clock, if it supports it."""
        env = getattr(self._source, "_env", None)
        if env is not None and hasattr(env, "advance_time"):
            env.advance_time(dt)

    def get_time(self) -> float:
        """Current simulation time from the source."""
        return self._source.get_time()

    def reset_stats(self) -> None:
        self._total_dwell_time = 0.0
        self._total_tuning_time = 0.0
        self._scan_count = 0
