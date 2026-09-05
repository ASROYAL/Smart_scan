"""RF environment — combines emitters and noise to generate IQ samples.

Maintains ground truth internally. Ground truth is ONLY accessible to the
evaluator, never to the receiver or scheduler.
"""

from __future__ import annotations

import numpy as np

from smartscan.core.models import AcquisitionMeta, EmitterConfig, GroundTruthEvent
from smartscan.simulation.emitters import BaseEmitter, create_emitter
from smartscan.simulation.noise import generate_awgn


class RFEnvironment:
    """Simulated RF environment with configurable emitters and noise."""

    def __init__(
        self,
        emitter_configs: list[EmitterConfig],
        noise_power_dbm: float = -100.0,
        sample_rate: float = 20e6,
        seed: int = 42,
    ) -> None:
        self._noise_power_dbm = noise_power_dbm
        self._sample_rate = sample_rate
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._emitters: list[BaseEmitter] = [
            create_emitter(cfg, noise_power_dbm, seed)
            for cfg in emitter_configs
        ]

        self._ground_truth_log: list[GroundTruthEvent] = []
        self._time = 0.0

    @property
    def emitters(self) -> list[BaseEmitter]:
        return list(self._emitters)

    @property
    def time(self) -> float:
        return self._time

    def advance_time(self, dt: float) -> None:
        self._time += dt

    def set_time(self, t: float) -> None:
        self._time = t

    def generate_samples(
        self,
        center_frequency: float,
        bandwidth: float,
        num_samples: int,
        time: float | None = None,
    ) -> np.ndarray:
        """Generate IQ samples for a specific frequency window.

        This is the ONLY method that bridges the environment to the receiver.
        It returns raw IQ samples — no labels, no ground truth.

        Args:
            center_frequency: Center of the observation window in Hz.
            bandwidth: Observation bandwidth in Hz.
            num_samples: Number of complex samples to generate.
            time: Observation time (defaults to internal clock).

        Returns:
            Complex128 array of IQ samples.
        """
        if time is None:
            time = self._time

        # Start with noise
        samples = generate_awgn(num_samples, self._noise_power_dbm, self._rng)

        # Add contribution from each active emitter in the observation window
        for emitter in self._emitters:
            contribution = emitter.generate_samples(
                time=time,
                num_samples=num_samples,
                sample_rate=self._sample_rate,
                center_frequency=center_frequency,
                bandwidth=bandwidth,
            )
            if contribution is not None:
                samples += contribution

        return samples

    def get_active_emitters(self, time: float | None = None) -> list[BaseEmitter]:
        """Return emitters currently transmitting. FOR EVALUATION ONLY."""
        if time is None:
            time = self._time
        return [e for e in self._emitters if e.is_active(time)]

    def get_ground_truth_at(
        self,
        time: float,
        freq_start: float,
        freq_end: float,
    ) -> list[GroundTruthEvent]:
        """Check if any emitter is active in a frequency/time window. FOR EVALUATION ONLY."""
        events = []
        for emitter in self._emitters:
            if not emitter.is_active(time):
                continue
            freq = emitter.get_frequency(time)
            half_bw = emitter.config.bandwidth / 2
            emitter_start = freq - half_bw
            emitter_end = freq + half_bw

            # Check frequency overlap
            if emitter_end > freq_start and emitter_start < freq_end:
                events.append(GroundTruthEvent(
                    emitter_id=emitter.emitter_id,
                    freq_start=max(emitter_start, freq_start),
                    freq_end=min(emitter_end, freq_end),
                    time_start=time,
                    time_end=time,
                    amplitude=emitter.config.amplitude,
                    snr_db=emitter.config.snr_db,
                ))
        return events

    def get_all_ground_truth(
        self, start_time: float, end_time: float,
    ) -> list[GroundTruthEvent]:
        """Get all ground truth events in a time range. FOR EVALUATION ONLY."""
        events = []
        for emitter in self._emitters:
            events.extend(emitter.get_ground_truth_events(start_time, end_time))
        return events

    def is_any_active_in_band(
        self, time: float, freq_start: float, freq_end: float,
    ) -> bool:
        """Quick check: is any emitter active in this freq/time window? EVALUATION ONLY."""
        return len(self.get_ground_truth_at(time, freq_start, freq_end)) > 0
