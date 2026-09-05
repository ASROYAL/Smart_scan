"""Signal emitter models — each type has its own activity schedule and IQ generation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from smartscan.core.models import EmitterConfig, EmitterType, GroundTruthEvent
from smartscan.simulation.noise import power_for_snr
from smartscan.simulation.waveform import generate_tone


class BaseEmitter(ABC):
    """Base class for signal emitters."""

    def __init__(self, config: EmitterConfig, noise_power_dbm: float) -> None:
        self.config = config
        self.noise_power_dbm = noise_power_dbm
        self.signal_power_dbm = power_for_snr(noise_power_dbm, config.snr_db)

    @property
    def emitter_id(self) -> int:
        return self.config.emitter_id

    @abstractmethod
    def is_active(self, time: float) -> bool:
        """Determine if this emitter is transmitting at the given time."""
        ...

    @abstractmethod
    def get_frequency(self, time: float) -> float:
        """Return center frequency at the given time."""
        ...

    def generate_samples(
        self,
        time: float,
        num_samples: int,
        sample_rate: float,
        center_frequency: float,
        bandwidth: float,
    ) -> np.ndarray | None:
        """Generate IQ contribution if active and within the observation window.

        Returns None if the emitter is inactive or outside the observation band.
        """
        if not self.is_active(time):
            return None

        emitter_freq = self.get_frequency(time)
        freq_offset = emitter_freq - center_frequency

        # Check if the emitter falls within the observation bandwidth
        if abs(freq_offset) > bandwidth / 2:
            return None

        signal = generate_tone(
            num_samples=num_samples,
            sample_rate=sample_rate,
            frequency_offset=freq_offset,
            power_dbm=self.signal_power_dbm,
        )
        return signal

    def get_ground_truth_events(
        self, start_time: float, end_time: float,
    ) -> list[GroundTruthEvent]:
        """Return all activity events in the time range. For evaluation only."""
        events = []
        dt = 0.001  # 1ms resolution for ground truth scanning
        t = start_time
        in_event = False
        event_start = 0.0

        while t <= end_time:
            active = self.is_active(t)
            freq = self.get_frequency(t)
            half_bw = self.config.bandwidth / 2

            if active and not in_event:
                event_start = t
                in_event = True
            elif not active and in_event:
                events.append(GroundTruthEvent(
                    emitter_id=self.emitter_id,
                    freq_start=freq - half_bw,
                    freq_end=freq + half_bw,
                    time_start=event_start,
                    time_end=t,
                    amplitude=self.config.amplitude,
                    snr_db=self.config.snr_db,
                ))
                in_event = False
            t += dt

        if in_event:
            freq = self.get_frequency(end_time)
            half_bw = self.config.bandwidth / 2
            events.append(GroundTruthEvent(
                emitter_id=self.emitter_id,
                freq_start=freq - half_bw,
                freq_end=freq + half_bw,
                time_start=event_start,
                time_end=end_time,
                amplitude=self.config.amplitude,
                snr_db=self.config.snr_db,
            ))

        return events


class ContinuousEmitter(BaseEmitter):
    """Always-on transmitter within its configured time window."""

    def is_active(self, time: float) -> bool:
        if time < self.config.start_time:
            return False
        if self.config.end_time is not None and time > self.config.end_time:
            return False
        return True

    def get_frequency(self, time: float) -> float:
        return self.config.center_frequency


class PeriodicBurstEmitter(BaseEmitter):
    """Transmits in periodic ON/OFF cycles."""

    def is_active(self, time: float) -> bool:
        if time < self.config.start_time:
            return False
        if self.config.end_time is not None and time > self.config.end_time:
            return False
        period = self.config.period or 1.0
        duty = self.config.duty_cycle or 0.5
        phase_in_cycle = (time - self.config.start_time) % period
        return phase_in_cycle < (period * duty)

    def get_frequency(self, time: float) -> float:
        return self.config.center_frequency


class RandomBurstEmitter(BaseEmitter):
    """Transmits random bursts according to a Poisson-like process.

    Uses a seeded RNG to pre-generate burst schedule for reproducibility.
    """

    def __init__(
        self, config: EmitterConfig, noise_power_dbm: float, seed: int = 0,
    ) -> None:
        super().__init__(config, noise_power_dbm)
        self._rng = np.random.default_rng(seed + config.emitter_id)
        self._bursts: list[tuple[float, float]] = []
        self._schedule_generated_until = 0.0

    def _ensure_schedule(self, until_time: float) -> None:
        """Lazily generate burst schedule up to the requested time."""
        if until_time <= self._schedule_generated_until:
            return

        rate = self.config.burst_rate or 1.0
        duration = self.config.burst_duration or 0.1
        t = self._schedule_generated_until or self.config.start_time

        end = self.config.end_time or until_time + 100.0
        target = min(until_time + 10.0, end)

        while t < target:
            interval = self._rng.exponential(1.0 / rate)
            t += interval
            if t < target:
                burst_end = min(t + duration, end)
                self._bursts.append((t, burst_end))

        self._schedule_generated_until = target

    def is_active(self, time: float) -> bool:
        if time < self.config.start_time:
            return False
        if self.config.end_time is not None and time > self.config.end_time:
            return False
        self._ensure_schedule(time)
        return any(start <= time <= end for start, end in self._bursts)

    def get_frequency(self, time: float) -> float:
        return self.config.center_frequency


class FrequencyHoppingEmitter(BaseEmitter):
    """Hops between predefined frequencies on a schedule."""

    def is_active(self, time: float) -> bool:
        if time < self.config.start_time:
            return False
        if self.config.end_time is not None and time > self.config.end_time:
            return False
        return True

    def get_frequency(self, time: float) -> float:
        freqs = self.config.hop_frequencies
        if not freqs:
            return self.config.center_frequency
        interval = self.config.hop_interval or 1.0
        idx = int((time - self.config.start_time) / interval) % len(freqs)
        return freqs[idx]


class ScanLikeEmitter(BaseEmitter):
    """Sweeps sequentially across frequencies, like a scanning transmitter."""

    def is_active(self, time: float) -> bool:
        if time < self.config.start_time:
            return False
        if self.config.end_time is not None and time > self.config.end_time:
            return False
        # Active during the ON portion of each hop interval
        interval = self.config.hop_interval or 1.0
        duty = self.config.duty_cycle or 0.5
        phase = (time - self.config.start_time) % interval
        return phase < interval * duty

    def get_frequency(self, time: float) -> float:
        freqs = self.config.hop_frequencies
        if not freqs:
            return self.config.center_frequency
        interval = self.config.hop_interval or 1.0
        idx = int((time - self.config.start_time) / interval) % len(freqs)
        return freqs[idx]


def create_emitter(
    config: EmitterConfig, noise_power_dbm: float, seed: int = 0,
) -> BaseEmitter:
    """Factory function to create the appropriate emitter type."""
    match config.emitter_type:
        case EmitterType.CONTINUOUS:
            return ContinuousEmitter(config, noise_power_dbm)
        case EmitterType.PERIODIC_BURST:
            return PeriodicBurstEmitter(config, noise_power_dbm)
        case EmitterType.RANDOM_BURST:
            return RandomBurstEmitter(config, noise_power_dbm, seed)
        case EmitterType.FREQUENCY_HOPPING:
            return FrequencyHoppingEmitter(config, noise_power_dbm)
        case EmitterType.SCAN_LIKE:
            return ScanLikeEmitter(config, noise_power_dbm)
        case _:
            raise ValueError(f"Unknown emitter type: {config.emitter_type}")
