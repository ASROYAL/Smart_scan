"""Signal emitter models — each type has its own activity schedule and IQ generation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from smartscan.core.models import (
    EmitterConfig,
    EmitterType,
    GroundTruthEvent,
    WaveformType,
)
from smartscan.simulation.noise import power_for_snr
from smartscan.simulation.waveform import (
    generate_bandlimited_noise,
    generate_chirp,
    generate_tone,
)


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
        dwell_end = time + num_samples / sample_rate
        intervals = self.activity_intervals(time, dwell_end)
        if not intervals:
            return None
        output = np.zeros(num_samples, dtype=np.complex128)
        contributed = False
        for active_start, active_end, emitter_freq in intervals:
            freq_offset = emitter_freq - center_frequency
            if abs(freq_offset) > bandwidth / 2:
                continue
            first = max(0, int(np.floor((active_start - time) * sample_rate)))
            last = min(num_samples, int(np.ceil((active_end - time) * sample_rate)))
            if last <= first:
                continue
            signal = self._synthesize(num_samples, sample_rate, freq_offset)
            output[first:last] += signal[first:last]
            contributed = True
        return output if contributed else None

    def _effective_waveform(self) -> WaveformType:
        # Default is a CW tone for every emitter (keeps energy detection tractable
        # and results comparable). Wider/realistic waveforms — chirp for radar LFM,
        # band-limited noise / digital for modulated emitters — are opt-in via
        # ``EmitterConfig.waveform``.
        return self.config.waveform if self.config.waveform is not None else WaveformType.TONE

    def _synthesize(self, num_samples, sample_rate, freq_offset):
        """Build the IQ for this emitter using its (possibly inferred) waveform.

        Power comes from ``snr_db`` (the single power control); ``bandwidth`` shapes
        the occupied spectrum for the non-tone waveforms.
        """
        wf = self._effective_waveform()
        p = self.signal_power_dbm
        bw = self.config.bandwidth
        if wf == WaveformType.TONE:
            return generate_tone(num_samples, sample_rate, freq_offset, power_dbm=p)
        if wf == WaveformType.CHIRP:
            return generate_chirp(num_samples, sample_rate, freq_offset, bw, power_dbm=p)
        if wf == WaveformType.PULSED:
            # gated CW: on for the first half of the window (representative pulse)
            sig = generate_tone(num_samples, sample_rate, freq_offset, power_dbm=p)
            gate = np.zeros(num_samples)
            gate[: max(1, num_samples // 2)] = 1.0
            return sig * gate
        # BANDLIMITED_NOISE / DIGITAL — occupy `bandwidth`
        return generate_bandlimited_noise(num_samples, sample_rate, bw,
                                          center_offset=freq_offset, power_dbm=p)

    def activity_intervals(
        self, start_time: float, end_time: float,
    ) -> list[tuple[float, float, float]]:
        """Return (t_start, t_end, frequency) tuples for each activity episode.

        A new episode begins whenever activity toggles on OR the emitter's
        frequency changes. Subclasses override this analytically (fast + exact);
        the default here walks the clock and is only a fallback.
        """
        intervals: list[tuple[float, float, float]] = []
        dt = 0.001
        t = start_time
        in_event = False
        ev_start = 0.0
        ev_freq = 0.0
        while t <= end_time:
            active = self.is_active(t)
            freq = self.get_frequency(t) if active else ev_freq
            if active and not in_event:
                ev_start, ev_freq, in_event = t, freq, True
            elif in_event and (not active or abs(freq - ev_freq) > 1e3):
                intervals.append((ev_start, t, ev_freq))
                if active:  # frequency changed while still on -> start a new episode
                    ev_start, ev_freq = t, freq
                else:
                    in_event = False
            t += dt
        if in_event:
            intervals.append((ev_start, end_time, ev_freq))
        return intervals

    def get_ground_truth_events(
        self, start_time: float, end_time: float,
    ) -> list[GroundTruthEvent]:
        """Frequency-aware activity events in the range. For evaluation only."""
        half_bw = self.config.bandwidth / 2
        return [
            GroundTruthEvent(
                emitter_id=self.emitter_id,
                freq_start=freq - half_bw, freq_end=freq + half_bw,
                time_start=t0, time_end=t1,
                amplitude=self.config.amplitude, snr_db=self.config.snr_db,
            )
            for (t0, t1, freq) in self.activity_intervals(start_time, end_time)
            if t1 > t0
        ]


def _window(config, start: float, end: float) -> tuple[float, float] | None:
    """Clip [start, end] to the emitter's active lifetime; None if empty."""
    es = max(start, config.start_time)
    ee = min(end, config.end_time) if config.end_time is not None else end
    return (es, ee) if ee > es else None


class ContinuousEmitter(BaseEmitter):
    """Always-on transmitter within its configured time window."""

    def is_active(self, time: float) -> bool:
        if time < self.config.start_time:
            return False
        return not (self.config.end_time is not None and time > self.config.end_time)

    def get_frequency(self, time: float) -> float:
        return self.config.center_frequency

    def activity_intervals(self, start_time, end_time):
        w = _window(self.config, start_time, end_time)
        return [(w[0], w[1], self.config.center_frequency)] if w else []


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

    def activity_intervals(self, start_time, end_time):
        w = _window(self.config, start_time, end_time)
        if not w:
            return []
        es, ee = w
        period = self.config.period or 1.0
        duty = self.config.duty_cycle or 0.5
        freq = self.config.center_frequency
        out = []
        k = int((es - self.config.start_time) // period)
        while True:
            cyc = self.config.start_time + k * period
            if cyc > ee:
                break
            on0, on1 = cyc, cyc + period * duty
            a, b = max(on0, es), min(on1, ee)
            if b > a:
                out.append((a, b, freq))
            k += 1
        return out


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

    def activity_intervals(self, start_time, end_time):
        w = _window(self.config, start_time, end_time)
        if not w:
            return []
        es, ee = w
        self._ensure_schedule(ee)
        freq = self.config.center_frequency
        out = []
        for bs, be in self._bursts:
            a, b = max(bs, es), min(be, ee)
            if b > a:
                out.append((a, b, freq))
        return out


class FrequencyHoppingEmitter(BaseEmitter):
    """Hops between predefined frequencies on a schedule."""

    def is_active(self, time: float) -> bool:
        if time < self.config.start_time:
            return False
        return not (self.config.end_time is not None and time > self.config.end_time)

    def get_frequency(self, time: float) -> float:
        freqs = self.config.hop_frequencies
        if not freqs:
            return self.config.center_frequency
        interval = self.config.hop_interval or 1.0
        idx = int((time - self.config.start_time) / interval) % len(freqs)
        return freqs[idx]

    def activity_intervals(self, start_time, end_time):
        w = _window(self.config, start_time, end_time)
        if not w:
            return []
        es, ee = w
        freqs = self.config.hop_frequencies or [self.config.center_frequency]
        interval = self.config.hop_interval or 1.0
        out = []
        k = int((es - self.config.start_time) // interval)
        while True:
            h0 = self.config.start_time + k * interval
            if h0 > ee:
                break
            a, b = max(h0, es), min(h0 + interval, ee)
            if b > a:
                out.append((a, b, freqs[k % len(freqs)]))
            k += 1
        return out


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

    def activity_intervals(self, start_time, end_time):
        w = _window(self.config, start_time, end_time)
        if not w:
            return []
        es, ee = w
        freqs = self.config.hop_frequencies or [self.config.center_frequency]
        interval = self.config.hop_interval or 1.0
        duty = self.config.duty_cycle or 0.5
        out = []
        k = int((es - self.config.start_time) // interval)
        while True:
            h0 = self.config.start_time + k * interval
            if h0 > ee:
                break
            a, b = max(h0, es), min(h0 + interval * duty, ee)
            if b > a:
                out.append((a, b, freqs[k % len(freqs)]))
            k += 1
        return out


class RadarScanEmitter(BaseEmitter):
    """Rotating-antenna radar seen by an ES receiver.

    A search radar transmits continuously, but its main beam only points at our
    receiver for a brief window (``beam_dwell``) once per antenna rotation
    (``scan_period``). From the receiver's viewpoint the emitter is therefore ON
    for ``beam_dwell`` seconds every ``scan_period`` — a very low duty cycle,
    which is what makes such emitters hard to intercept.

    Optionally frequency-agile: if ``hop_frequencies`` are given, the radar
    changes carrier on each rotation (idx by rotation number).
    """

    def is_active(self, time: float) -> bool:
        if time < self.config.start_time:
            return False
        if self.config.end_time is not None and time > self.config.end_time:
            return False
        scan_period = self.config.scan_period or 1.0
        beam_dwell = self.config.beam_dwell or (scan_period * 0.05)
        phase = (time - self.config.start_time) % scan_period
        return phase < beam_dwell

    def get_frequency(self, time: float) -> float:
        freqs = self.config.hop_frequencies
        if not freqs:
            return self.config.center_frequency
        scan_period = self.config.scan_period or 1.0
        rotation = int((time - self.config.start_time) / scan_period)
        return freqs[rotation % len(freqs)]

    def activity_intervals(self, start_time, end_time):
        w = _window(self.config, start_time, end_time)
        if not w:
            return []
        es, ee = w
        freqs = self.config.hop_frequencies or [self.config.center_frequency]
        scan_period = self.config.scan_period or 1.0
        beam_dwell = self.config.beam_dwell or (scan_period * 0.05)
        out = []
        k = int((es - self.config.start_time) // scan_period)
        while True:
            r0 = self.config.start_time + k * scan_period
            if r0 > ee:
                break
            a, b = max(r0, es), min(r0 + beam_dwell, ee)      # one illumination
            if b > a:
                out.append((a, b, freqs[k % len(freqs)]))
            k += 1
        return out


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
        case EmitterType.RADAR_SCAN:
            return RadarScanEmitter(config, noise_power_dbm)
        case _:
            raise ValueError(f"Unknown emitter type: {config.emitter_type}")
