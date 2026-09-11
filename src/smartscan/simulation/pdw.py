"""Pulse Descriptor Word (PDW) environment and loaders.

Radar ESM datasets — such as the Alan Turing Institute's *Turing Synthetic Radar
Dataset* (Scan Mode) — are distributed as **Pulse Descriptor Words**: one row per
intercepted pulse, with time-of-arrival, RF frequency, pulse width, angle-of-arrival
and amplitude, plus an emitter label. This module turns such a PDW table into an
``RFEnvironment``-compatible source so the *same* DSP → detector → state → scheduler
→ evaluator pipeline runs on real recorded pulse data with no changes downstream.

The mapping is exactly the one the EW problem statement describes: for each
frequency band and time slot, the environment answers *transmission* or
*non-transmission* from the pulses that fall in that band/time.

Two ways to obtain PDWs:
  * ``generate_synthetic_pdws`` — seeded synthetic radars (for demos/tests, no
    external download);
  * ``load_pdws_csv`` — read a CSV whose columns map to the Turing schema.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smartscan.core.models import GroundTruthEvent
from smartscan.simulation.noise import generate_awgn, power_for_snr
from smartscan.simulation.waveform import generate_tone


@dataclass(frozen=True)
class PulseDescriptorWord:
    """One intercepted radar pulse (the Turing dataset's 5-D vector + label)."""
    toa: float            # time of arrival, seconds
    frequency: float      # RF centre frequency, Hz
    pulse_width: float    # seconds
    amplitude: float      # linear
    emitter_id: int       # ground-truth label
    aoa: float = 0.0      # angle of arrival, degrees (unused by the scan scheduler)
    snr_db: float = 25.0

    @property
    def toe(self) -> float:
        """Time of end (TOA + pulse width)."""
        return self.toa + self.pulse_width


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# Column-name aliases so a Turing-style export loads without hand-editing.
_COL_ALIASES = {
    "toa": ("toa", "time_of_arrival", "time", "t"),
    "frequency": ("frequency", "freq", "rf", "rf_freq", "center_frequency", "cf"),
    "pulse_width": ("pulse_width", "pw", "width"),
    "amplitude": ("amplitude", "amp", "power", "pa"),
    "emitter_id": ("emitter_id", "emitter", "label", "cluster", "id"),
    "aoa": ("aoa", "angle_of_arrival", "angle", "doa"),
}


def _resolve(fieldnames: list[str], key: str) -> str | None:
    lower = {f.lower(): f for f in fieldnames}
    for alias in _COL_ALIASES[key]:
        if alias in lower:
            return lower[alias]
    return None


def load_pdws_csv(path: str | Path, default_snr_db: float = 25.0) -> list[PulseDescriptorWord]:
    """Load PDWs from a CSV file.

    Recognised columns (case-insensitive, aliases accepted): toa, frequency,
    pulse_width, amplitude, emitter_id, aoa. Missing optional columns default
    sensibly. Frequency is assumed in Hz and time in seconds.
    """
    path = Path(path)
    pdws: list[PulseDescriptorWord] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return pdws
        cols = {k: _resolve(list(reader.fieldnames), k) for k in _COL_ALIASES}
        if cols["toa"] is None or cols["frequency"] is None:
            raise ValueError(
                f"PDW CSV must have time-of-arrival and frequency columns; got {reader.fieldnames}"
            )
        for row in reader:
            toa = float(row[cols["toa"]])
            freq = float(row[cols["frequency"]])
            pw = float(row[cols["pulse_width"]]) if cols["pulse_width"] else 1e-6
            amp = float(row[cols["amplitude"]]) if cols["amplitude"] else 1.0
            eid = int(float(row[cols["emitter_id"]])) if cols["emitter_id"] else 0
            aoa = float(row[cols["aoa"]]) if cols["aoa"] else 0.0
            pdws.append(PulseDescriptorWord(
                toa=toa, frequency=freq, pulse_width=pw, amplitude=amp,
                emitter_id=eid, aoa=aoa, snr_db=default_snr_db,
            ))
    pdws.sort(key=lambda p: p.toa)
    return pdws


def save_pdws_csv(pdws: list[PulseDescriptorWord], path: str | Path) -> None:
    """Write PDWs to a CSV file (Turing-compatible column names)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["toa", "frequency", "pulse_width", "amplitude", "aoa", "emitter_id"])
        for p in pdws:
            w.writerow([p.toa, p.frequency, p.pulse_width, p.amplitude, p.aoa, p.emitter_id])


# ---------------------------------------------------------------------------
# Synthetic generator (so the pipeline demos without the 70 GB download)
# ---------------------------------------------------------------------------

def generate_synthetic_pdws(
    num_emitters: int = 6,
    duration: float = 20.0,
    total_bw: float = 1000e6,
    center: float = 500e6,
    seed: int = 42,
    snr_db: float = 25.0,
) -> list[PulseDescriptorWord]:
    """Generate a seeded PDW stream of scanning / agile radars.

    Each emitter has an antenna scan period and a pulse repetition interval (PRI);
    it emits a short burst of pulses each time its beam sweeps past the receiver.
    Roughly mirrors the Turing Scan-Mode structure (agile + scanning emitters).
    """
    rng = np.random.default_rng(seed)
    pdws: list[PulseDescriptorWord] = []
    lo = center - total_bw / 2
    hi = center + total_bw / 2

    for eid in range(num_emitters):
        base_freq = rng.uniform(lo + 20e6, hi - 20e6)
        scan_period = rng.uniform(1.5, 4.0)          # antenna rotation, s
        beam_dwell = rng.uniform(0.1, 0.35)           # illumination per rotation, s
        pri = rng.uniform(1e-3, 5e-3)                 # pulse repetition interval, s
        pw = rng.uniform(0.5e-6, 5e-6)                # pulse width, s
        agile = rng.random() < 0.4                    # some emitters are frequency-agile
        agile_set = [base_freq, rng.uniform(lo + 20e6, hi - 20e6)]

        t0 = 0.0
        rotation = 0
        while t0 < duration:
            freq = agile_set[rotation % 2] if agile else base_freq
            # emit pulses during the beam dwell
            t = t0
            beam_end = min(t0 + beam_dwell, duration)
            while t < beam_end:
                pdws.append(PulseDescriptorWord(
                    toa=t, frequency=freq, pulse_width=pw, amplitude=1.0,
                    emitter_id=eid, aoa=float(rng.uniform(0, 360)), snr_db=snr_db,
                ))
                t += pri
            t0 += scan_period
            rotation += 1

    pdws.sort(key=lambda p: p.toa)
    return pdws


# ---------------------------------------------------------------------------
# PDW environment (RFEnvironment-compatible, duck-typed)
# ---------------------------------------------------------------------------

class PDWEnvironment:
    """Serves IQ + ground truth from a PDW table, mirroring ``RFEnvironment``.

    A frequency band is "transmitting" at time *t* if any pulse overlaps that
    band and is on at *t* (its [TOA, TOA+PW] window contains *t*, widened by a
    small dwell tolerance so a scan that lands mid-PRI still sees the emitter).
    """

    def __init__(
        self,
        pdws: list[PulseDescriptorWord],
        noise_power_dbm: float = -100.0,
        sample_rate: float = 20e6,
        seed: int = 42,
        dwell_tolerance: float = 0.02,
        pulse_bandwidth: float = 5e6,
    ) -> None:
        self._pdws = sorted(pdws, key=lambda p: p.toa)
        self._toas = np.array([p.toa for p in self._pdws]) if self._pdws else np.array([])
        self._noise_power_dbm = noise_power_dbm
        self._sample_rate = sample_rate
        self._rng = np.random.default_rng(seed)
        self._time = 0.0
        self._dwell_tol = dwell_tolerance
        self._pulse_bw = pulse_bandwidth

    # --- clock (matches RFEnvironment) ---
    @property
    def time(self) -> float:
        return self._time

    def advance_time(self, dt: float) -> None:
        self._time += dt

    def set_time(self, t: float) -> None:
        self._time = t

    # --- active-pulse query (explicit interval overlap) ---
    def _integration_window(self, time: float, dwell: float) -> tuple[float, float]:
        """The receiver's observation interval [t0, t1] for this scan.

        Uses the ACTUAL dwell when provided; a point query (dwell=0) falls back to
        ``self._dwell_tol`` as a minimum integration window (documented, not a
        fixed ±tolerance around the time).
        """
        eff = dwell if dwell > 0 else self._dwell_tol
        return time, time + eff

    def _pulses_overlapping(self, t0: float, t1: float) -> list[PulseDescriptorWord]:
        """Pulses whose [TOA, TOA+PW] interval overlaps the scan window [t0, t1]."""
        if len(self._toas) == 0:
            return []
        lo = np.searchsorted(self._toas, t0 - 1.0)   # generous left bound (PW << 1 s)
        hi = np.searchsorted(self._toas, t1, side="right")
        return [p for p in self._pdws[lo:hi] if p.toe >= t0 and p.toa <= t1]

    def generate_samples(
        self,
        center_frequency: float,
        bandwidth: float,
        num_samples: int,
        time: float | None = None,
    ) -> np.ndarray:
        """Noise + a representative carrier per emitter with a pulse in the dwell.

        NOTE: this is an *abstract binary-activity* model — a pulse overlapping the
        receiver's dwell makes the emitter detectable (one carrier per emitter per
        dwell), rather than depositing only the microseconds of true pulse energy
        (which an energy detector could not see in a 10 ms dwell). It exercises the
        scheduler against realistic pulse *timing* while keeping detection tractable.
        """
        if time is None:
            time = self._time
        dwell = num_samples / self._sample_rate
        t0, t1 = self._integration_window(time, dwell)
        samples = generate_awgn(num_samples, self._noise_power_dbm, self._rng)
        seen_emitters: set[int] = set()
        for p in self._pulses_overlapping(t0, t1):
            offset = p.frequency - center_frequency
            if abs(offset) > bandwidth / 2 or p.emitter_id in seen_emitters:
                continue
            seen_emitters.add(p.emitter_id)
            samples += generate_tone(
                num_samples=num_samples, sample_rate=self._sample_rate,
                frequency_offset=offset, power_dbm=power_for_snr(self._noise_power_dbm, p.snr_db),
            )
        return samples

    # --- ground truth (evaluation only, matches RFEnvironment) ---
    def is_any_active_in_band(self, time, freq_start, freq_end, dwell: float = 0.0) -> bool:
        t0, t1 = self._integration_window(time, dwell)
        for p in self._pulses_overlapping(t0, t1):
            if freq_end > p.frequency - self._pulse_bw / 2 and freq_start < p.frequency + self._pulse_bw / 2:
                return True
        return False

    def get_ground_truth_at(
        self, time: float, freq_start: float, freq_end: float, dwell: float = 0.0,
    ) -> list[GroundTruthEvent]:
        t0, t1 = self._integration_window(time, dwell)
        half_bw = self._pulse_bw / 2
        return [
            GroundTruthEvent(
                emitter_id=p.emitter_id,
                freq_start=max(p.frequency - half_bw, freq_start),
                freq_end=min(p.frequency + half_bw, freq_end),
                time_start=max(p.toa, t0), time_end=min(p.toe, t1),
                amplitude=p.amplitude, snr_db=p.snr_db,
            )
            for p in self._pulses_overlapping(t0, t1)
            if freq_end > p.frequency - half_bw and freq_start < p.frequency + half_bw
        ]

    def get_all_ground_truth(self, start_time: float, end_time: float) -> list[GroundTruthEvent]:
        """Group consecutive pulses of one emitter into illumination events."""
        by_emitter: dict[int, list[PulseDescriptorWord]] = {}
        for p in self._pdws:
            if start_time <= p.toa <= end_time:
                by_emitter.setdefault(p.emitter_id, []).append(p)

        events: list[GroundTruthEvent] = []
        for eid, pulses in by_emitter.items():
            pulses.sort(key=lambda p: p.toa)
            group_start = pulses[0].toa
            prev = pulses[0].toa
            freq = pulses[0].frequency
            gap = 0.05  # pulses more than 50 ms apart start a new illumination event
            for p in pulses[1:]:
                if p.toa - prev > gap or abs(p.frequency - freq) > 1e3:
                    events.append(GroundTruthEvent(
                        emitter_id=eid, freq_start=freq - self._pulse_bw / 2,
                        freq_end=freq + self._pulse_bw / 2,
                        time_start=group_start, time_end=prev,
                        amplitude=1.0, snr_db=pulses[0].snr_db,
                    ))
                    group_start = p.toa
                    freq = p.frequency
                prev = p.toa
            events.append(GroundTruthEvent(
                emitter_id=eid, freq_start=freq - self._pulse_bw / 2,
                freq_end=freq + self._pulse_bw / 2,
                time_start=group_start, time_end=prev,
                amplitude=1.0, snr_db=pulses[0].snr_db,
            ))
        events.sort(key=lambda e: e.time_start)
        return events
