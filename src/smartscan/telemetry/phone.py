"""Phone-link state and alert rules shared by the dashboard and simulator.

The values describe a locally paired, consented training beacon. RSSI supports
relative signal trend only; it does not provide bearing or geographic position.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LinkState(str, Enum):
    OFFLINE = "OFFLINE"
    PAIRED = "PAIRED"
    ACTIVE = "ACTIVE"
    LIVE = "LIVE RSSI"


class AlertLevel(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    CRITICAL = "WAR MODE"


@dataclass(frozen=True)
class AlertMemory:
    """Debounced alert state; streak prevents threshold noise from flapping."""

    level: AlertLevel = AlertLevel.NORMAL
    candidate: AlertLevel = AlertLevel.NORMAL
    streak: int = 0


def rssi_to_quality(rssi_dbm: int | None) -> tuple[str, int]:
    """Map RSSI to a bounded display score and descriptive quality."""

    if rssi_dbm is None:
        return "UNAVAILABLE", 0
    percent = max(0, min(100, round((rssi_dbm + 100) * 100 / 60)))
    if rssi_dbm >= -55:
        label = "EXCELLENT"
    elif rssi_dbm >= -67:
        label = "GOOD"
    elif rssi_dbm >= -78:
        label = "FAIR"
    else:
        label = "WEAK"
    return label, percent


def classify_link(
    *, paired: bool, explicitly_connected: bool, rssi_dbm: int | None, sample_age: float | None
) -> LinkState:
    """Classify link state without treating a missing macOS flag as disconnected."""

    if rssi_dbm is not None and sample_age is not None and sample_age <= 3.0:
        return LinkState.LIVE
    if explicitly_connected:
        return LinkState.ACTIVE
    if paired:
        return LinkState.PAIRED
    return LinkState.OFFLINE


def _desired_level(current: AlertLevel, quality: int) -> AlertLevel:
    # Hysteresis: CRITICAL clears below 86; CAUTION clears below 76.
    if current == AlertLevel.CRITICAL:
        if quality >= 86:
            return AlertLevel.CRITICAL
        return AlertLevel.CAUTION if quality >= 76 else AlertLevel.NORMAL
    if current == AlertLevel.CAUTION:
        if quality > 90:
            return AlertLevel.CRITICAL
        return AlertLevel.CAUTION if quality >= 76 else AlertLevel.NORMAL
    if quality > 90:
        return AlertLevel.CRITICAL
    if quality >= 80:
        return AlertLevel.CAUTION
    return AlertLevel.NORMAL


def advance_alert(
    memory: AlertMemory, quality: int, confirmation_samples: int = 2
) -> AlertMemory:
    """Advance the 80/90 percent alert state with debounce and hysteresis."""

    desired = _desired_level(memory.level, quality)
    if desired == memory.level:
        return AlertMemory(level=memory.level, candidate=memory.level, streak=0)
    streak = memory.streak + 1 if memory.candidate == desired else 1
    if streak >= max(1, confirmation_samples):
        return AlertMemory(level=desired, candidate=desired, streak=0)
    return AlertMemory(level=memory.level, candidate=desired, streak=streak)


def quality_to_snr(quality: int) -> float:
    """Map a telemetry score into the detector's useful synthetic-SNR range.

    The training target is a 5 MHz digital-like waveform.  Calibration sweeps
    place the energy detector's transition between roughly -3 and 0 dB.  Keep
    the full 0--100 proxy range around that transition so different inputs
    change detectability instead of all saturating at probability one.
    """

    bounded = max(0, min(100, quality))
    return -6.0 + bounded * 0.06
