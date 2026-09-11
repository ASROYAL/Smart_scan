"""External, consented telemetry used by training and demonstration workflows."""

from smartscan.telemetry.phone import (
    AlertLevel,
    AlertMemory,
    LinkState,
    advance_alert,
    classify_link,
    quality_to_snr,
    rssi_to_quality,
)

__all__ = [
    "AlertLevel",
    "AlertMemory",
    "LinkState",
    "advance_alert",
    "classify_link",
    "quality_to_snr",
    "rssi_to_quality",
]
