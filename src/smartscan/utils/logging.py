"""Structured logging for scan events — CSV and JSON export."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("smartscan")


@dataclass
class ScanLogEntry:
    """One row in the scan log."""
    scan_number: int
    timestamp: float
    band_id: int
    center_frequency: float
    dwell_time: float
    scheduler_score: float
    detected: bool
    estimated_snr_db: float
    hit: bool
    reward: float
    processing_latency_ms: float
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ScanLogger:
    """Accumulates scan log entries and exports to CSV/JSON."""

    def __init__(self) -> None:
        self._entries: list[ScanLogEntry] = []

    def log(self, entry: ScanLogEntry) -> None:
        self._entries.append(entry)
        logger.info(
            "scan=%d band=%d detected=%s snr=%.1fdB reward=%.3f",
            entry.scan_number,
            entry.band_id,
            entry.detected,
            entry.estimated_snr_db,
            entry.reward,
        )

    @property
    def entries(self) -> list[ScanLogEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def export_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._entries:
            return
        fieldnames = list(self._entries[0].to_dict().keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self._entries:
                writer.writerow(entry.to_dict())

    def export_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self._entries], f, indent=2)

    def clear(self) -> None:
        self._entries.clear()
