"""SQLite persistence for band states and observations.

Can run fully in memory (":memory:") for tests, or against a file for durable
history across runs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from smartscan.core.models import BandObservation, BandState


class ScanDatabase:
    """SQLite-backed store for band states and observation history."""

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS band_states (
                band_id INTEGER PRIMARY KEY,
                freq_start REAL, freq_end REAL,
                last_scan_time REAL, last_detection_time REAL,
                hit_count INTEGER, miss_count INTEGER, observation_count INTEGER,
                rolling_activity_prob REAL, estimated_period REAL,
                avg_power_db REAL, avg_snr_db REAL, confidence REAL
            );
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                band_id INTEGER, timestamp REAL, detected INTEGER,
                confidence REAL, peak_power_db REAL, avg_power_db REAL,
                noise_floor_db REAL, estimated_snr_db REAL
            );
            CREATE INDEX IF NOT EXISTS idx_obs_band ON observations(band_id);
            CREATE INDEX IF NOT EXISTS idx_obs_time ON observations(timestamp);
            """
        )
        self._conn.commit()

    def save_band_state(self, state: BandState) -> None:
        self._conn.execute(
            """
            INSERT INTO band_states VALUES (
                :band_id, :freq_start, :freq_end, :last_scan_time, :last_detection_time,
                :hit_count, :miss_count, :observation_count, :rolling_activity_prob,
                :estimated_period, :avg_power_db, :avg_snr_db, :confidence
            )
            ON CONFLICT(band_id) DO UPDATE SET
                last_scan_time=excluded.last_scan_time,
                last_detection_time=excluded.last_detection_time,
                hit_count=excluded.hit_count, miss_count=excluded.miss_count,
                observation_count=excluded.observation_count,
                rolling_activity_prob=excluded.rolling_activity_prob,
                estimated_period=excluded.estimated_period,
                avg_power_db=excluded.avg_power_db, avg_snr_db=excluded.avg_snr_db,
                confidence=excluded.confidence
            """,
            state.model_dump(),
        )
        self._conn.commit()

    def load_band_state(self, band_id: int) -> BandState | None:
        row = self._conn.execute(
            "SELECT * FROM band_states WHERE band_id = ?", (band_id,)
        ).fetchone()
        if row is None:
            return None
        return BandState(**dict(row))

    def save_observation(self, obs: BandObservation) -> None:
        self._conn.execute(
            """
            INSERT INTO observations
                (band_id, timestamp, detected, confidence, peak_power_db,
                 avg_power_db, noise_floor_db, estimated_snr_db)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (obs.band_id, obs.timestamp, int(obs.detected), obs.confidence,
             obs.peak_power_db, obs.avg_power_db, obs.noise_floor_db, obs.estimated_snr_db),
        )
        self._conn.commit()

    def load_observations(self, band_id: int) -> list[BandObservation]:
        rows = self._conn.execute(
            "SELECT * FROM observations WHERE band_id = ? ORDER BY timestamp", (band_id,)
        ).fetchall()
        return [
            BandObservation(
                band_id=r["band_id"], timestamp=r["timestamp"],
                detected=bool(r["detected"]), confidence=r["confidence"],
                peak_power_db=r["peak_power_db"], avg_power_db=r["avg_power_db"],
                noise_floor_db=r["noise_floor_db"], estimated_snr_db=r["estimated_snr_db"],
            )
            for r in rows
        ]

    def observation_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ScanDatabase:
        return self

    def __exit__(self, *args) -> None:
        self.close()
