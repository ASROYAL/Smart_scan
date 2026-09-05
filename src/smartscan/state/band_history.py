"""Per-band observation history with rolling statistics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from smartscan.core.models import BandObservation


@dataclass
class BandHistory:
    """Time-indexed observation history for a single frequency band."""

    band_id: int
    max_records: int = 500
    _observations: deque[BandObservation] = field(default_factory=deque)
    detection_times: list[float] = field(default_factory=list)

    def add(self, obs: BandObservation) -> None:
        if len(self._observations) >= self.max_records:
            self._observations.popleft()
        self._observations.append(obs)
        if obs.detected:
            self.detection_times.append(obs.timestamp)

    @property
    def observations(self) -> list[BandObservation]:
        return list(self._observations)

    def recent(self, n: int) -> list[BandObservation]:
        """Return the most recent n observations."""
        obs = list(self._observations)
        return obs[-n:]

    def recent_hits(self, n: int) -> int:
        return sum(1 for o in self.recent(n) if o.detected)

    def recent_misses(self, n: int) -> int:
        return sum(1 for o in self.recent(n) if not o.detected)

    def detection_sequence(self, n: int) -> list[bool]:
        return [o.detected for o in self.recent(n)]

    def inter_detection_intervals(self) -> list[float]:
        """Time gaps between consecutive detections."""
        if len(self.detection_times) < 2:
            return []
        times = sorted(self.detection_times)
        return [times[i + 1] - times[i] for i in range(len(times) - 1)]

    def __len__(self) -> int:
        return len(self._observations)
