"""Timing instrumentation for profiling pipeline stages."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TimingStats:
    """Accumulated timing statistics for one named stage."""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count

    def record(self, elapsed_ms: float) -> None:
        self.count += 1
        self.total_ms += elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)


class TimingInstrument:
    """Collects per-stage timing measurements across the pipeline."""

    def __init__(self) -> None:
        self._stats: dict[str, TimingStats] = defaultdict(TimingStats)

    @contextmanager
    def measure(self, stage: str):
        """Context manager that records elapsed time for a named stage."""
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._stats[stage].record(elapsed_ms)

    def get_stats(self, stage: str) -> TimingStats:
        return self._stats[stage]

    def all_stats(self) -> dict[str, TimingStats]:
        return dict(self._stats)

    def summary(self) -> str:
        lines = ["Pipeline Timing Summary", "=" * 60]
        for stage, stats in sorted(self._stats.items()):
            lines.append(
                f"  {stage:30s}  calls={stats.count:5d}  "
                f"avg={stats.avg_ms:8.2f}ms  "
                f"min={stats.min_ms:8.2f}ms  "
                f"max={stats.max_ms:8.2f}ms"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        self._stats.clear()
