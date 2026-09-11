"""Abstract base class for all scan schedulers.

Schedulers receive band states (derived from observations only) and produce
scan decisions. They NEVER receive ground truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from smartscan.core.models import BandObservation, BandState, ScanDecision


class BaseScheduler(ABC):
    """Base class for all scanning strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable scheduler name."""
        ...

    @abstractmethod
    def select_band(
        self,
        band_states: list[BandState],
        current_time: float,
    ) -> ScanDecision:
        """Choose the next frequency band to scan.

        Args:
            band_states: Current state of all frequency bands (observation-derived only).
            current_time: Current simulation time in seconds.

        Returns:
            ScanDecision describing where to tune next.
        """
        ...

    @abstractmethod
    def update(
        self,
        decision: ScanDecision,
        observation: BandObservation,
        reward: float = 0.0,
    ) -> None:
        """Update internal state after receiving an observation.

        Args:
            decision: The scan decision that was executed.
            observation: The resulting observation (detection result, NOT ground truth).
            reward: The shared shaped reward for this scan (same value the runner
                reports), so every scheduler optimises the same objective.
        """
        ...

    def reset(self) -> None:
        """Reset scheduler to initial state. Override if scheduler has learnable state."""
