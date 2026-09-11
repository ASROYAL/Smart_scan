"""Abstract base class for activity predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from smartscan.core.models import BandState


class BasePredictor(ABC):
    """Base class for temporal activity prediction."""

    @abstractmethod
    def predict_activity_probability(
        self,
        band_state: BandState,
        future_time: float,
    ) -> float:
        """Estimate probability that a band will be active at future_time.

        Args:
            band_state: Current state of the band (observation-derived only).
            future_time: Time at which to predict activity.

        Returns:
            Probability in [0, 1].
        """
        ...

    @abstractmethod
    def update(self, band_state: BandState, detected: bool, timestamp: float) -> None:
        """Update predictor with a new observation."""
        ...

    def reset(self) -> None:
        """Reset predictor state."""
