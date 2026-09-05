"""Abstract base class for signal detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from smartscan.core.models import AcquisitionMeta, DetectionResult


class BaseDetector(ABC):
    """Base class for activity detection algorithms."""

    @abstractmethod
    def detect(
        self,
        samples: np.ndarray,
        meta: AcquisitionMeta,
    ) -> DetectionResult:
        """Analyze IQ samples and determine if activity is present.

        Args:
            samples: Complex IQ samples.
            meta: Acquisition metadata (frequency, sample rate, etc.).

        Returns:
            DetectionResult with detected flag, power, SNR, etc.
            Must NOT use any ground-truth information.
        """
        ...
