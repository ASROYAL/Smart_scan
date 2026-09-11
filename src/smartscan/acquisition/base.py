"""Abstract RF source interface — the hardware abstraction layer.

Any RF data source (simulator, recorded file, future SDR) implements this.
The interface returns ONLY IQ samples and acquisition metadata.
It NEVER exposes ground truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from smartscan.core.models import AcquisitionMeta


class RFSource(ABC):
    """Abstract base class for all RF sample sources."""

    @abstractmethod
    def read_samples(
        self,
        center_frequency: float,
        bandwidth: float,
        num_samples: int,
    ) -> tuple[np.ndarray, AcquisitionMeta]:
        """Acquire IQ samples at the specified frequency.

        Returns:
            Tuple of (complex IQ samples, acquisition metadata).
            The IQ array has dtype complex64 or complex128.
            Metadata contains NO ground-truth information.
        """
        ...

    @abstractmethod
    def tune(self, center_frequency: float) -> None:
        """Tune the source to a new center frequency."""
        ...

    @abstractmethod
    def get_sample_rate(self) -> float:
        """Return the current sample rate in samples/second."""
        ...

    @abstractmethod
    def get_center_frequency(self) -> float:
        """Return the current center frequency in Hz."""
        ...

    @abstractmethod
    def get_time(self) -> float:
        """Return the current simulation/acquisition time in seconds."""
        ...

    def close(self) -> None:
        """Release any resources. Default is no-op."""
