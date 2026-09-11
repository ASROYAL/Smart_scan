"""RF source backed by a PDW environment (radar ESM data).

Presents the same ``RFSource`` interface as ``SimulatedRFSource`` so the receiver,
DSP, scheduler and evaluator are all identical — only the underlying truth comes
from pulse descriptor words instead of the tone simulator.
"""

from __future__ import annotations

import numpy as np

from smartscan.acquisition.base import RFSource
from smartscan.core.models import AcquisitionMeta
from smartscan.simulation.pdw import PDWEnvironment


class PDWRFSource(RFSource):
    """RFSource over a PDWEnvironment. Returns IQ + metadata only, never labels."""

    def __init__(self, environment: PDWEnvironment) -> None:
        self._env = environment
        self._center_frequency = 0.0
        self._sample_rate = environment._sample_rate

    def read_samples(
        self, center_frequency: float, bandwidth: float, num_samples: int,
    ) -> tuple[np.ndarray, AcquisitionMeta]:
        self._center_frequency = center_frequency
        samples = self._env.generate_samples(
            center_frequency=center_frequency, bandwidth=bandwidth, num_samples=num_samples,
        )
        meta = AcquisitionMeta(
            center_frequency=center_frequency, sample_rate=self._sample_rate,
            bandwidth=bandwidth, num_samples=num_samples,
            timestamp=self._env.time, dwell_time=num_samples / self._sample_rate,
        )
        return samples, meta

    def tune(self, center_frequency: float) -> None:
        self._center_frequency = center_frequency

    def get_sample_rate(self) -> float:
        return self._sample_rate

    def get_center_frequency(self) -> float:
        return self._center_frequency

    def get_time(self) -> float:
        return self._env.time
