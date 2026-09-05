"""Simulated RF source — wraps RFEnvironment behind the RFSource interface.

Returns IQ samples only. Ground truth is NOT exposed through this adapter.
"""

from __future__ import annotations

import numpy as np

from smartscan.acquisition.base import RFSource
from smartscan.core.models import AcquisitionMeta
from smartscan.simulation.environment import RFEnvironment


class SimulatedRFSource(RFSource):
    """RF source backed by the simulated environment.

    The environment generates IQ samples; this adapter strips away any
    ground-truth access and presents only the RFSource interface.
    """

    def __init__(self, environment: RFEnvironment) -> None:
        self._env = environment
        self._center_frequency = 0.0
        self._sample_rate = environment._sample_rate

    def read_samples(
        self,
        center_frequency: float,
        bandwidth: float,
        num_samples: int,
    ) -> tuple[np.ndarray, AcquisitionMeta]:
        self._center_frequency = center_frequency
        samples = self._env.generate_samples(
            center_frequency=center_frequency,
            bandwidth=bandwidth,
            num_samples=num_samples,
        )

        meta = AcquisitionMeta(
            center_frequency=center_frequency,
            sample_rate=self._sample_rate,
            bandwidth=bandwidth,
            num_samples=num_samples,
            timestamp=self._env.time,
            dwell_time=num_samples / self._sample_rate,
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
