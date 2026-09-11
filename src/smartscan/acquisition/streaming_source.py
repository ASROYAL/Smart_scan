"""Receive-only streaming RF source boundary for hardware-in-the-loop drivers."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from smartscan.acquisition.base import RFSource
from smartscan.core.models import AcquisitionMeta


class StreamingRFSource(RFSource):
    """Thread-safe bounded IQ ring buffer fed by an external receive-only driver.

    A SoapySDR/UHD/device process can call ``push`` from its receive callback and
    provide a tune handler. This class deliberately contains no transmit operation.
    """

    def __init__(
        self,
        sample_rate: float,
        center_frequency: float,
        *,
        max_buffer_seconds: float = 2.0,
        read_timeout: float = 1.0,
        calibration_db: float = 0.0,
        tune_handler: Callable[[float], None] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._center_frequency = center_frequency
        self._capacity = max(1, int(sample_rate * max_buffer_seconds))
        self._timeout = read_timeout
        self._scale = 10.0 ** (calibration_db / 20.0)
        self._tune_handler = tune_handler
        self._chunks: deque[np.ndarray] = deque()
        self._available = 0
        self._condition = threading.Condition()
        self._start = time.monotonic()
        self.dropped_samples = 0

    def push(self, samples: np.ndarray) -> None:
        chunk = np.asarray(samples, dtype=np.complex128).ravel() * self._scale
        with self._condition:
            merged = np.concatenate([*self._chunks, chunk]) if self._chunks else chunk
            overflow = max(0, len(merged) - self._capacity)
            self.dropped_samples += overflow
            retained = merged[overflow:]
            self._chunks.clear()
            if len(retained):
                self._chunks.append(retained)
            self._available = len(retained)
            self._condition.notify_all()

    def read_samples(
        self, center_frequency: float, bandwidth: float, num_samples: int,
    ) -> tuple[np.ndarray, AcquisitionMeta]:
        deadline = time.monotonic() + self._timeout
        with self._condition:
            while self._available < num_samples:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"IQ underflow: requested {num_samples}, available {self._available}"
                    )
                self._condition.wait(remaining)
            merged = np.concatenate(list(self._chunks))
            output = merged[:num_samples].copy()
            remainder = merged[num_samples:]
            self._chunks.clear()
            if len(remainder):
                self._chunks.append(remainder)
            self._available = len(remainder)
        timestamp = self.get_time()
        return output, AcquisitionMeta(
            center_frequency=center_frequency,
            sample_rate=self._sample_rate,
            bandwidth=bandwidth,
            num_samples=num_samples,
            timestamp=timestamp,
            dwell_time=num_samples / self._sample_rate,
        )

    def tune(self, center_frequency: float) -> None:
        if self._tune_handler is not None:
            self._tune_handler(center_frequency)
        self._center_frequency = center_frequency

    def get_sample_rate(self) -> float:
        return self._sample_rate

    def get_center_frequency(self) -> float:
        return self._center_frequency

    def get_time(self) -> float:
        return time.monotonic() - self._start

    def channel_power(self, samples: np.ndarray, channels: int) -> np.ndarray:
        """Return FFT-channelized mean power for monitoring and coarse selection."""

        if channels <= 0:
            raise ValueError("channels must be positive")
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2
        return np.array([part.mean() for part in np.array_split(spectrum, channels)])
