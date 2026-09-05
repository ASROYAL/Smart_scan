"""Recorded IQ file source.

Reads complex64 binary IQ files with metadata supplied separately (JSON sidecar).
The SAME DSP pipeline processes these samples as processes simulated samples.

File format:
    - IQ data: raw complex64 binary (interleaved float32 I, float32 Q), little-endian.
    - Metadata: a JSON sidecar file `<name>.json` with:
        {
          "sample_rate": 20000000.0,
          "center_frequency": 100000000.0,
          "start_time": 0.0
        }

This adapter models a receiver reading windows of a recording. Tuning to a
different center frequency is only meaningful within the recorded band; requests
outside the recorded band return the available samples at the recorded center.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from smartscan.acquisition.base import RFSource
from smartscan.core.models import AcquisitionMeta


class RecordedIQSource(RFSource):
    """RF source backed by a recorded complex64 IQ file."""

    def __init__(self, iq_path: str | Path, meta_path: str | Path | None = None) -> None:
        self._iq_path = Path(iq_path)
        if not self._iq_path.exists():
            raise FileNotFoundError(f"IQ file not found: {self._iq_path}")

        if meta_path is None:
            meta_path = self._iq_path.with_suffix(".json")
        self._meta_path = Path(meta_path)
        if not self._meta_path.exists():
            raise FileNotFoundError(f"Metadata sidecar not found: {self._meta_path}")

        with open(self._meta_path) as f:
            meta = json.load(f)
        self._sample_rate = float(meta["sample_rate"])
        self._recorded_center = float(meta["center_frequency"])
        self._start_time = float(meta.get("start_time", 0.0))

        # Load IQ data (complex64)
        self._iq = np.fromfile(self._iq_path, dtype=np.complex64)
        self._read_pos = 0
        self._center_frequency = self._recorded_center
        self._time = self._start_time

    @property
    def num_samples_available(self) -> int:
        return len(self._iq)

    @property
    def recorded_center_frequency(self) -> float:
        return self._recorded_center

    def read_samples(
        self,
        center_frequency: float,
        bandwidth: float,
        num_samples: int,
    ) -> tuple[np.ndarray, AcquisitionMeta]:
        self._center_frequency = center_frequency

        # Read a contiguous window, wrapping around if we run past the end
        end = self._read_pos + num_samples
        if end <= len(self._iq):
            samples = self._iq[self._read_pos:end].copy()
        else:
            # Wrap around to the start (streaming loop over the recording)
            first = self._iq[self._read_pos:]
            remaining = num_samples - len(first)
            second = self._iq[:remaining]
            samples = np.concatenate([first, second])
        self._read_pos = end % max(len(self._iq), 1)

        meta = AcquisitionMeta(
            center_frequency=center_frequency,
            sample_rate=self._sample_rate,
            bandwidth=bandwidth,
            num_samples=len(samples),
            timestamp=self._time,
            dwell_time=len(samples) / self._sample_rate,
        )
        self._time += meta.dwell_time
        return samples.astype(np.complex128), meta

    def tune(self, center_frequency: float) -> None:
        self._center_frequency = center_frequency

    def get_sample_rate(self) -> float:
        return self._sample_rate

    def get_center_frequency(self) -> float:
        return self._center_frequency

    def get_time(self) -> float:
        return self._time


def write_iq_file(
    iq: np.ndarray,
    iq_path: str | Path,
    sample_rate: float,
    center_frequency: float,
    start_time: float = 0.0,
) -> None:
    """Write complex IQ samples to a complex64 binary file with a JSON sidecar."""
    iq_path = Path(iq_path)
    iq_path.parent.mkdir(parents=True, exist_ok=True)
    iq.astype(np.complex64).tofile(iq_path)

    meta = {
        "sample_rate": sample_rate,
        "center_frequency": center_frequency,
        "start_time": start_time,
    }
    with open(iq_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
