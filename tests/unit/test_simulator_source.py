"""Tests for SimulatedRFSource — verifying the information wall."""

import inspect

import numpy as np
import pytest

from smartscan.acquisition.simulator_source import SimulatedRFSource
from smartscan.core.models import AcquisitionMeta, EmitterConfig, EmitterType
from smartscan.simulation.environment import RFEnvironment


def make_source():
    emitters = [
        EmitterConfig(
            emitter_id=0, emitter_type=EmitterType.CONTINUOUS,
            center_frequency=100e6, bandwidth=5e6, amplitude=1.0, snr_db=20.0,
        ),
    ]
    env = RFEnvironment(emitters, noise_power_dbm=-100.0, sample_rate=20e6, seed=42)
    return SimulatedRFSource(env), env


class TestSimulatedRFSource:
    def test_returns_complex_samples(self):
        src, _ = make_source()
        samples, meta = src.read_samples(100e6, 20e6, 1000)
        assert np.iscomplexobj(samples)
        assert len(samples) == 1000

    def test_returns_acquisition_meta(self):
        src, _ = make_source()
        _, meta = src.read_samples(100e6, 20e6, 1000)
        assert isinstance(meta, AcquisitionMeta)
        assert meta.center_frequency == 100e6
        assert meta.bandwidth == 20e6
        assert meta.num_samples == 1000

    def test_tune(self):
        src, _ = make_source()
        src.tune(200e6)
        assert src.get_center_frequency() == 200e6

    def test_sample_rate(self):
        src, _ = make_source()
        assert src.get_sample_rate() == 20e6


class TestInformationSeparation:
    """Verify that the RFSource interface does NOT leak ground truth."""

    def test_meta_has_no_ground_truth_fields(self):
        src, _ = make_source()
        _, meta = src.read_samples(100e6, 20e6, 100)
        meta_dict = meta.model_dump()
        forbidden_keys = {"active", "emitters", "ground_truth", "is_active", "label", "truth"}
        found = set(meta_dict.keys()) & forbidden_keys
        assert not found, f"Meta leaks ground truth via keys: {found}"

    def test_source_has_no_ground_truth_methods(self):
        src, _ = make_source()
        public_methods = [
            m for m in dir(src) if not m.startswith("_") and callable(getattr(src, m))
        ]
        forbidden = {"get_ground_truth", "get_active_emitters", "is_active",
                      "get_truth", "ground_truth"}
        found = set(public_methods) & forbidden
        assert not found, f"Source exposes ground truth methods: {found}"

    def test_return_type_is_only_samples_and_meta(self):
        src, _ = make_source()
        result = src.read_samples(100e6, 20e6, 100)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], AcquisitionMeta)

    def test_samples_contain_no_metadata_attributes(self):
        """IQ samples should be plain numpy arrays, not custom objects with labels."""
        src, _ = make_source()
        samples, _ = src.read_samples(100e6, 20e6, 100)
        assert type(samples) is np.ndarray
        assert not hasattr(samples, "ground_truth")
        assert not hasattr(samples, "is_active")
        assert not hasattr(samples, "label")
