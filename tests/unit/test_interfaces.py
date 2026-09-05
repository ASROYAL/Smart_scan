"""Tests for abstract interfaces — verify they cannot be instantiated."""

import numpy as np
import pytest

from smartscan.acquisition.base import RFSource
from smartscan.core.models import AcquisitionMeta, BandObservation, BandState, ScanDecision
from smartscan.dsp.base import BaseDetector
from smartscan.prediction.base import BasePredictor
from smartscan.schedulers.base import BaseScheduler


class TestRFSourceABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            RFSource()

    def test_concrete_subclass_works(self):
        class DummySource(RFSource):
            def read_samples(self, cf, bw, n):
                return np.zeros(n, dtype=np.complex128), AcquisitionMeta(
                    center_frequency=cf, sample_rate=1e6, bandwidth=bw,
                    num_samples=n, timestamp=0.0, dwell_time=n / 1e6,
                )
            def tune(self, cf): pass
            def get_sample_rate(self): return 1e6
            def get_center_frequency(self): return 100e6
            def get_time(self): return 0.0

        src = DummySource()
        samples, meta = src.read_samples(100e6, 1e6, 100)
        assert len(samples) == 100
        assert meta.center_frequency == 100e6


class TestBaseSchedulerABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseScheduler()

    def test_concrete_subclass_works(self):
        class DummyScheduler(BaseScheduler):
            @property
            def name(self): return "dummy"
            def select_band(self, band_states, current_time):
                bs = band_states[0]
                return ScanDecision(
                    band_id=bs.band_id, center_frequency=bs.center_frequency,
                    bandwidth=bs.bandwidth, dwell_time=0.01, timestamp=current_time,
                )
            def update(self, decision, observation): pass

        sched = DummyScheduler()
        states = [BandState(band_id=0, freq_start=100e6, freq_end=120e6)]
        decision = sched.select_band(states, 0.0)
        assert decision.band_id == 0


class TestBaseDetectorABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseDetector()


class TestBasePredictorABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BasePredictor()
