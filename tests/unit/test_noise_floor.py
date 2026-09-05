"""Tests for noise floor estimation."""

import numpy as np
import pytest

from smartscan.core.models import NoiseEstMethod
from smartscan.dsp.noise_floor import compute_threshold, estimate_noise_floor


class TestEstimateNoiseFloor:
    def test_median_of_flat_noise(self):
        rng = np.random.default_rng(42)
        psd = rng.normal(-90, 2, 1024)
        est = estimate_noise_floor(psd, method=NoiseEstMethod.MEDIAN)
        assert est == pytest.approx(-90, abs=1.0)

    def test_robust_to_signal_peaks(self):
        """A few strong bins should not move the median much."""
        rng = np.random.default_rng(42)
        psd = rng.normal(-90, 1, 1024)
        # Inject strong signal in 5% of bins
        psd[:50] = -30
        est = estimate_noise_floor(psd, method=NoiseEstMethod.MEDIAN)
        assert est == pytest.approx(-90, abs=2.0)

    def test_percentile_method(self):
        rng = np.random.default_rng(42)
        psd = rng.normal(-90, 2, 1024)
        est = estimate_noise_floor(psd, method=NoiseEstMethod.PERCENTILE, percentile=25)
        # 25th percentile of noise should be slightly below median
        assert est < np.median(psd) + 0.5

    def test_moving_method(self):
        rng = np.random.default_rng(42)
        psd = rng.normal(-85, 1, 512)
        est = estimate_noise_floor(psd, method=NoiseEstMethod.MOVING)
        assert est == pytest.approx(-85, abs=2.0)


class TestComputeThreshold:
    def test_adds_margin(self):
        assert compute_threshold(-90.0, 6.0) == pytest.approx(-84.0)

    def test_zero_margin(self):
        assert compute_threshold(-90.0, 0.0) == pytest.approx(-90.0)
