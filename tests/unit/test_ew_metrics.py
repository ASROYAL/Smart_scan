"""Tests for EW-vocabulary figures of merit."""

import pytest

from smartscan.core.config import DetectorConfig
from smartscan.core.models import AcquisitionMeta
from smartscan.dsp.detector import EnergyDetector
from smartscan.evaluation.metrics import (
    ScanRecord,
    average_intercept_time_error,
    intercept_rate,
    percentage_correct_predictions,
    sensitivity_curve,
)


def _rec(predicted, truly, detected=None):
    return ScanRecord(scan_number=0, timestamp=0.0, band_id=0,
                      detected=detected if detected is not None else truly,
                      truly_active=truly, reward=0.0, predicted_active=predicted)


class TestPercentageCorrectPredictions:
    def test_all_correct(self):
        recs = [_rec(True, True), _rec(False, False)]
        assert percentage_correct_predictions(recs) == 1.0

    def test_half_correct(self):
        recs = [_rec(True, True), _rec(True, False)]
        assert percentage_correct_predictions(recs) == 0.5

    def test_empty(self):
        assert percentage_correct_predictions([]) == 0.0


class TestInterceptTimeError:
    def test_basic_error(self):
        predicted = {0: 10.0, 1: 20.0}
        actual = {0: 11.0, 1: 18.0}
        # errors: 1.0, 2.0 -> mean 1.5
        assert average_intercept_time_error(predicted, actual) == pytest.approx(1.5)

    def test_missing_actual_ignored(self):
        predicted = {0: 10.0, 1: 20.0}
        actual = {0: 12.0}  # band 1 has no truth
        assert average_intercept_time_error(predicted, actual) == pytest.approx(2.0)

    def test_no_overlap(self):
        assert average_intercept_time_error({0: 1.0}, {5: 2.0}) == 0.0


class TestInterceptRate:
    def test_alias_of_hit_rate(self):
        recs = [_rec(True, True, detected=True), _rec(True, True, detected=False)]
        assert intercept_rate(recs) == 0.5


class TestSensitivityCurve:
    def test_pd_increases_with_snr(self):
        det = EnergyDetector(DetectorConfig(fft_size=1024, threshold_margin_db=6.0))

        def make_meta(n):
            return AcquisitionMeta(center_frequency=100e6, sample_rate=20e6,
                                   bandwidth=20e6, num_samples=n, timestamp=0.0,
                                   dwell_time=n / 20e6)

        curve = sensitivity_curve(det, make_meta, snr_values=[-10.0, 0.0, 30.0],
                                  trials=15, num_samples=2048, seed=1)
        assert len(curve) == 3
        pd_low = curve[0][1]    # very low SNR
        pd_high = curve[-1][1]  # high SNR
        assert pd_high >= pd_low
        assert pd_high > 0.8    # strong signal is reliably detected
        # PFA (third element) is a valid probability
        assert 0.0 <= curve[0][2] <= 1.0
