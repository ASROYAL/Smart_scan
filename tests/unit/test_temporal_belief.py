"""Tests for observation-derived temporal belief fusion."""

from smartscan.core.models import BandObservation, BandState
from smartscan.prediction.temporal_belief import TemporalBeliefEnsemble


def _observation(detected: bool) -> BandObservation:
    return BandObservation(
        band_id=0, timestamp=1.0, detected=detected, confidence=0.9,
        peak_power_db=-50, avg_power_db=-60, noise_floor_db=-100, estimated_snr_db=20,
    )


def test_repeated_observed_hits_raise_temporal_belief():
    state = BandState(band_id=0, freq_start=90e6, freq_end=110e6, last_scan_time=1.0)
    model = TemporalBeliefEnsemble(1)
    before = model.predict(state, 1.0)
    for _ in range(5):
        model.update(_observation(True))
    assert model.predict(state, 1.0) > before
