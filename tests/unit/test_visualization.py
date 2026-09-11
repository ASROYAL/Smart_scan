"""Smoke tests for visualization helpers — verify figures build without error."""

import numpy as np
import plotly.graph_objects as go

from smartscan.core.config import DetectorConfig
from smartscan.core.models import AcquisitionMeta
from smartscan.evaluation.evaluator import RunArtifacts
from smartscan.evaluation.metrics import ScanRecord
from smartscan.simulation.waveform import generate_tone
from smartscan.visualization import metrics_viz, spectrum, timeline


def make_artifacts(n=20):
    artifacts = RunArtifacts()
    for i in range(n):
        artifacts.records.append(ScanRecord(
            scan_number=i, timestamp=i * 0.01, band_id=i % 5,
            detected=(i % 3 == 0), truly_active=(i % 4 == 0), reward=float(i % 2),
        ))
        artifacts.band_visit_counts[i % 5] = artifacts.band_visit_counts.get(i % 5, 0) + 1
    return artifacts


class TestSpectrumViz:
    def test_spectrum_figure(self):
        samples = generate_tone(4096, sample_rate=20e6, frequency_offset=2e6, power_dbm=-60)
        meta = AcquisitionMeta(
            center_frequency=100e6, sample_rate=20e6, bandwidth=20e6,
            num_samples=4096, timestamp=0.0, dwell_time=4096 / 20e6,
        )
        fig = spectrum.spectrum_figure(samples, meta, DetectorConfig(fft_size=1024))
        assert isinstance(fig, go.Figure)

    def test_waterfall_figure(self):
        psd = np.random.randn(10, 64)
        freqs = np.linspace(90, 110, 64)
        times = np.linspace(0, 1, 10)
        fig = spectrum.waterfall_figure(psd, freqs, times)
        assert isinstance(fig, go.Figure)

    def test_band_activity_heatmap(self):
        activity = np.random.rand(10, 5)
        fig = spectrum.band_activity_heatmap(activity, 5, np.linspace(0, 1, 10))
        assert isinstance(fig, go.Figure)


class TestTimelineViz:
    def test_band_selection_timeline(self):
        fig = timeline.band_selection_timeline(make_artifacts())
        assert isinstance(fig, go.Figure)

    def test_cumulative_detections(self):
        fig = timeline.cumulative_detections(make_artifacts())
        assert isinstance(fig, go.Figure)

    def test_reward_over_time(self):
        fig = timeline.reward_over_time(make_artifacts())
        assert isinstance(fig, go.Figure)

    def test_true_vs_detected(self):
        fig = timeline.true_vs_detected_timeline(make_artifacts(), band_id=0)
        assert isinstance(fig, go.Figure)


class TestMetricsViz:
    def test_discovery_delay_distribution(self):
        fig = metrics_viz.discovery_delay_distribution([0.1, 0.2, 0.3])
        assert isinstance(fig, go.Figure)

    def test_discovery_delay_empty(self):
        fig = metrics_viz.discovery_delay_distribution([])
        assert isinstance(fig, go.Figure)

    def test_priority_heatmap(self):
        pm = np.random.rand(10, 5)
        fig = metrics_viz.priority_heatmap(pm, 5, np.linspace(0, 1, 10))
        assert isinstance(fig, go.Figure)
