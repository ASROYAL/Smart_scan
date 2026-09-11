"""Spectrum and waterfall visualizations using plotly."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from smartscan.core.config import DetectorConfig
from smartscan.core.models import AcquisitionMeta
from smartscan.dsp.noise_floor import compute_threshold, estimate_noise_floor
from smartscan.dsp.psd import compute_psd


def spectrum_figure(
    samples: np.ndarray,
    meta: AcquisitionMeta,
    detector_config: DetectorConfig,
) -> go.Figure:
    """Plot the PSD of an observation with noise floor and threshold lines."""
    freqs, psd_db = compute_psd(
        samples=samples,
        sample_rate=meta.sample_rate,
        method=detector_config.psd_method,
        window=detector_config.window,
        fft_size=detector_config.fft_size,
        center_frequency=meta.center_frequency,
    )
    noise_floor = estimate_noise_floor(
        psd_db, method=detector_config.noise_method, percentile=detector_config.percentile,
    )
    threshold = compute_threshold(noise_floor, detector_config.threshold_margin_db)

    freqs_mhz = freqs / 1e6
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs_mhz, y=psd_db, mode="lines", name="PSD",
                              line={"color": "#2c7fb8"}))
    fig.add_hline(y=noise_floor, line_dash="dot", line_color="gray",
                  annotation_text="noise floor")
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text="threshold")
    fig.update_layout(
        title=f"Spectrum @ {meta.center_frequency/1e6:.1f} MHz",
        xaxis_title="Frequency (MHz)",
        yaxis_title="Power (dB)",
        height=400,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def waterfall_figure(
    psd_matrix: np.ndarray,
    freqs_mhz: np.ndarray,
    times: np.ndarray,
) -> go.Figure:
    """Spectrogram/waterfall heatmap of PSD over time.

    Args:
        psd_matrix: Shape (num_time_steps, num_freq_bins) of PSD in dB.
        freqs_mhz: Frequency axis in MHz.
        times: Time axis in seconds.
    """
    fig = go.Figure(data=go.Heatmap(
        z=psd_matrix,
        x=freqs_mhz,
        y=times,
        colorscale="Viridis",
        colorbar={"title": "dB"},
    ))
    fig.update_layout(
        title="Waterfall (PSD over time)",
        xaxis_title="Frequency (MHz)",
        yaxis_title="Time (s)",
        height=450,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def band_activity_heatmap(
    activity_matrix: np.ndarray,
    num_bands: int,
    times: np.ndarray,
    title: str = "Band activity over time",
) -> go.Figure:
    """Heatmap of per-band activity (detected or scheduler priority) over time.

    Args:
        activity_matrix: Shape (num_time_steps, num_bands).
        num_bands: Number of bands.
        times: Time axis.
    """
    fig = go.Figure(data=go.Heatmap(
        z=activity_matrix.T,
        x=times,
        y=list(range(num_bands)),
        colorscale="Hot",
        colorbar={"title": "value"},
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Band ID",
        height=450,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig
