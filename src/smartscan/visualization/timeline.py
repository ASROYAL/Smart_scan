"""Timeline visualizations — band selection, hit/miss, cumulative detections."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from smartscan.evaluation.evaluator import RunArtifacts


def band_selection_timeline(artifacts: RunArtifacts) -> go.Figure:
    """Scatter of which band was scanned at each step, colored by hit/miss."""
    records = artifacts.records
    steps = [r.scan_number for r in records]
    bands = [r.band_id for r in records]
    colors = ["#2ca02c" if r.detected else "#d62728" for r in records]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=bands, mode="markers",
        marker={"color": colors, "size": 5},
        name="scan",
        text=["hit" if r.detected else "miss" for r in records],
    ))
    fig.update_layout(
        title="Scheduler band-selection timeline (green=hit, red=miss)",
        xaxis_title="Scan step",
        yaxis_title="Band ID",
        height=400,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def cumulative_detections(artifacts: RunArtifacts) -> go.Figure:
    """Cumulative count of detections over scan steps."""
    records = artifacts.records
    steps = [r.scan_number for r in records]
    cumulative = np.cumsum([1 if r.detected else 0 for r in records])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=cumulative, mode="lines",
                              line={"color": "#1f77b4"}))
    fig.update_layout(
        title="Cumulative detections",
        xaxis_title="Scan step",
        yaxis_title="Cumulative detections",
        height=350,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def reward_over_time(artifacts: RunArtifacts) -> go.Figure:
    """Reward per step and a running average."""
    records = artifacts.records
    steps = [r.scan_number for r in records]
    rewards = [r.reward for r in records]
    running_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=rewards, mode="markers",
                              marker={"size": 3, "color": "lightgray"}, name="reward"))
    fig.add_trace(go.Scatter(x=steps, y=running_avg, mode="lines",
                              line={"color": "#ff7f0e", "width": 2}, name="running avg"))
    fig.update_layout(
        title="Reward over time",
        xaxis_title="Scan step",
        yaxis_title="Reward",
        height=350,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def true_vs_detected_timeline(
    artifacts: RunArtifacts, band_id: int,
) -> go.Figure:
    """For one band, plot true activity vs detected activity over the scans of it."""
    records = [r for r in artifacts.records if r.band_id == band_id]
    times = [r.timestamp for r in records]
    truly = [1 if r.truly_active else 0 for r in records]
    detected = [1 if r.detected else 0 for r in records]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=truly, mode="lines+markers",
                              name="truly active (ground truth)",
                              line={"color": "#2ca02c"}))
    fig.add_trace(go.Scatter(x=times, y=detected, mode="markers",
                              name="detected",
                              marker={"color": "#d62728", "symbol": "x", "size": 8}))
    fig.update_layout(
        title=f"Band {band_id}: true vs detected activity",
        xaxis_title="Time (s)",
        yaxis_title="Active (1) / Inactive (0)",
        height=350,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig
