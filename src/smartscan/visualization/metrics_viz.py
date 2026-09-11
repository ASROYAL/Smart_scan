"""Metric comparison visualizations across schedulers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def scheduler_comparison_bar(
    summary_df: pd.DataFrame, metric: str, scenario: str | None = None,
) -> go.Figure:
    """Bar chart comparing a metric across schedulers (optionally one scenario)."""
    df = summary_df
    if scenario is not None:
        df = df[df["scenario"] == scenario]

    fig = px.bar(
        df, x="scheduler", y=metric, color="scheduler",
        facet_col="scenario" if scenario is None else None,
        title=f"{metric} by scheduler" + (f" — {scenario}" if scenario else ""),
    )
    fig.update_layout(height=400, margin={"l": 40, "r": 20, "t": 60, "b": 40})
    return fig


def discovery_delay_distribution(delays: list[float]) -> go.Figure:
    """Histogram of discovery delays."""
    fig = go.Figure()
    if delays:
        fig.add_trace(go.Histogram(x=np.array(delays) * 1000, nbinsx=30,
                                    marker_color="#1f77b4"))
    fig.update_layout(
        title="Discovery delay distribution",
        xaxis_title="Delay (ms)",
        yaxis_title="Count",
        height=350,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def priority_heatmap(
    priority_matrix: np.ndarray, num_bands: int, times: np.ndarray,
) -> go.Figure:
    """Heatmap of scheduler priority scores per band over time."""
    fig = go.Figure(data=go.Heatmap(
        z=priority_matrix.T,
        x=times,
        y=list(range(num_bands)),
        colorscale="Plasma",
        colorbar={"title": "priority"},
    ))
    fig.update_layout(
        title="Scheduler priority heatmap",
        xaxis_title="Time (s)",
        yaxis_title="Band ID",
        height=450,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
    )
    return fig


def metric_radar(summary_df: pd.DataFrame, scenario: str) -> go.Figure:
    """Radar chart comparing schedulers across normalized metrics for one scenario."""
    df = summary_df[summary_df["scenario"] == scenario].copy()

    # Metrics where higher is better
    metrics = ["PD", "discovery_ratio", "scan_efficiency", "coverage", "avg_reward"]
    fig = go.Figure()
    for _, row in df.iterrows():
        values = [row[m] for m in metrics]
        values.append(values[0])  # close the loop
        fig.add_trace(go.Scatterpolar(
            r=values, theta=[*metrics, metrics[0]],
            fill="toself", name=row["scheduler"],
        ))
    fig.update_layout(
        title=f"Scheduler comparison — {scenario}",
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        height=450,
    )
    return fig
