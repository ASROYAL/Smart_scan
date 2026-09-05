"""Streamlit dashboard for the Smart Adaptive Spectrum Scan system.

Displays ACTUAL experiment data — every chart comes from a real run, never
fabricated values. Run with:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable when run via streamlit
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from smartscan.acquisition.simulator_source import SimulatedRFSource
from smartscan.core.config import load_config
from smartscan.core.models import SchedulerType
from smartscan.dsp.detector import EnergyDetector
from smartscan.evaluation.experiment import build_and_run
from smartscan.receiver.receiver import Receiver
from smartscan.simulation.environment import RFEnvironment
from smartscan.simulation.scenarios import ALL_SCENARIOS, get_scenario
from smartscan.visualization import metrics_viz, spectrum, timeline

st.set_page_config(page_title="Smart Spectrum Scan", layout="wide")

CONFIG_PATH = ROOT / "config" / "default.yaml"


@st.cache_data(show_spinner=False)
def run_experiment_cached(scheduler: str, scenario: str, num_bands: int,
                          num_steps: int, seed: int):
    config = load_config(str(CONFIG_PATH))
    config.environment.num_bands = num_bands
    config.simulation.num_steps = num_steps
    config.simulation.seed = seed
    outcome = build_and_run(config, scheduler, scenario, seed=seed)
    return outcome, config


@st.cache_data(show_spinner=False)
def sample_spectrum(scenario: str, num_bands: int, seed: int, center_mhz: float):
    """Grab one observation window for the live-spectrum view."""
    config = load_config(str(CONFIG_PATH))
    config.environment.num_bands = num_bands
    scen = get_scenario(scenario, seed=seed, num_bands=num_bands,
                        total_bw=config.environment.total_bandwidth,
                        center=config.environment.center_frequency)
    env = RFEnvironment(scen.emitters, noise_power_dbm=scen.noise_power_dbm,
                        sample_rate=config.environment.sample_rate, seed=seed)
    src = SimulatedRFSource(env)
    rx = Receiver(src, config.receiver)
    samples, meta = rx.observe(center_mhz * 1e6)
    return samples, meta, config.detector


# ------------------------------------------------------------------ Sidebar
st.sidebar.title("Smart Spectrum Scan")
st.sidebar.caption("Adaptive RF monitoring — all data from live experiments")

scheduler = st.sidebar.selectbox(
    "Scheduler", [s.value for s in SchedulerType], index=5,
)
scenario = st.sidebar.selectbox("Scenario", ALL_SCENARIOS, index=2)
num_bands = st.sidebar.slider("Number of bands", 10, 100, 50, step=10)
num_steps = st.sidebar.slider("Scan steps", 100, 3000, 800, step=100)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

run_clicked = st.sidebar.button("Run experiment", type="primary")

st.title("Smart Adaptive Spectrum Scan & Signal Monitoring")

if run_clicked or "outcome" not in st.session_state:
    with st.spinner("Running experiment..."):
        outcome, config = run_experiment_cached(
            scheduler, scenario, num_bands, num_steps, int(seed),
        )
        st.session_state["outcome"] = outcome
        st.session_state["config"] = config
        st.session_state["params"] = (scheduler, scenario, num_bands, num_steps, int(seed))

outcome = st.session_state["outcome"]
config = st.session_state["config"]
result = outcome.result

tabs = st.tabs([
    "Overview", "Live Spectrum", "Receiver & Scheduler",
    "Band History", "Performance", "Scheduler Comparison",
])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    st.subheader(f"{result.scheduler_name} on '{result.scenario_name}'")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prob. of Detection", f"{result.probability_of_detection:.2f}")
    c2.metric("Prob. False Alarm", f"{result.probability_of_false_alarm:.3f}")
    c3.metric("Discovery Ratio", f"{result.activity_discovery_ratio:.2f}")
    c4.metric("Scan Efficiency", f"{result.scan_efficiency:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Discovery Delay", f"{result.avg_discovery_delay*1000:.0f} ms")
    c6.metric("Band Coverage", f"{result.band_coverage:.2f}")
    c7.metric("Starvation Rate", f"{result.starvation_rate:.3f}")
    c8.metric("Avg Reward", f"{result.avg_reward:.3f}")

    st.plotly_chart(timeline.band_selection_timeline(outcome.artifacts),
                    use_container_width=True)
    st.plotly_chart(timeline.cumulative_detections(outcome.artifacts),
                    use_container_width=True)

# ------------------------------------------------------------------ Live Spectrum
with tabs[1]:
    st.subheader("Live spectrum snapshot")
    center_default = config.environment.center_frequency / 1e6
    total_bw_mhz = config.environment.total_bandwidth / 1e6
    center_mhz = st.slider(
        "Observation center (MHz)",
        center_default - total_bw_mhz / 2 + 10,
        center_default + total_bw_mhz / 2 - 10,
        center_default, step=10.0,
    )
    samples, meta, det_cfg = sample_spectrum(scenario, num_bands, int(seed), center_mhz)
    st.plotly_chart(spectrum.spectrum_figure(samples, meta, det_cfg),
                    use_container_width=True)
    st.caption("PSD of the receiver's 20 MHz window. Red dashed = detection threshold, "
               "gray dotted = estimated noise floor. The detector sees only this — no ground truth.")

# ------------------------------------------------------------------ Receiver & Scheduler
with tabs[2]:
    st.subheader("Receiver state")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total scans", result.num_steps)
    c2.metric("Simulated duration", f"{result.duration:.2f} s")
    c3.metric("Instantaneous BW", f"{config.receiver.instantaneous_bandwidth/1e6:.0f} MHz")

    st.subheader("Scheduler behavior")
    st.plotly_chart(timeline.reward_over_time(outcome.artifacts),
                    use_container_width=True)

    # Most-visited bands
    visits = outcome.artifacts.band_visit_counts
    visit_df = pd.DataFrame(
        [{"band_id": b, "visits": v} for b, v in sorted(visits.items())]
    )
    st.bar_chart(visit_df.set_index("band_id"))
    st.caption("Visits per band — adaptive schedulers concentrate on active bands.")

# ------------------------------------------------------------------ Band History
with tabs[3]:
    st.subheader("Band history table")
    states = outcome.state_manager.all_states()
    rows = []
    for s in states:
        rows.append({
            "Band": s.band_id,
            "Freq (MHz)": f"{s.center_frequency/1e6:.0f}",
            "Hits": s.hit_count,
            "Misses": s.miss_count,
            "Activity Prob": round(s.rolling_activity_prob, 3),
            "Last Seen (s)": round(s.last_scan_time, 3) if s.last_scan_time >= 0 else "—",
            "Last Detect (s)": round(s.last_detection_time, 3) if s.last_detection_time >= 0 else "—",
            "Est Period (s)": round(s.estimated_period, 3) if s.estimated_period else "—",
            "Avg SNR (dB)": round(s.avg_snr_db, 1),
            "Confidence": round(s.confidence, 2),
        })
    df = pd.DataFrame(rows)
    # Sort by hits so active bands surface
    df = df.sort_values("Hits", ascending=False)
    st.dataframe(df, use_container_width=True, height=500)

# ------------------------------------------------------------------ Performance
with tabs[4]:
    st.subheader("Discovery delay distribution")
    # Recompute delays for the histogram
    from smartscan.evaluation import metrics as metrics_mod
    from smartscan.evaluation.evaluator import Evaluator
    ev = Evaluator(outcome.environment)
    ev.set_state_manager(outcome.state_manager)
    gt_events = outcome.environment.get_all_ground_truth(0.0, result.duration)
    det_map = ev._match_detections_to_events(outcome.artifacts.records, gt_events)
    delays = metrics_mod.discovery_delays([e.time_start for e in gt_events], det_map)
    st.plotly_chart(metrics_viz.discovery_delay_distribution(delays),
                    use_container_width=True)

    st.subheader("Pipeline timing")
    timing = outcome.artifacts.timing
    timing_rows = [
        {"Stage": stage, "Calls": s.count, "Avg (ms)": round(s.avg_ms, 3),
         "Max (ms)": round(s.max_ms, 3)}
        for stage, s in sorted(timing.all_stats().items())
    ]
    st.dataframe(pd.DataFrame(timing_rows), use_container_width=True)

# ------------------------------------------------------------------ Scheduler Comparison
with tabs[5]:
    st.subheader("Compare all schedulers on this scenario")
    st.caption("Runs every scheduler on the identical environment. This may take a moment.")
    if st.button("Run comparison"):
        with st.spinner("Benchmarking all schedulers..."):
            comp_rows = []
            for sched in [s.value for s in SchedulerType]:
                out, _ = run_experiment_cached(sched, scenario, num_bands,
                                                num_steps, int(seed))
                r = out.result
                comp_rows.append({
                    "scheduler": r.scheduler_name,
                    "scenario": scenario,
                    "PD": r.probability_of_detection,
                    "discovery_ratio": r.activity_discovery_ratio,
                    "avg_delay_ms": r.avg_discovery_delay * 1000,
                    "scan_efficiency": r.scan_efficiency,
                    "coverage": r.band_coverage,
                    "avg_reward": r.avg_reward,
                })
            comp_df = pd.DataFrame(comp_rows)
            st.session_state["comp_df"] = comp_df

    if "comp_df" in st.session_state:
        comp_df = st.session_state["comp_df"]
        st.dataframe(comp_df, use_container_width=True)
        metric = st.selectbox(
            "Metric to chart",
            ["scan_efficiency", "discovery_ratio", "avg_delay_ms", "avg_reward",
             "PD", "coverage"],
        )
        import plotly.express as px
        fig = px.bar(comp_df, x="scheduler", y=metric, color="scheduler",
                     title=f"{metric} by scheduler — {scenario}")
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(
    "Ground truth is used ONLY by the evaluator. The scheduler and detector "
    "never see it — they infer activity from IQ samples alone."
)
