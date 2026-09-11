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
sys.path.insert(0, str(ROOT / "dashboard"))

import pandas as pd
import streamlit as st
from offline_help import render_help
from phone_link import render_phone_link

from smartscan.acquisition.simulator_source import SimulatedRFSource
from smartscan.core.config import load_config
from smartscan.core.models import SchedulerType
from smartscan.evaluation.experiment import build_and_run
from smartscan.receiver.receiver import Receiver
from smartscan.simulation.environment import RFEnvironment
from smartscan.simulation.scenarios import ALL_SCENARIOS, get_scenario
from smartscan.visualization import metrics_viz, spectrum, timeline

st.set_page_config(page_title="Cognitive EW Receiver Scheduler", layout="wide",
                   page_icon="📡")

CONFIG_PATH = ROOT / "config" / "default.yaml"

workspace_view = st.sidebar.radio("Workspace", ["Dashboard", "Phone Link", "Offline help"])
if workspace_view == "Offline help":
    render_help(ROOT / "docs" / "offline-help.md")
    st.stop()

# ---- unified operations-console palette -----------------------------------
st.sidebar.divider()
P = {
    "bg": "#020604", "bg2": "#020604", "surf": "#050b08", "elev": "#08120d",
    "bd": "#24563a", "tx": "#c9e7d4", "mut": "#67a77f", "acc": "#35ff9a",
    "acc2": "#69b7ff", "good": "#35ff9a", "bad": "#ff3030", "sh": "none",
    "orb1": "transparent", "orb2": "transparent", "plot_tmpl": "plotly_dark",
    "plot_bg": "#030806", "grid": "#173323",
}
bg, surf, bd, tx, mut, acc, acc2, good, bad = (
    P[k] for k in ("bg", "surf", "bd", "tx", "mut", "acc", "acc2", "good", "bad")
)

if workspace_view == "Phone Link":
    render_phone_link(P)
    st.stop()

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
      html, body, .stApp, [class*="css"], [data-testid="stMarkdownContainer"] {{
        font-family:'JetBrains Mono',ui-monospace,monospace; }}
      #MainMenu, footer, [data-testid="stHeader"] {{visibility:hidden; background:transparent !important; height:0;}}
      .stApp {{background:{bg} !important;}}
      section[data-testid="stSidebar"] > div {{background:{surf} !important;
        border-right:1px solid {bd};}}
      .stApp, .stApp p, .stApp li, [data-testid="stMarkdownContainer"] {{color:{tx};}}
      h1,h2,h3,h4 {{color:{tx} !important; letter-spacing:.035em; text-transform:uppercase;}}
      [data-testid="stWidgetLabel"] p {{color:{mut} !important; font-weight:600; font-size:0.75rem;
        letter-spacing:.05em; text-transform:uppercase;}}
      button[data-baseweb="tab"] {{color:{mut} !important; font-weight:600;}}
      button[data-baseweb="tab"][aria-selected="true"] {{color:{tx} !important;}}
      [data-testid="stPlotlyChart"] {{border:1px solid {bd}; border-radius:0; background:{surf}; padding:8px;}}
      div[data-testid="stMetric"] {{background:{surf}; border:1px solid {bd}; border-radius:0;
        padding:12px 14px;}}
      [data-testid="stMetricValue"] {{color:{tx} !important;}}
      [data-testid="stMetricLabel"] p {{color:{mut} !important;}}
      .stButton>button {{background:#102d1e; color:#d9ffe8; font-weight:700;
        border:1px solid {acc}; border-radius:0; padding:8px 16px;}}
      .stButton>button:hover {{background:#173d29; color:white;}}
      div[data-testid="stDataFrame"] {{border:1px solid {bd}; border-radius:0; overflow:hidden;}}
      /* ---- hero ---- */
      .hero {{border-left:7px solid {acc}; border-top:1px solid {bd}; border-bottom:1px solid {bd};
        background:{surf}; padding:9px 14px; margin-bottom:8px;}}
      .eyebrow {{font-family:'JetBrains Mono',monospace; font-size:0.7rem; font-weight:700; letter-spacing:0.16em;
        color:{mut}; text-transform:uppercase;}}
      .htitle {{font-size:1.45rem; font-weight:700; color:{tx}; margin-top:4px; line-height:1.12;
        letter-spacing:.06em;}}
      .hsub {{color:{mut}; font-size:0.78rem; margin-top:4px; max-width:960px;}}
      /* ---- status pill ---- */
      .status {{font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:{mut}; background:{surf};
        border:1px solid {bd}; border-radius:0; padding:10px 16px; margin:14px 0 6px;}}
      .status b {{color:{acc}; font-weight:700;}}
      /* ---- metric tiles ---- */
      .tgrid {{display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:6px 0 2px;}}
      .tile {{background:{surf}; border:1px solid {bd}; border-radius:0; padding:13px 15px;
        position:relative; overflow:hidden;}}
      .tile::after {{content:''; position:absolute; inset:0 0 auto 0; height:2px;
        background:{acc}; opacity:.65;}}
      .tile .lab {{font-size:0.72rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; color:{mut};}}
      .tile .val {{font-size:1.85rem; font-weight:800; color:{tx}; margin-top:8px; letter-spacing:-0.02em;
        font-variant-numeric:tabular-nums;}}
      .chip {{display:inline-block; margin-top:9px; font-family:'JetBrains Mono',monospace; font-size:0.7rem;
        font-weight:700; padding:2px 9px; border-radius:0; border:1px solid currentColor;}}
      .chip.up {{color:{good}; background:{good}1f;}} .chip.down {{color:{bad}; background:{bad}1f;}}
    </style>
    <div class="hero">
      <div class="eyebrow">SMARTSCAN // SIMULATED RF SURVEILLANCE // RESEARCH TRAINING ENVIRONMENT</div>
      <div class="htitle">SPECTRUM OPERATIONS CONSOLE</div>
      <div class="hsub">NARROWBAND RECEIVER SCHEDULING // ONLINE HIT-MISS LEARNING // CONTROLLED GROUND-TRUTH EVALUATION</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _style(fig):
    """Apply the current (light/dark) theme consistently to every chart."""
    fig.update_layout(
        template=P["plot_tmpl"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["plot_bg"],
        font={"color": P["tx"], "size": 12, "family": "JetBrains Mono"},
        colorway=[P["acc"], P["acc2"], P["good"], P["bad"], "#8163D6"],
        title_font={"color": P["tx"], "size": 15, "family": "JetBrains Mono"},
        margin={"l": 48, "r": 24, "t": 46, "b": 42},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(gridcolor=P["grid"], zerolinecolor=P["grid"])
    fig.update_yaxes(gridcolor=P["grid"], zerolinecolor=P["grid"])
    return fig


def tile(label, value, chip=None, up=True, i=0, tone=None):
    """One operational metric tile (HTML)."""
    chip_html = f"<span class='chip {'up' if up else 'down'}'>{chip}</span>" if chip else ""
    vstyle = f"color:{tone}" if tone else ""
    return (f"<div class='tile' data-order='{i}'>"
            f"<div class='lab'>{label}</div>"
            f"<div class='val' style='{vstyle}'>{value}</div>{chip_html}</div>")


def tile_grid(tiles):
    st.markdown("<div class='tgrid'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)


def _confusion(records):
    """Count true-positive / false-alarm / miss / true-negative from a run.

    This is the exact rule the evaluator uses to decide whether a scan was a
    genuine HIT: the detector's `detected` flag is compared with `truly_active`,
    which the evaluator reads from the hidden ground truth (never shown to the
    detector or scheduler).
    """
    tp = sum(1 for r in records if r.detected and r.truly_active)
    fp = sum(1 for r in records if r.detected and not r.truly_active)
    fn = sum(1 for r in records if (not r.detected) and r.truly_active)
    tn = sum(1 for r in records if (not r.detected) and not r.truly_active)
    return tp, fp, fn, tn


@st.cache_data(show_spinner=False)
def run_experiment_cached(scheduler: str, scenario: str, num_bands: int,
                          num_steps: int, seed: int, detector: str = "energy"):
    config = load_config(str(CONFIG_PATH))
    config.environment.num_bands = num_bands
    config.simulation.num_steps = num_steps
    config.simulation.seed = seed
    config.detector.detector_type = detector
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
st.sidebar.title("Mission setup")
st.sidebar.caption("All figures below come from a live experiment — no fabricated values.")

scheduler = st.sidebar.selectbox(
    "Scheduler (the brain)", [s.value for s in SchedulerType],
    index=[s.value for s in SchedulerType].index("adaptive"),
    help="round_robin/random = open-loop baselines; adaptive/bandits/q_learning = learning strategies",
)
detector = st.sidebar.selectbox(
    "Detector (the sensor)", ["energy", "matched_filter", "cyclostationary"], index=0,
    help="energy = PSD threshold; matched_filter = coherent peak-bin; cyclostationary = autocorrelation feature",
)
scenario = st.sidebar.selectbox("Scenario (the environment)", ALL_SCENARIOS,
                                index=ALL_SCENARIOS.index("radar_scan") if "radar_scan" in ALL_SCENARIOS else 2)
num_bands = st.sidebar.slider("Number of bands", 10, 100, 50, step=10)
num_steps = st.sidebar.slider("Scan steps (mission length)", 100, 3000, 800, step=100)
seed = st.sidebar.number_input("Random seed (reproducibility)", value=42, step=1)

run_clicked = st.sidebar.button("▶ Run experiment", type="primary", use_container_width=True)

if run_clicked or "outcome" not in st.session_state:
    with st.spinner("Running experiment..."):
        outcome, config = run_experiment_cached(
            scheduler, scenario, num_bands, num_steps, int(seed), detector,
        )
        st.session_state["outcome"] = outcome
        st.session_state["config"] = config
        st.session_state["params"] = (scheduler, scenario, num_bands, num_steps, int(seed))

outcome = st.session_state["outcome"]
config = st.session_state["config"]
result = outcome.result

tabs = st.tabs([
    "Mission Overview", "Live Spectrum", "Receiver & Scheduler",
    "Emitter Memory", "Performance", "Scheduler Comparison",
])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    # instrument status strip
    st.markdown(
        f"<div class='status'>DETECTOR <b>{detector}</b> &nbsp;|&nbsp; SCHEDULER <b>{result.scheduler_name}</b> "
        f"&nbsp;|&nbsp; SCENARIO <b>{result.scenario_name}</b> &nbsp;|&nbsp; BANDS <b>{num_bands}</b> "
        f"&nbsp;|&nbsp; SCANS <b>{result.num_steps}</b> &nbsp;|&nbsp; SEED <b>{int(seed)}</b> "
        f"&nbsp;|&nbsp; SIM-TIME <b>{result.duration:.1f}s</b> &nbsp;|&nbsp; RUNTIME <b>{result.wall_clock_seconds:.1f}s</b></div>",
        unsafe_allow_html=True,
    )

    # baseline (round-robin) on the identical environment, for "vs baseline" deltas
    base = None
    if result.scheduler_name != "round_robin":
        base_out, _ = run_experiment_cached("round_robin", scenario, num_bands,
                                            num_steps, int(seed), detector)
        base = base_out.result

    def _d(cur, ref, pct=False, invert=False):
        if base is None or ref is None:
            return None
        diff = cur - ref
        if invert:  # for delay, lower is better
            diff = -diff
        return (f"{diff*100:+.0f}% vs RR" if pct else f"{diff:+.2f} vs RR")

    _b = base
    tile_grid([
        tile("Prob. of Detection", f"{result.probability_of_detection:.2f}", i=0),
        tile("Prob. False Alarm", f"{result.probability_of_false_alarm:.3f}", i=1),
        tile("Intercept Ratio", f"{result.activity_discovery_ratio:.2f}",
             chip=_d(result.activity_discovery_ratio, _b.activity_discovery_ratio if _b else None),
             up=(_b is None or result.activity_discovery_ratio >= _b.activity_discovery_ratio), i=2),
        tile("Scan Efficiency", f"{result.scan_efficiency:.3f}",
             chip=_d(result.scan_efficiency, _b.scan_efficiency if _b else None),
             up=(_b is None or result.scan_efficiency >= _b.scan_efficiency), i=3),
        tile("Avg Intercept Time", f"{result.avg_discovery_delay*1000:.0f} ms",
             chip=(f"{(result.avg_discovery_delay-_b.avg_discovery_delay)*1000:+.0f} ms vs RR" if _b else None),
             up=(_b is None or result.avg_discovery_delay <= _b.avg_discovery_delay), i=4),
        tile("Band Coverage", f"{result.band_coverage:.2f}", i=5),
        tile("Prediction Accuracy", f"{result.prediction_accuracy:.2f}", i=6),
        tile("Avg Reward", f"{result.avg_reward:.3f}",
             chip=_d(result.avg_reward, _b.avg_reward if _b else None),
             up=(_b is None or result.avg_reward >= _b.avg_reward), i=7),
        tile("Missed Event Rate", f"{result.missed_event_rate:.2f}",
             up=False, tone=P["bad"], i=8),
        tile("Censored Intercept", f"{result.censored_avg_intercept_time*1000:.0f} ms",
             up=False, i=9),
    ])

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.plotly_chart(_style(timeline.band_selection_timeline(outcome.artifacts)),
                    use_container_width=True)
    st.plotly_chart(_style(timeline.cumulative_detections(outcome.artifacts)),
                    use_container_width=True)

    # -------- Detection Audit: how a "hit" is verified --------
    st.divider()
    st.subheader("🔍 Detection audit — how a hit is verified")
    tp, fp, fn, tn = _confusion(outcome.artifacts.records)
    tile_grid([
        tile("True hits · TP", tp, tone=P["good"], i=0),
        tile("False alarms · FP", fp, tone=P["bad"], i=1),
        tile("Misses · FN", fn, tone=P["bad"], i=2),
        tile("Correct rejects · TN", tn, tone=P["mut"], i=3),
    ])

    pd_ = tp / (tp + fn) if (tp + fn) else 0.0
    pfa_ = fp / (fp + tn) if (fp + tn) else 0.0
    st.markdown(
        f"""<div class='audit' style='color:{P['mut']};font-size:0.9rem;line-height:1.7;margin-top:10px'>
        Each scan's <b>detected</b> flag (from the DSP energy/feature test on the raw IQ — <i>no ground truth</i>)
        is compared by the <b>evaluator</b> against <b>truly&nbsp;active</b>, read from the hidden ground truth.
        A scan counts as a <b>true hit only when both are true</b>.<br>
        &nbsp;&nbsp;• Probability of Detection &nbsp;PD = TP / (TP+FN) = {tp}/{tp+fn} = <b>{pd_:.2f}</b><br>
        &nbsp;&nbsp;• Probability of False Alarm &nbsp;PFA = FP / (FP+TN) = {fp}/{fp+tn} = <b>{pfa_:.3f}</b><br>
        The scheduler is <b>rewarded on its own `detected` flag</b>, never on the truth — so it must genuinely
        infer activity. The truth is used <i>only here, for scoring</i>.</div>""",
        unsafe_allow_html=True,
    )

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
    st.plotly_chart(_style(spectrum.spectrum_figure(samples, meta, det_cfg)),
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
    st.plotly_chart(_style(timeline.reward_over_time(outcome.artifacts)),
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
    rows = [
        {
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
        }
        for s in states
    ]
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
    st.plotly_chart(_style(metrics_viz.discovery_delay_distribution(delays)),
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
                                                num_steps, int(seed), detector)
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
        st.plotly_chart(_style(fig), use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(
    "🔒 Information wall: ground truth is used ONLY by the evaluator. The scheduler "
    "and detector never see it — they infer activity from IQ samples alone "
    "(enforced by automated tests)."
)
st.sidebar.caption(
    "Research demonstration for the DRDO-sponsored Smart India Hackathon problem "
    "'Smart Scan Strategy for Electronic Warfare'. All data is simulated; this is "
    "not an official DRDO system and contains no classified information."
)
