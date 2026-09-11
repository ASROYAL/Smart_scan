"""Operational phone-link console for consented Android and iPhone devices."""

from __future__ import annotations

import contextlib
import html
import json
import math
import platform
import random
import re
import subprocess
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from smartscan.core.config import load_config
from smartscan.core.models import DetectorType, SchedulerType
from smartscan.evaluation.experiment import build_and_run
from smartscan.simulation.scenarios import scenario_phone_training
from smartscan.telemetry.companion import get_companion_server
from smartscan.telemetry.phone import (
    AlertLevel,
    AlertMemory,
    LinkState,
    advance_alert,
    classify_link,
    quality_to_snr,
    rssi_to_quality,
)

ROOT = Path(__file__).resolve().parent.parent
PREFERENCES_PATH = ROOT / "data" / "phone_link_preferences.json"
UI_REFRESH_SECONDS = 0.1
WAR_UI_REFRESH_SECONDS = 0.5
INTAKE_INTERVALS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


def signal_quality(rssi_dbm: int | None) -> tuple[str, int]:
    """Compatibility wrapper retaining the dashboard's original label casing."""

    label, quality = rssi_to_quality(rssi_dbm)
    return label.title(), quality


@dataclass(frozen=True)
class BluetoothDevice:
    name: str
    address: str
    connected: bool
    rssi_dbm: int | None

    @property
    def label(self) -> str:
        suffix = self.address[-5:] if self.address else "LOCAL"
        state = "LIVE" if self.rssi_dbm is not None else ("ACTIVE" if self.connected else "PAIRED")
        return f"{self.name} · {suffix} · {state}"


@dataclass(frozen=True)
class ContactAssessment:
    """Observation-derived training hypothesis for the War Mode display."""

    object_type: str
    confidence: int
    track_state: str
    motion: str
    volatility: float
    mean_quality: float
    recommendation: str
    evidence: tuple[str, ...]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "attrib_yes", "connected"}


def _as_rssi(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"-?\d+", str(value))
    if not match:
        return None
    reading = int(match.group())
    return reading if -127 <= reading <= 20 else None


def parse_system_profiler(payload: dict[str, Any]) -> list[BluetoothDevice]:
    """Extract paired devices from different macOS system_profiler layouts."""

    found: list[BluetoothDevice] = []

    def visit(node: Any, parent_name: str | None = None) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item, parent_name)
            return
        if not isinstance(node, dict):
            return
        keys = {str(key).lower() for key in node}
        if keys & {
            "device_address",
            "device_rssi",
            "device_connected",
            "device_minorclassofdevice_string",
        }:
            name = str(
                node.get("device_name")
                or node.get("_name")
                or parent_name
                or "Bluetooth device"
            )
            address = str(node.get("device_address") or node.get("address") or "")
            rssi = _as_rssi(node.get("device_rssi"))
            found.append(
                BluetoothDevice(
                    name=name,
                    address=address,
                    # macOS often omits device_connected for phones; a current
                    # RSSI is direct evidence that the link is active.
                    connected=_as_bool(node.get("device_connected", False)) or rssi is not None,
                    rssi_dbm=rssi,
                )
            )
        for key, value in node.items():
            visit(value, str(key) if isinstance(value, dict) else parent_name)

    visit(payload)
    unique: dict[str, BluetoothDevice] = {}
    for device in found:
        identity = device.address.lower() or device.name.lower()
        previous = unique.get(identity)
        if previous is None or (device.connected, device.rssi_dbm is not None) > (
            previous.connected,
            previous.rssi_dbm is not None,
        ):
            unique[identity] = device
    return sorted(unique.values(), key=lambda item: (not item.connected, item.name.lower()))


def read_bluetooth_devices(timeout: float = 12.0) -> tuple[list[BluetoothDevice], str | None]:
    """Read Android/iPhone pairing and RSSI using local macOS telemetry."""

    if platform.system() != "Darwin":
        return [], "Bluetooth RSSI collection is available only on macOS."
    try:
        completed = subprocess.run(
            ["system_profiler", "SPBluetoothDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"macOS Bluetooth query failed: {exc}"
    if completed.returncode != 0:
        return [], completed.stderr.strip() or "system_profiler returned no data"
    try:
        return parse_system_profiler(json.loads(completed.stdout)), None
    except json.JSONDecodeError:
        return [], "macOS returned invalid Bluetooth data."


def _load_preferences() -> None:
    if st.session_state.get("phone_preferences_loaded"):
        return
    preferences: dict[str, Any] = {}
    with contextlib.suppress(FileNotFoundError, OSError, json.JSONDecodeError):
        preferences = json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))
    stored_source = preferences.get("source", "Phone browser link")
    st.session_state.setdefault("phone_source_saved", stored_source)
    st.session_state.setdefault("phone_source", stored_source)
    st.session_state.setdefault("phone_device_address", preferences.get("address", ""))
    st.session_state.setdefault("phone_device_name", preferences.get("name", "AUTO SELECT"))
    requested_interval = float(preferences.get("intake_interval", 0.1))
    selected_interval = min(INTAKE_INTERVALS, key=lambda value: abs(value - requested_interval))
    st.session_state.setdefault("phone_intake_interval_saved", selected_interval)
    st.session_state.setdefault("phone_intake_interval", selected_interval)
    st.session_state.setdefault("phone_monitoring", True)
    st.session_state.setdefault("phone_alert_memory", AlertMemory())
    st.session_state.setdefault("phone_signal_history", [])
    st.session_state["phone_preferences_loaded"] = True
    get_companion_server()


def _save_preferences() -> None:
    source = st.session_state.get(
        "phone_source", st.session_state.get("phone_source_saved", "Phone browser link")
    )
    intake_interval = st.session_state.get(
        "phone_intake_interval", st.session_state.get("phone_intake_interval_saved", 0.1)
    )
    st.session_state["phone_source_saved"] = source
    st.session_state["phone_intake_interval_saved"] = intake_interval
    payload = {
        "source": source,
        "address": st.session_state.get("phone_device_address", ""),
        "name": st.session_state.get("phone_device_name", "AUTO SELECT"),
        "intake_interval": intake_interval,
    }
    try:
        PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREFERENCES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _demo_rssi(sample_number: int) -> int:
    """Training trace that crosses NORMAL, CAUTION, and WAR MODE thresholds."""

    baseline = -55 + 13 * math.sin(sample_number / 18.0)
    return round(baseline + random.uniform(-1.5, 1.5))


def _find_selected(devices: list[BluetoothDevice]) -> BluetoothDevice | None:
    address = st.session_state.get("phone_device_address", "").lower()
    selected = next((item for item in devices if item.address.lower() == address), None)
    if selected is None and devices:
        selected = devices[0]
        st.session_state["phone_device_address"] = selected.address
        st.session_state["phone_device_name"] = selected.name
        _save_preferences()
    return selected


def _empty_snapshot() -> dict[str, Any]:
    return {
        "name": st.session_state.get("phone_device_name", "NO DEVICE"),
        "rssi_dbm": None,
        "quality": 0,
        "quality_label": "NO DATA",
        "link_state": LinkState.OFFLINE,
        "alert": AlertLevel.NORMAL,
        "sample_age": None,
        "source": "PAUSED",
        "error": None,
    }


def _poll_phone() -> dict[str, Any]:
    """Poll at hardware speed while allowing the HUD to refresh at 10 Hz."""

    _load_preferences()
    existing = st.session_state.get("phone_snapshot")
    if not st.session_state.get("phone_monitoring", True):
        snapshot = dict(existing or _empty_snapshot())
        snapshot["alert"] = AlertLevel.NORMAL
        snapshot["quality_label"] = "PAUSED"
        get_companion_server().set_alert(AlertLevel.NORMAL.value, int(snapshot["quality"]))
        return snapshot

    now = time.monotonic()
    source = st.session_state.get(
        "phone_source", st.session_state.get("phone_source_saved", "Phone browser link")
    )
    intake_interval = float(
        st.session_state.get(
            "phone_intake_interval", st.session_state.get("phone_intake_interval_saved", 0.1)
        )
    )
    if st.session_state.get("phone_alert_source") != source:
        st.session_state["phone_alert_memory"] = AlertMemory()
        st.session_state["phone_alert_source"] = source
    new_sample = False
    error: str | None = None
    selected: BluetoothDevice | None = None
    quality_override: int | None = None
    sample_age_override: float | None = None
    link_state_override: LinkState | None = None

    if source == "Demo signal":
        last_poll = float(st.session_state.get("phone_last_demo_poll", -1e9))
        if now - last_poll < intake_interval and existing is not None:
            return existing
        history = st.session_state["phone_signal_history"]
        selected = BluetoothDevice("TRAINING PHONE", "DEMO", True, _demo_rssi(len(history)))
        st.session_state["phone_last_rssi_at"] = now
        st.session_state["phone_last_demo_poll"] = now
        new_sample = True
    elif source == "Phone browser link":
        server = get_companion_server()
        server.set_interval_ms(round(intake_interval * 1000))
        status = server.registry.latest()
        error = server.error
        selected = BluetoothDevice("IPHONE / ANDROID WEB LINK", status.client_id, status.connected, None)
        quality_override = status.quality
        sample_age_override = status.age_seconds
        link_state_override = LinkState.LIVE if status.connected else LinkState.OFFLINE
        previous_sequence = int(st.session_state.get("phone_companion_sequence", -1))
        new_sample = status.connected and status.sequence != previous_sequence
        st.session_state["phone_companion_sequence"] = status.sequence
    else:
        last_poll = float(st.session_state.get("phone_last_hardware_poll", -1e9))
        if now - last_poll < intake_interval and existing is not None:
            return existing
        devices, error = read_bluetooth_devices()
        st.session_state["phone_last_hardware_poll"] = now
        selected = _find_selected(devices)
        if selected is not None and selected.rssi_dbm is not None:
            st.session_state["phone_last_rssi_at"] = now
            new_sample = True

    rssi = selected.rssi_dbm if selected else None
    last_rssi_at = st.session_state.get("phone_last_rssi_at")
    sample_age = (
        sample_age_override
        if source == "Phone browser link"
        else (now - float(last_rssi_at) if last_rssi_at is not None else None)
    )
    link_state = link_state_override or classify_link(
        paired=selected is not None,
        explicitly_connected=bool(selected and selected.connected),
        rssi_dbm=rssi,
        sample_age=sample_age,
    )
    quality_label, quality = rssi_to_quality(rssi)
    if quality_override is not None:
        quality_label, quality = "TRANSPORT LINK", quality_override
    alert_memory: AlertMemory = st.session_state["phone_alert_memory"]
    if source == "Phone browser link" and link_state == LinkState.OFFLINE:
        alert_memory = AlertMemory()
        st.session_state["phone_alert_memory"] = alert_memory
    if new_sample:
        alert_memory = advance_alert(alert_memory, quality)
        st.session_state["phone_alert_memory"] = alert_memory
        history = st.session_state["phone_signal_history"]
        history.append(
            {
                "time": datetime.now().astimezone(),
                "rssi_dbm": rssi,
                "quality": quality,
                "device": selected.name if selected else "NO DEVICE",
                "source": (
                    "DEMO"
                    if source == "Demo signal"
                    else ("PHONE WEB" if source == "Phone browser link" else "macOS Bluetooth")
                ),
            }
        )
        del history[:-1200]

    snapshot = {
        "name": selected.name if selected else st.session_state.get("phone_device_name", "NO DEVICE"),
        "rssi_dbm": rssi,
        "quality": quality,
        "quality_label": quality_label,
        "link_state": link_state,
        "alert": alert_memory.level,
        "sample_age": sample_age,
        "source": (
            "DEMO"
            if source == "Demo signal"
            else ("PHONE WEB" if source == "Phone browser link" else "MAC BT")
        ),
        "error": error,
    }
    st.session_state["phone_snapshot"] = snapshot
    get_companion_server().set_alert(alert_memory.level.value, quality)
    return snapshot


def _trend(history: list[dict[str, Any]]) -> str:
    recent = history[-12:]
    if len(recent) < 3:
        return "ACQUIRING"
    values = [
        int(row.get("quality", rssi_to_quality(row.get("rssi_dbm"))[1])) for row in recent
    ]
    delta = values[-1] - values[0]
    if delta >= 3:
        return "APPROACHING"
    if delta <= -3:
        return "RECEDING"
    return "STABLE"


def assess_training_contact(
    snapshot: dict[str, Any], history: list[dict[str, Any]]
) -> ContactAssessment:
    """Fuse recent link observations into an explicitly simulated contact assessment.

    The demo maps device categories to airborne contacts. This is scenario
    configuration, not RF object recognition.
    """

    recent = history[-30:]
    values = [float(row.get("quality", 0)) for row in recent]
    mean_quality = sum(values) / len(values) if values else float(snapshot.get("quality", 0))
    volatility = float(pd.Series(values).std()) if len(values) > 1 else 0.0
    if math.isnan(volatility):
        volatility = 0.0
    name = str(snapshot.get("name", "UNKNOWN CONTACT"))
    identity = name.lower()
    if any(token in identity for token in ("airpod", "earbud", "buds", "headphone")):
        object_type = "DRONE FORMATION"
    elif any(token in identity for token in ("iphone", "android", "phone", "web link")):
        object_type = "FIGHTER AIRCRAFT"
    elif any(token in identity for token in ("watch", "tablet", "ipad", "laptop", "macbook")):
        object_type = "SURVEILLANCE AIRCRAFT"
    else:
        object_type = "UNIDENTIFIED AIRBORNE CONTACT"
    motion = _trend(history)
    freshness = snapshot.get("sample_age")
    freshness_score = 18 if freshness is not None and float(freshness) <= 2.0 else 7
    sample_score = min(24, len(values) * 2)
    stability_score = max(0, round(18 - min(volatility, 18)))
    strength_score = round(min(35, max(0, mean_quality - 60) * 0.875))
    confidence = min(97, max(32, freshness_score + sample_score + stability_score + strength_score))
    if motion == "APPROACHING":
        track_state = "CLOSING / PRIORITY TRACK"
        recommendation = "Maintain continuous track and execute defensive intercept simulation."
    elif motion == "RECEDING":
        track_state = "OPENING / MONITOR"
        recommendation = "Preserve track continuity and verify the contact is leaving the sector."
    else:
        track_state = "HOLDING / HIGH PROXIMITY"
        recommendation = "Maintain custody and run identification checks against prior observations."
    evidence = (
        f"Scenario identity rule: {name} -> {object_type}",
        f"{len(values)} recent observations; mean link quality {mean_quality:.1f}%",
        f"Observed motion {motion.lower()}; signal volatility {volatility:.1f} points",
        "Classification is a local training hypothesis; no ground-truth emitter label was used.",
    )
    return ContactAssessment(
        object_type=object_type,
        confidence=confidence,
        track_state=track_state,
        motion=motion,
        volatility=volatility,
        mean_quality=mean_quality,
        recommendation=recommendation,
        evidence=evidence,
    )


def _alert_css(alert: AlertLevel) -> str:
    color = {
        AlertLevel.NORMAL: "#2cff8f",
        AlertLevel.CAUTION: "#ffd400",
        AlertLevel.CRITICAL: "#ff2020",
    }[alert]
    animation = "edgePulse .55s steps(2,end) infinite" if alert != AlertLevel.NORMAL else "none"
    return textwrap.dedent(f"""
    <style>
      .phone-edge {{position:fixed;inset:0;pointer-events:none;z-index:9998;
        box-shadow:inset 0 0 0 3px {color}, inset 0 0 42px {color}66;
        opacity:{'0.95' if alert != AlertLevel.NORMAL else '0'};animation:{animation};}}
      @keyframes edgePulse {{0%,100%{{opacity:.25}}50%{{opacity:1}}}}
    </style>
    """)


def render_alarm_controller() -> None:
    """Render a persistent, user-armed Web Audio alarm controller.

    Browsers require one user gesture before sound is permitted. After ARM is
    pressed once, the component automatically sounds on future alert changes.
    """

    status_url = get_companion_server().status_url
    components.html(
        f"""
        <style>
        body{{margin:0;background:#020604;color:#9cc9ab;font-family:monospace}}
        .alarm{{display:flex;align-items:center;gap:10px;border:1px solid #24563a;
          padding:6px 9px;background:#030806}}
        button{{background:#102d1e;color:#d9ffe8;border:1px solid #35ff9a;padding:6px 12px;
          font-family:monospace;font-weight:bold}}#alarmState{{font-size:11px;letter-spacing:.08em}}
        </style><div class="alarm"><b>AUDIO WARNING BUS</b><button id="arm">ARM AUDIO</button>
        <button id="mute">SILENCE</button><span id="alarmState">DISARMED</span></div>
        <script>
        const statusUrl={json.dumps(status_url)}; let ctx=null,armed=false,muted=false;
        let last='NORMAL',lastTone=0;
        function tone(freq,duration,delay=0){{if(!armed||muted||!ctx)return;
          setTimeout(()=>{{const o=ctx.createOscillator(),g=ctx.createGain();o.frequency.value=freq;
          g.gain.setValueAtTime(.0001,ctx.currentTime);g.gain.exponentialRampToValueAtTime(.22,ctx.currentTime+.015);
          g.gain.exponentialRampToValueAtTime(.0001,ctx.currentTime+duration);o.connect(g);g.connect(ctx.destination);
          o.start();o.stop(ctx.currentTime+duration+.02)}},delay)}}
        function warn(level){{if(level==='WAR MODE'){{tone(920,.16);tone(690,.16,210);tone(920,.2,420)}}
          else if(level==='CAUTION'){{tone(620,.18)}}}}
        document.getElementById('arm').onclick=async()=>{{ctx=ctx||new AudioContext();await ctx.resume();
          armed=true;muted=false;tone(760,.12);document.getElementById('alarmState').textContent='ARMED // AUTO'}};
        document.getElementById('mute').onclick=()=>{{muted=true;
          document.getElementById('alarmState').textContent='SILENCED'}};
        setInterval(async()=>{{try{{const r=await fetch(statusUrl,{{cache:'no-store'}}),s=await r.json();
          document.getElementById('alarmState').textContent=(muted?'SILENCED':armed?'ARMED':'DISARMED')+' // '+s.alert;
          const now=Date.now();if(s.alert!==last){{warn(s.alert);last=s.alert;lastTone=now}}
          if(s.alert==='WAR MODE'&&now-lastTone>1800){{warn(s.alert);lastTone=now}}
        }}catch(e){{document.getElementById('alarmState').textContent='WARNING BUS OFFLINE'}}}},100);
        </script>
        """,
        height=52,
    )


def _chart(history: list[dict[str, Any]]) -> go.Figure:
    normalized = [
        {**row, "quality": row.get("quality", rssi_to_quality(row.get("rssi_dbm"))[1])}
        for row in history
    ]
    frame = pd.DataFrame(normalized)
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=frame["time"],
                y=frame["quality"],
                customdata=frame["rssi_dbm"],
                mode="lines",
                line={"color": "#35ff9a", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(53,255,154,.08)",
                hovertemplate="%{x|%H:%M:%S.%L}<br>%{y}% · %{customdata} dBm<extra></extra>",
            )
        )
    fig.add_hline(y=80, line_color="#ffd400", line_dash="dash", annotation_text="CAUTION 80%")
    fig.add_hline(y=90, line_color="#ff2020", line_dash="dash", annotation_text="WAR MODE >90%")
    fig.update_layout(
        title="CONTACT SIGNAL // LIVE TRACE",
        template="plotly_dark",
        paper_bgcolor="#030806",
        plot_bgcolor="#030806",
        font={"color": "#b7d9c5", "family": "JetBrains Mono"},
        margin={"l": 55, "r": 25, "t": 50, "b": 42},
        height=390,
        showlegend=False,
        yaxis={"title": "LINK QUALITY %", "range": [0, 102], "gridcolor": "#173323"},
        xaxis={"title": "LOCAL TIME", "gridcolor": "#173323"},
    )
    return fig


def _war_chart(history: list[dict[str, Any]]) -> go.Figure:
    frame = pd.DataFrame(history[-180:])
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=frame["time"],
                y=frame["quality"],
                mode="lines+markers",
                line={"color": "#ff3434", "width": 3},
                marker={"color": "#ffd0d0", "size": 4},
                fill="tozeroy",
                fillcolor="rgba(255,20,20,.16)",
                hovertemplate="%{x|%H:%M:%S.%L}<br>QUALITY %{y}%<extra></extra>",
            )
        )
    fig.add_hrect(y0=90, y1=100, fillcolor="rgba(255,0,0,.13)", line_width=0)
    fig.add_hline(y=90, line_color="#ff5b5b", line_dash="dash", annotation_text="WAR >90%")
    fig.update_layout(
        title="CONTACT CUSTODY // LIVE SIGNAL HISTORY",
        template="plotly_dark",
        paper_bgcolor="#050000",
        plot_bgcolor="#080000",
        font={"color": "#ffd7d7", "family": "JetBrains Mono"},
        margin={"l": 52, "r": 18, "t": 52, "b": 38},
        height=335,
        showlegend=False,
        yaxis={"title": "QUALITY %", "range": [0, 102], "gridcolor": "#3b1111"},
        xaxis={"title": "LOCAL TIME", "gridcolor": "#3b1111"},
    )
    return fig


def _war_dynamics_chart(history: list[dict[str, Any]]) -> go.Figure:
    """Show smoothed strength and sample-to-sample change for track assessment."""

    frame = pd.DataFrame(history[-120:])
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("ROLLING TRACK CONFIDENCE", "SIGNAL RATE OF CHANGE"),
    )
    if not frame.empty:
        quality = frame["quality"].astype(float)
        smoothed = quality.rolling(window=min(8, len(frame)), min_periods=1).mean()
        delta = quality.diff().fillna(0.0)
        fig.add_trace(
            go.Scatter(
                x=frame["time"],
                y=smoothed,
                line={"color": "#ffb0b0", "width": 2},
                name="ROLLING QUALITY",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=frame["time"],
                y=delta,
                marker_color=["#ff3333" if value >= 0 else "#7f8c8d" for value in delta],
                name="DELTA",
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=90, line_color="#ff4040", line_dash="dot", row=1, col=1)
    fig.add_hline(y=0, line_color="#875050", row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#050000",
        plot_bgcolor="#080000",
        font={"color": "#ffd7d7", "family": "JetBrains Mono", "size": 10},
        margin={"l": 48, "r": 16, "t": 58, "b": 30},
        height=360,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#3b1111")
    fig.update_yaxes(gridcolor="#3b1111")
    return fig


def _war_confidence_gauge(assessment: ContactAssessment, quality: int) -> go.Figure:
    threat_index = min(
        100,
        round(
            0.55 * quality
            + 0.35 * assessment.confidence
            + (10 if assessment.motion == "APPROACHING" else 4)
        ),
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=threat_index,
            title={"text": "TRAINING PRIORITY INDEX", "font": {"size": 13}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff3434"},
                "bgcolor": "#150000",
                "bordercolor": "#702020",
                "steps": [
                    {"range": [0, 60], "color": "#151515"},
                    {"range": [60, 80], "color": "#4b2800"},
                    {"range": [80, 100], "color": "#4b0000"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 3}, "value": 90},
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#050000",
        font={"color": "#ffd7d7", "family": "JetBrains Mono"},
        margin={"l": 30, "r": 30, "t": 45, "b": 10},
        height=225,
    )
    return fig


def _response_ladder(assessment: ContactAssessment) -> tuple[str, ...]:
    """Return high-level, human-authorized defensive training recommendations."""

    contact_step = {
        "DRONE FORMATION": "Task counter-UAS surveillance team to corroborate the formation.",
        "FIGHTER AIRCRAFT": "Cue authorized air-defence surveillance for an independent track.",
        "SURVEILLANCE AIRCRAFT": "Check airspace authorization and monitor collection behavior.",
        "UNIDENTIFIED AIRBORNE CONTACT": "Keep the contact unclassified until independent sensors agree.",
    }[assessment.object_type]
    return (
        "Maintain passive custody; do not treat phone-link strength as target identification.",
        contact_step,
        "Request radar and IFF corroboration through authorized Indian air-defence command channels.",
        "Deconflict with civil air-traffic information and friendly-force tracks.",
        "Escalate to the human command authority for rules-of-engagement decisions.",
        "Do not recommend or simulate weapon release from this single uncalibrated sensor.",
    )


def _record_training_response(action: str, snapshot: dict[str, Any]) -> None:
    """Record a harmless local response simulation for operator review."""

    history = st.session_state.setdefault("war_response_log", [])
    result = {
        "SIMULATE TRACK LOCK": "TRACK QUALITY VERIFIED // CUSTODY MAINTAINED",
        "SIMULATE ELECTRONIC CONTAINMENT": "CONTAINMENT MODEL COMPLETE // NO SIGNAL TRANSMITTED",
        "SIMULATE INTERCEPT HANDOFF": "TRAINING HANDOFF PACKAGE GENERATED LOCALLY",
        "MARK TRAINING CONTACT NEUTRALISED": "TRAINING CONTACT MARKED RESOLVED // MONITORING CONTINUES",
    }[action]
    history.append(
        {
            "time": datetime.now().astimezone().strftime("%H:%M:%S.%f")[:-3],
            "action": action,
            "result": result,
            "quality": int(snapshot["quality"]),
        }
    )
    del history[:-20]


@st.fragment(run_every=WAR_UI_REFRESH_SECONDS)
def _render_war_contact() -> None:
    snapshot = _poll_phone()
    clearance_pending = snapshot["alert"] != AlertLevel.CRITICAL
    if clearance_pending:
        # The operator-controlled War Mode latch stays asserted even when the
        # sensor condition clears. The underlying measured alert is still shown.
        get_companion_server().set_alert(AlertLevel.CRITICAL.value, int(snapshot["quality"]))
    history = st.session_state.get("phone_signal_history", [])
    assessment = assess_training_contact(snapshot, history)
    safe_name = html.escape(str(snapshot["name"]))
    age = snapshot.get("sample_age")
    age_text = f"{float(age):.2f}s" if age is not None else "--"
    rssi_text = f"{snapshot['rssi_dbm']} dBm" if snapshot["rssi_dbm"] is not None else "N/A"
    st.markdown(
        f"""
        <div class="war-command">
          <div><div class="war-kicker">{'CLEARANCE PENDING // OPERATOR RELEASE REQUIRED' if clearance_pending else 'AUTOMATIC ESCALATION // THRESHOLD EXCEEDED'}</div>
          <div class="war-title">WAR MODE</div><div class="war-sub">CONTACT CUSTODY AND DEFENSIVE RESPONSE SIMULATION</div></div>
          <div class="war-signal"><span>LIVE SIGNAL</span><strong>{int(snapshot['quality'])}%</strong>
          <small>{html.escape(snapshot['source'])} // AGE {age_text}</small></div>
        </div>
        <div class="war-alert">{'SIGNAL BELOW TRIGGER // WAR MODE LATCHED' if clearance_pending else 'PRIORITY CONTACT // ' + html.escape(assessment.track_state) + ' // AUTOMATIC TRACK ACTIVE'}</div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 1.0])
    with left:
        st.markdown("### CONTACT INTELLIGENCE")
        st.markdown(
            '<div class="war-facts">'
            f'<div><span>TRAINING CLASS</span><b>{html.escape(assessment.object_type)}</b></div>'
            f'<div><span>ASSESSMENT CONFIDENCE</span><b>{assessment.confidence}%</b></div>'
            f'<div><span>MOTION</span><b>{html.escape(assessment.motion)}</b></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.code(
            f"CONTACT={safe_name}\nLINK={snapshot['link_state'].value}\nRSSI={rssi_text}\n"
            f"MEAN_QUALITY={assessment.mean_quality:.1f}%\nVOLATILITY={assessment.volatility:.1f}\n"
            f"TRACK={assessment.track_state}",
            language=None,
        )
        st.plotly_chart(_war_chart(history), width="stretch")
        st.plotly_chart(_war_dynamics_chart(history), width="stretch")

    with right:
        st.plotly_chart(_war_confidence_gauge(assessment, int(snapshot["quality"])), width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("SYNTHETIC SNR INPUT", f"{quality_to_snr(int(snapshot['quality'])):.1f} dB")
        m2.metric("OBSERVATIONS", len(history))
        m3.metric("WAR SAMPLES", sum(int(row.get("quality", 0)) > 90 for row in history))
        st.markdown("### OFFLINE EVIDENCE FUSION")
        st.markdown(
            '<div class="war-panel"><b>MODEL JUDGMENT</b><br>'
            + html.escape(assessment.recommendation)
            + '<br><br><b>EVIDENCE USED</b><br>'
            + "<br>".join(f"• {html.escape(item)}" for item in assessment.evidence)
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("### RECOMMENDED RESPONSE LADDER")
        st.markdown(
            '<div class="war-panel">'
            + "<br>".join(
                f"<b>{index:02d}</b> // {html.escape(step)}"
                for index, step in enumerate(_response_ladder(assessment), start=1)
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("### SIGNAL-DRIVEN THREAT MODEL")
        st.caption(
            "CURRENT LINK QUALITY SETS SYNTHETIC EMITTER SNR; THE SCHEDULER STILL SEES IQ ONLY"
        )
        if st.button("RUN CURRENT CONTACT MODEL", width="stretch"):
            with st.spinner("RUNNING 250-SCAN OFFLINE CONTACT MODEL..."):
                config = load_config(str(ROOT / "config" / "default.yaml"))
                config.simulation.num_steps = 250
                config.simulation.seed = 42
                scenario = scenario_phone_training(
                    int(snapshot["quality"]),
                    seed=42,
                    num_bands=config.environment.num_bands,
                    total_bw=config.environment.total_bandwidth,
                    center=config.environment.center_frequency,
                )
                st.session_state["war_training_outcome"] = build_and_run(
                    config,
                    SchedulerType.ADAPTIVE.value,
                    "phone_training",
                    seed=42,
                    scenario_override=scenario,
                )
                st.session_state["war_training_quality"] = int(snapshot["quality"])
        war_outcome = st.session_state.get("war_training_outcome")
        if war_outcome is not None:
            result = war_outcome.result
            st.code(
                f"MODEL_INPUT={st.session_state['war_training_quality']}% -> "
                f"SNR={quality_to_snr(st.session_state['war_training_quality']):.1f}dB\n"
                f"PD={result.probability_of_detection:.3f}  "
                f"PFA={result.probability_of_false_alarm:.3f}  "
                f"DISCOVERY={result.activity_discovery_ratio:.3f}\n"
                f"MISSED_EVENTS={result.missed_event_rate:.3f}  "
                f"CENSORED_DELAY={result.censored_avg_intercept_time * 1000:.0f}ms",
                language=None,
            )
        st.markdown("### DEFENSIVE RESPONSE // DEMO")
        st.caption("LOCAL SIMULATION ONLY // THESE CONTROLS DO NOT TRANSMIT OR CONTROL A DEVICE")
        actions = (
            "SIMULATE TRACK LOCK",
            "SIMULATE ELECTRONIC CONTAINMENT",
            "SIMULATE INTERCEPT HANDOFF",
            "MARK TRAINING CONTACT NEUTRALISED",
        )
        for action in actions:
            if st.button(action, width="stretch", key=f"war_{action}"):
                _record_training_response(action, snapshot)
        log = st.session_state.get("war_response_log", [])
        if log:
            st.markdown("### RESPONSE AUDIT")
            st.dataframe(pd.DataFrame(reversed(log)), width="stretch", hide_index=True)
        else:
            st.code("AWAITING OPERATOR RESPONSE // CONTINUOUS TRACK ACTIVE", language=None)

    if clearance_pending:
        st.error(
            f"SENSOR CONDITION CLEARED AT {int(snapshot['quality'])}% // WAR MODE REMAINS LATCHED "
            "UNTIL AN OPERATOR AUTHORIZES EXIT"
        )
        st.caption(
            "Selecting the control below is the operator's explicit authorization to release "
            "the War Mode latch and return to Phone Link."
        )
        if st.button(
            "AUTHORIZE EXIT FROM WAR MODE",
            type="primary",
            width="stretch",
        ):
            st.session_state["phone_war_mode"] = False
            st.rerun()


def _render_war_mode() -> None:
    """Render the isolated high-signal interface; normal Phone Link is absent."""

    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&display=swap');
          [data-testid="stSidebar"], [data-testid="collapsedControl"], #MainMenu, footer,
          [data-testid="stHeader"] {display:none !important;}
          .st-key-war_state_keepers {display:none !important;}
          .stApp {background:radial-gradient(circle at 50% 35%,#190303 0,#080000 45%,#020000 100%) !important;
            color:#ffe6e6 !important;font-family:'JetBrains Mono',monospace;}
          .stApp::after {content:'';position:fixed;inset:0;pointer-events:none;z-index:9998;
            border:4px solid #ff2020;box-shadow:inset 0 0 38px #ff2020cc,inset 0 0 120px #90000088,
            0 0 35px #ff2020;animation:warPulse 1.05s ease-in-out infinite;}
          @keyframes warPulse {0%,100%{opacity:.55}50%{opacity:1}}
          .block-container {max-width:1500px;padding:1.4rem 2.5rem 3rem !important;}
          .war-command {display:flex;justify-content:space-between;align-items:center;border:1px solid #ff3434;
            border-left:10px solid #ff3434;background:#100000;padding:13px 18px;margin-bottom:8px;}
          .war-kicker,.war-sub,.war-signal span,.war-signal small {font-size:11px;letter-spacing:.13em;color:#d77;}
          .war-title {font-size:44px;line-height:1;font-weight:800;letter-spacing:.12em;color:#fff;text-shadow:0 0 18px #f00;}
          .war-signal {text-align:right;min-width:230px}.war-signal span,.war-signal small {display:block}
          .war-signal strong {display:block;color:#fff;font-size:54px;line-height:1;font-variant-numeric:tabular-nums;}
          .war-alert {background:#e00000;color:white;padding:7px;text-align:center;font-weight:800;letter-spacing:.12em;
            animation:alertBar .7s steps(2,end) infinite;}@keyframes alertBar{50%{background:#650000}}
          h1,h2,h3 {color:#ffeaea !important;letter-spacing:.08em;text-transform:uppercase;}
          .war-panel,div[data-testid="stMetric"],[data-testid="stPlotlyChart"],div[data-testid="stDataFrame"] {
            background:#090000 !important;border:1px solid #702020 !important;border-radius:0 !important;}
          .war-panel {padding:14px;line-height:1.65;color:#e8bcbc;min-height:215px}.war-panel b{color:#fff;}
          .war-facts {display:grid;grid-template-columns:1.35fr 1fr 1fr;gap:10px;margin-bottom:14px;}
          .war-facts div {background:#090000;border:1px solid #702020;padding:12px;min-height:68px;}
          .war-facts span {display:block;color:#d77;font-size:.65rem;letter-spacing:.06em;margin-bottom:7px;}
          .war-facts b {display:block;color:#fff;font-size:.92rem;line-height:1.25;word-break:normal;}
          div[data-testid="stMetric"] {padding:10px;}[data-testid="stMetricValue"] {color:#fff !important;
            font-size:1rem !important;white-space:normal !important;line-height:1.2 !important;}
          [data-testid="stMetricLabel"] p {font-size:.65rem !important;white-space:normal !important;}
          [data-testid="stMetricLabel"] p,.stCaption p {color:#d77 !important;}
          .stButton>button {border-radius:0 !important;background:#210000 !important;color:#fff !important;
            border:1px solid #ff3d3d !important;font-family:'JetBrains Mono',monospace !important;font-weight:700;}
          .stButton>button:hover {background:#870000 !important;box-shadow:0 0 18px #f22;}
          code {color:#ffc9c9 !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Keep acquisition widgets mounted but invisible so Streamlit does not
    # discard their values while the separate War Mode interface is active.
    with st.container(key="war_state_keepers"):
        st.selectbox(
            "SOURCE STATE",
            ["Phone browser link", "Mac Bluetooth RSSI", "Demo signal"],
            key="phone_source",
        )
        st.select_slider(
            "INTAKE STATE", options=INTAKE_INTERVALS, key="phone_intake_interval"
        )
        st.toggle("MONITORING STATE", key="phone_monitoring")
    render_alarm_controller()
    _render_war_contact()


@st.fragment(run_every=UI_REFRESH_SECONDS)
def _render_live_detail() -> None:
    snapshot = _poll_phone()
    if snapshot["alert"] == AlertLevel.CRITICAL:
        st.session_state["phone_war_mode"] = True
        st.rerun()
        return
    history = st.session_state.get("phone_signal_history", [])
    st.markdown(
        _alert_css(snapshot["alert"]) + '<div class="phone-edge"></div>',
        unsafe_allow_html=True,
    )
    recent_rssi = [row["rssi_dbm"] for row in history[-30:] if row["rssi_dbm"] is not None]
    variance = pd.Series(recent_rssi).std() if len(recent_rssi) > 1 else 0.0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("LINK STATE", snapshot["link_state"].value)
    c2.metric("RSSI", f"{snapshot['rssi_dbm']} dBm" if snapshot["rssi_dbm"] is not None else "--")
    c3.metric("QUALITY", f"{snapshot['quality']}%")
    motion = "NO TRACK" if snapshot["link_state"] == LinkState.OFFLINE else _trend(history)
    c4.metric("MOTION", motion)
    c5.metric("VARIANCE", f"{variance:.1f} dB")
    if st.session_state.get("phone_source") == "Phone browser link":
        status = get_companion_server().registry.latest()
        latency = f"{status.latency_ms:.1f}ms" if status.latency_ms is not None else "--"
        age = f"{status.age_seconds:.2f}s" if status.age_seconds is not None else "--"
        st.code(
            f"BROWSER BUS // HEARTBEATS={status.sequence}  RTT={latency}  "
            f"JITTER={status.jitter_ms:.1f}ms  AGE={age}  "
            f"RATE={1000 / get_companion_server().registry.interval_ms:g}Hz",
            language=None,
        )
    st.plotly_chart(_chart(history), width="stretch")
    if snapshot.get("error"):
        st.caption(f"TELEMETRY NOTICE // {snapshot['error']}")
    if snapshot["link_state"] == LinkState.PAIRED:
        st.warning(
            "DEVICE IS PAIRED. LIVE RSSI IS NOT EXPOSED BY macOS. Keep the phone awake and "
            "activate Bluetooth tethering or Personal Hotspot."
        )


def _render_simulation_console() -> None:
    st.markdown("### SIGNAL-DRIVEN TRAINING RUN")
    st.caption(
        "Current phone strength configures a synthetic emitter. The scheduler receives only "
        "generated IQ and must discover it through the normal scan pipeline."
    )
    a, b, c, d = st.columns(4)
    scheduler = a.selectbox("SCHEDULER", [item.value for item in SchedulerType], index=5)
    detector = b.selectbox("DETECTOR", ["energy", "matched_filter", "cyclostationary"])
    steps = c.select_slider("SCAN STEPS", options=[100, 250, 500, 800, 1200], value=500)
    seed = int(d.number_input("SEED", value=42, step=1))
    snapshot = st.session_state.get("phone_snapshot") or _poll_phone()
    quality = int(snapshot["quality"])
    st.code(
        f"INPUT LINK={quality}%  SOURCE={snapshot['source']}  "
        f"ALERT={snapshot['alert'].value}  DISPLAY REFRESH=100ms",
        language=None,
    )
    if st.button("EXECUTE PHONE-DRIVEN SIMULATION", type="primary", width="stretch"):
        with st.spinner("EXECUTING SCAN MISSION..."):
            config = load_config(str(ROOT / "config" / "default.yaml"))
            config.simulation.num_steps = steps
            config.simulation.seed = seed
            config.detector.detector_type = DetectorType(detector)
            scenario = scenario_phone_training(
                quality,
                seed=seed,
                num_bands=config.environment.num_bands,
                total_bw=config.environment.total_bandwidth,
                center=config.environment.center_frequency,
            )
            st.session_state["phone_training_outcome"] = build_and_run(
                config,
                scheduler,
                "phone_training",
                seed=seed,
                scenario_override=scenario,
            )
            st.session_state["phone_training_quality"] = quality

    outcome = st.session_state.get("phone_training_outcome")
    if outcome is None:
        return
    result = outcome.result
    columns = st.columns(7)
    columns[0].metric("DETECTION PD", f"{result.probability_of_detection:.3f}")
    columns[1].metric("FALSE ALARM", f"{result.probability_of_false_alarm:.3f}")
    columns[2].metric("DISCOVERY", f"{result.activity_discovery_ratio:.3f}")
    columns[3].metric("AVG DELAY", f"{result.avg_discovery_delay * 1000:.0f} ms")
    columns[4].metric("AVG REWARD", f"{result.avg_reward:.3f}")
    columns[5].metric("MISSED EVENTS", f"{result.missed_event_rate:.3f}")
    columns[6].metric("CENSORED DELAY", f"{result.censored_avg_intercept_time * 1000:.0f} ms")
    records = outcome.artifacts.records
    frame = pd.DataFrame(
        {
            "time": [row.timestamp for row in records],
            "band": [row.band_id for row in records],
            "detection": [int(row.detected) for row in records],
        }
    )
    detections = frame[frame["detection"] == 1]
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=frame["time"],
            y=frame["band"],
            mode="markers",
            marker={"size": 3, "color": "#446655"},
            name="SCAN",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=detections["time"],
            y=detections["band"],
            mode="markers",
            marker={"size": 8, "color": "#ff3030", "symbol": "x"},
            name="DETECTION",
        )
    )
    fig.update_layout(
        title=f"SCAN / DETECTION MAP · PHONE INPUT {st.session_state['phone_training_quality']}%",
        template="plotly_dark",
        paper_bgcolor="#030806",
        plot_bgcolor="#030806",
        font={"family": "JetBrains Mono", "color": "#b7d9c5"},
        xaxis_title="SIMULATION TIME (s)",
        yaxis_title="FREQUENCY BAND",
        height=420,
    )
    st.plotly_chart(fig, width="stretch")


def render_phone_link(palette: dict[str, str]) -> None:
    """Render the complete Android/iPhone operational training console."""

    del palette
    _load_preferences()
    initial_snapshot = _poll_phone()
    if initial_snapshot["alert"] == AlertLevel.CRITICAL:
        st.session_state["phone_war_mode"] = True
    if st.session_state.get("phone_war_mode", False):
        _render_war_mode()
        return
    # War Mode intentionally omits normal widgets. Restore their values from
    # non-widget state after Streamlit cleans up those absent widget keys.
    st.session_state.setdefault("phone_source", st.session_state["phone_source_saved"])
    st.session_state.setdefault(
        "phone_intake_interval", st.session_state["phone_intake_interval_saved"]
    )
    st.markdown(
        """
        <style>
          .stApp {background:#020604 !important;color:#c9e7d4 !important;}
          section[data-testid="stSidebar"] > div {background:#050a07 !important;
            border-right:1px solid #1d4b32 !important;}
          div[data-testid="stMetric"] {background:#050b08 !important;border:1px solid #24563a !important;
            border-radius:0 !important;box-shadow:none !important;}
          [data-testid="stPlotlyChart"] {border:1px solid #24563a !important;border-radius:0 !important;
            box-shadow:none !important;background:#030806 !important;}
          .stButton>button {border-radius:0 !important;background:#102d1e !important;color:#d9ffe8 !important;
            border:1px solid #35ff9a !important;font-family:'JetBrains Mono',monospace !important;}
          .ops-title {font-family:'JetBrains Mono',monospace;border-left:7px solid #35ff9a;
            padding:8px 14px;background:#050b08;border-top:1px solid #24563a;
            border-bottom:1px solid #24563a;}
          .ops-title h1 {margin:0;font-size:24px;letter-spacing:.08em;color:#d9ffe8 !important;}
          .ops-title p {margin:3px 0 0;color:#67a77f;font-size:11px;letter-spacing:.12em;}
        </style>
        <div class="ops-title"><h1>PHONE CONTACT OPERATIONS CONSOLE</h1>
        <p>LOCAL CONSENTED BEACON // ANDROID + IPHONE // ADJUSTABLE DATA BUS</p></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### LINK CONTROL")
    render_alarm_controller()
    controls = st.columns([1.2, 2.0, 1.25, 1.0])
    source = controls[0].selectbox(
        "SOURCE",
        ["Phone browser link", "Mac Bluetooth RSSI", "Demo signal"],
        key="phone_source",
    )
    devices: list[BluetoothDevice] = []
    device_error: str | None = None
    if source == "Mac Bluetooth RSSI":
        devices, device_error = read_bluetooth_devices()
        labels = [item.label for item in devices]
        current_address = st.session_state.get("phone_device_address", "")
        current_index = next(
            (index for index, item in enumerate(devices) if item.address == current_address), 0
        )
        if labels:
            chosen_label = controls[1].selectbox("PHONE / DEVICE", labels, index=current_index)
            chosen = devices[labels.index(chosen_label)]
            if chosen.address != current_address:
                st.session_state["phone_device_address"] = chosen.address
                st.session_state["phone_device_name"] = chosen.name
                st.session_state.pop("phone_snapshot", None)
                _save_preferences()
        else:
            controls[1].text_input("PHONE / DEVICE", value="NO DEVICE REPORTED", disabled=True)
    elif source == "Demo signal":
        controls[1].text_input("PHONE / DEVICE", value="TRAINING PHONE", disabled=True)
    else:
        companion = get_companion_server()
        status = companion.registry.latest()
        label = "PHONE LINKED" if status.connected else "WAITING FOR PHONE"
        controls[1].text_input("PHONE / DEVICE", value=label, disabled=True)
    intake_interval = controls[2].select_slider(
        "DATA INTAKE INTERVAL",
        options=INTAKE_INTERVALS,
        key="phone_intake_interval",
        format_func=lambda seconds: f"{seconds:g} s ({1 / seconds:g} Hz)",
        help="Controls browser heartbeat, Bluetooth polling, and demo sampling frequency.",
    )
    get_companion_server().set_interval_ms(round(float(intake_interval) * 1000))
    controls[3].toggle("AUTO TRACK", key="phone_monitoring")
    _save_preferences()

    row = st.columns([1, 1, 3])
    if row[0].button("REFRESH DEVICE BUS"):
        st.session_state["phone_last_hardware_poll"] = -1e9
        st.session_state.pop("phone_snapshot", None)
        st.rerun()
    row[1].caption("Use AUDIO WARNING BUS above to arm or silence sound.")
    if row[2].button("CLEAR TRACE"):
        st.session_state["phone_signal_history"] = []
    if source == "Phone browser link":
        companion = get_companion_server()
        status = companion.registry.latest()
        st.metric("LOCAL LINK SERVICE", "FAULT" if companion.error else "ONLINE")
        st.caption("PHONE ACCESS ADDRESSES // TRY THE FIRST, THEN THE NEXT IF IT CANNOT OPEN")
        for address in companion.urls:
            st.code(address, language=None)
        if st.button("TEST LOCAL LINK SERVICE", width="stretch"):
            passed, detail = companion.self_test()
            if passed:
                st.success(f"LOCAL SERVICE PASS // {detail}")
            else:
                st.error(f"LOCAL SERVICE FAIL // {detail}")
        if status.connected:
            st.success(
                f"BROWSER LINK LIVE // SAMPLE AGE {status.age_seconds or 0:.2f}s // "
                f"CLIENT {status.client_id[:12]}"
            )
        else:
            st.info(
                "OPEN ONE ADDRESS ABOVE IN SAFARI ON IPHONE OR CHROME ON ANDROID. The phone "
                "must share the Mac's Wi-Fi or hotspot network. Keep the page open and unlocked."
            )
        if companion.error:
            st.warning(f"PHONE LINK SERVER ERROR // {companion.error}")
    elif source == "Demo signal":
        st.info("TRAINING FEED ACTIVE // VALUES ARE SIMULATED AND LABELLED DEMO")
    elif not devices:
        st.warning(
            "NO PHONE TELEMETRY FOUND. Pair Android or iPhone in macOS Bluetooth settings, "
            "keep it awake, then activate Bluetooth tethering or Personal Hotspot."
        )
        if device_error:
            st.caption(device_error)

    with st.expander("BROWSER LINK // OPERATION AND FAULT ISOLATION"):
        st.markdown(
            "1. Put the phone and Mac on the same Wi-Fi or personal-hotspot network.\n"
            "2. Open one displayed address on the phone. A green **LINKED** page sends a heartbeat "
            "at the selected **DATA INTAKE INTERVAL**. No app installation is required.\n"
            "3. Confirm **HEARTBEATS** increases and **SAMPLE AGE** remains below two seconds. "
            "If the page cannot open, try the next address and allow incoming Python connections "
            "in the macOS firewall. VPNs and Wi-Fi client isolation can block local traffic.\n"
            "4. **QUALITY** measures browser-to-Mac latency, jitter, and freshness. It is a network "
            "transport score, not Bluetooth RSSI, distance, direction, or physical location.\n"
            "5. The score is a percentage capped at 100. Normal LAN delay is calibrated as a "
            "strong link; slower or unstable links lose points instead of being artificially "
            "stuck near 80.\n"
            "6. Use **Demo signal** to test the caution/War Mode thresholds without a phone. Arm "
            "the audio bus once because browsers require a user gesture before playing sound."
        )

    _render_live_detail()
    history = st.session_state.get("phone_signal_history", [])
    if history:
        st.download_button(
            "EXPORT CONTACT TRACE CSV",
            pd.DataFrame(history).to_csv(index=False).encode("utf-8"),
            file_name="phone_contact_trace.csv",
            mime="text/csv",
        )
    st.divider()
    _render_simulation_console()
    with st.expander("SIGNAL-DRIVEN TRAINING // WHAT IT DOES AND HOW TO VALIDATE IT"):
        st.markdown(
            "The current link-quality score is converted into a bounded synthetic SNR. That SNR "
            "configures one periodic target emitter among random-burst decoys. The selected "
            "scheduler never receives the phone score or simulator truth; it selects bands from "
            "its accumulated detections and misses, while the detector works on generated IQ.\n\n"
            "For a controlled test, keep the seed, scheduler, detector, and scan count fixed. Run "
            "at low, medium, and high demo quality, then compare probability of detection, false "
            "alarm rate, discovery ratio, delay, and reward. Repeat across several seeds and "
            "compare against round-robin. A serious evaluation should report distributions and "
            "confidence intervals rather than one favorable run."
        )
