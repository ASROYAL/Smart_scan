"""Streamlit smoke test for the phone operations console."""

from streamlit.testing.v1 import AppTest

APP = """
import sys
sys.path.insert(0, "dashboard")
from phone_link import render_phone_link

palette = {
    "acc": "#f90", "acc2": "#0cf", "good": "#0c6", "bad": "#f33",
    "tx": "#fff", "grid": "#173323", "plot_tmpl": "plotly_dark",
}
render_phone_link(palette)
"""

WAR_APP = """
import sys
from datetime import datetime, timezone
sys.path.insert(0, "dashboard")
import phone_link
from smartscan.telemetry.phone import AlertLevel, LinkState

snapshot = {
    "name": "TRAINING PHONE", "rssi_dbm": -43, "quality": 95,
    "quality_label": "EXCELLENT", "link_state": LinkState.LIVE,
    "alert": AlertLevel.CRITICAL, "sample_age": 0.1, "source": "DEMO", "error": None,
}
phone_link._poll_phone = lambda: snapshot
phone_link.st.session_state["phone_signal_history"] = [
    {"time": datetime.now(timezone.utc), "quality": q, "rssi_dbm": -43,
     "device": "TRAINING PHONE", "source": "DEMO"}
    for q in range(82, 96)
]
phone_link._render_war_mode()
"""

WAR_CLEAR_APP = WAR_APP.replace(
    '"alert": AlertLevel.CRITICAL',
    '"alert": AlertLevel.NORMAL',
).replace('"quality": 95', '"quality": 72')


def test_phone_console_renders_without_exception():
    app = AppTest.from_string(APP, default_timeout=15).run()

    assert not app.exception
    assert "PHONE CONTACT OPERATIONS CONSOLE" in app.markdown[0].value
    assert app.toggle[0].label == "AUTO TRACK"
    assert app.select_slider[0].label == "DATA INTAKE INTERVAL"
    assert app.select_slider[0].value == 0.1
    assert app.button[-1].label == "EXECUTE PHONE-DRIVEN SIMULATION"


def test_war_mode_isolated_interface_renders_response_controls():
    app = AppTest.from_string(WAR_APP, default_timeout=15).run()

    assert not app.exception
    page = " ".join(item.value for item in app.markdown)
    assert "WAR MODE" in page
    assert "PHONE CONTACT OPERATIONS CONSOLE" not in page
    assert "FIGHTER AIRCRAFT" in page
    labels = [button.label for button in app.button]
    assert "SIMULATE TRACK LOCK" in labels
    assert "MARK TRAINING CONTACT NEUTRALISED" in labels


def test_war_mode_requires_operator_authorization_after_signal_clears():
    app = AppTest.from_string(WAR_CLEAR_APP, default_timeout=15).run()

    assert not app.exception
    page = " ".join(item.value for item in app.markdown)
    assert "WAR MODE" in page
    assert "CLEARANCE PENDING" in page
    exit_button = next(
        button for button in app.button if button.label == "AUTHORIZE EXIT FROM WAR MODE"
    )
    assert not exit_button.disabled
