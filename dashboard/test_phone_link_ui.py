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


def test_phone_console_renders_without_exception():
    app = AppTest.from_string(APP, default_timeout=15).run()

    assert not app.exception
    assert "PHONE CONTACT OPERATIONS CONSOLE" in app.markdown[0].value
    assert app.toggle[0].label == "AUTO TRACK"
    assert app.select_slider[0].label == "DATA INTAKE INTERVAL"
    assert app.select_slider[0].value == 0.1
    assert app.button[-1].label == "EXECUTE PHONE-DRIVEN SIMULATION"
