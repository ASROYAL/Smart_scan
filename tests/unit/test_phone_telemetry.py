"""Phone-link state, alert, and simulation bridge tests."""

from smartscan.simulation.scenarios import scenario_phone_training
from smartscan.telemetry.phone import (
    AlertLevel,
    AlertMemory,
    LinkState,
    advance_alert,
    classify_link,
    quality_to_snr,
    rssi_to_quality,
)


def test_missing_connected_flag_does_not_mark_paired_phone_offline():
    assert classify_link(
        paired=True, explicitly_connected=False, rssi_dbm=None, sample_age=None
    ) == LinkState.PAIRED


def test_recent_rssi_proves_live_link_for_iphone_or_android():
    assert classify_link(
        paired=True, explicitly_connected=False, rssi_dbm=-58, sample_age=0.4
    ) == LinkState.LIVE


def test_alert_thresholds_debounce_and_hysteresis():
    state = AlertMemory()
    state = advance_alert(state, 82)
    assert state.level == AlertLevel.NORMAL
    state = advance_alert(state, 82)
    assert state.level == AlertLevel.CAUTION
    state = advance_alert(state, 94)
    state = advance_alert(state, 94)
    assert state.level == AlertLevel.CRITICAL
    state = advance_alert(state, 88)
    assert state.level == AlertLevel.CRITICAL


def test_war_mode_requires_signal_above_ninety_percent():
    state = AlertMemory()
    state = advance_alert(state, 90)
    state = advance_alert(state, 90)
    assert state.level == AlertLevel.CAUTION

    state = advance_alert(state, 91)
    state = advance_alert(state, 91)
    assert state.level == AlertLevel.CRITICAL


def test_rssi_quality_and_training_snr_are_bounded():
    assert rssi_to_quality(-40) == ("EXCELLENT", 100)
    assert rssi_to_quality(None) == ("UNAVAILABLE", 0)
    assert quality_to_snr(-1) == -6.0
    assert quality_to_snr(101) == 0.0


def test_phone_training_scenario_uses_strength_only_as_environment_input():
    weak = scenario_phone_training(10, seed=4)
    strong = scenario_phone_training(95, seed=4)
    assert weak.emitters[0].center_frequency == strong.emitters[0].center_frequency
    assert weak.emitters[0].snr_db < strong.emitters[0].snr_db
    assert weak.emitters[0].waveform.value == "digital"
    assert strong.name == "phone_training"
