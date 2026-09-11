"""Tests for local phone telemetry and training contact assessment."""

from datetime import UTC, datetime

from phone_link import (
    _response_ladder,
    assess_training_contact,
    build_training_bridge_input,
    parse_system_profiler,
    signal_quality,
)

from smartscan.telemetry.phone import AlertLevel, LinkState


def test_parse_connected_android_phone_with_rssi():
    payload = {
        "SPBluetoothDataType": [
            {
                "devices_list": [
                    {
                        "Aijaz's Android": {
                            "device_address": "AA-BB-CC-DD-EE-FF",
                            "device_connected": "attrib_Yes",
                            "device_rssi": "-58",
                        }
                    }
                ]
            }
        ]
    }

    devices = parse_system_profiler(payload)

    assert len(devices) == 1
    assert devices[0].name == "Aijaz's Android"
    assert devices[0].connected is True
    assert devices[0].rssi_dbm == -58


def test_parse_missing_rssi_is_honest():
    payload = {
        "SPBluetoothDataType": [
            {
                "devices_list": [
                    {
                        "iPhone": {
                            "device_address": "00-11-22-33-44-55",
                            "device_connected": "attrib_No",
                        }
                    }
                ]
            }
        ]
    }

    device = parse_system_profiler(payload)[0]
    assert device.connected is False
    assert device.rssi_dbm is None


def test_signal_quality_thresholds_and_bounds():
    assert signal_quality(-45) == ("Excellent", 92)
    assert signal_quality(-60)[0] == "Good"
    assert signal_quality(-72)[0] == "Fair"
    assert signal_quality(-90)[0] == "Weak"
    assert signal_quality(-120) == ("Weak", 0)
    assert signal_quality(None) == ("Unavailable", 0)


def _critical_snapshot(name: str) -> dict:
    return {
        "name": name,
        "rssi_dbm": -43,
        "quality": 94,
        "quality_label": "EXCELLENT",
        "link_state": LinkState.LIVE,
        "alert": AlertLevel.CRITICAL,
        "sample_age": 0.2,
        "source": "DEMO",
        "error": None,
    }


def test_training_assessment_maps_phone_to_fighter_with_observation_evidence():
    history = [
        {"time": datetime.now(UTC), "quality": quality, "rssi_dbm": -50}
        for quality in range(82, 95)
    ]

    assessment = assess_training_contact(_critical_snapshot("TRAINING PHONE"), history)

    assert assessment.object_type == "FIGHTER AIRCRAFT"
    assert assessment.motion == "APPROACHING"
    assert 32 <= assessment.confidence <= 97
    assert "13 recent observations" in assessment.evidence[1]
    assert "no ground-truth emitter label" in assessment.evidence[-1]


def test_training_assessment_maps_airpods_to_drone_formation():
    history = [
        {"time": datetime.now(UTC), "quality": 94, "rssi_dbm": -43}
        for _ in range(15)
    ]

    assessment = assess_training_contact(_critical_snapshot("Aijaz AirPods Pro"), history)

    assert assessment.object_type == "DRONE FORMATION"
    assert assessment.motion == "STABLE"


def test_training_assessment_maps_other_devices_without_claiming_real_identification():
    history = [
        {"time": datetime.now(UTC), "quality": 93, "rssi_dbm": -44} for _ in range(12)
    ]

    watch = assess_training_contact(_critical_snapshot("Field Watch"), history)
    unknown = assess_training_contact(_critical_snapshot("BT-DEVICE-17"), history)

    assert watch.object_type == "SURVEILLANCE AIRCRAFT"
    assert unknown.object_type == "UNIDENTIFIED AIRBORNE CONTACT"
    assert "independent sensors" in _response_ladder(unknown)[1][2]
    assert _response_ladder(unknown)[-1][1] == "NO WEAPON RECOMMENDATION"


def test_training_bridge_rejects_zero_or_offline_input():
    snapshot = _critical_snapshot("iPhone")
    snapshot["quality"] = 0
    snapshot["link_state"] = LinkState.OFFLINE
    snapshot["source"] = "PHONE WEB"

    bridge = build_training_bridge_input(snapshot)

    assert bridge.valid is False
    assert bridge.synthetic_snr_db == -8.0
    assert "NO LIVE TELEMETRY" in bridge.status
    assert bridge.basis.startswith("NETWORK TRANSPORT PROXY")


def test_training_bridge_captures_valid_demo_proxy():
    bridge = build_training_bridge_input(_critical_snapshot("TRAINING PHONE"))

    assert bridge.valid is True
    assert bridge.quality == 94
    assert bridge.synthetic_snr_db == 27.72
