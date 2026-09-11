"""Tests for the local Bluetooth telemetry parser."""

from phone_link import parse_system_profiler, signal_quality


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
