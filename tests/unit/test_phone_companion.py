"""Tests for the cross-platform phone browser companion telemetry."""

import json
from urllib.request import urlopen

import pytest

from smartscan.telemetry.companion import CompanionRegistry, CompanionServer, link_quality


def test_link_quality_penalizes_latency_and_jitter():
    assert link_quality(10.0, 1.0, 0.1) > link_quality(80.0, 15.0, 0.1)
    assert link_quality(10.0, 1.0, 3.0) == 0
    assert 90 <= link_quality(40.0, 5.0, 0.1) <= 100


def test_slow_intake_interval_extends_freshness_window():
    assert link_quality(25.0, 2.0, 3.0, expected_interval_seconds=5.0) > 90
    assert link_quality(25.0, 2.0, 13.0, expected_interval_seconds=5.0) == 0


def test_registry_reports_recent_heartbeat_as_connected():
    registry = CompanionRegistry()
    registry.heartbeat("iphone", 12.0)
    registry.heartbeat("iphone", 14.0)

    status = registry.latest()
    assert status.connected is True
    assert status.client_id == "iphone"
    assert status.sequence == 2
    assert status.quality > 80


def test_companion_http_heartbeat_end_to_end():
    server = CompanionServer(host="127.0.0.1", port=0)
    server.start()
    if server._server is None:
        pytest.skip(f"local listener unavailable in sandbox: {server.error}")
    port = server._server.server_port
    assert server.port == port
    assert all(f":{port}/" in address for address in server.urls)
    assert server.self_test()[0] is True
    server.set_interval_ms(500)
    address = (
        f"http://127.0.0.1:{port}/heartbeat?token={server.token}"
        "&client=iphone-test&rtt=11.5"
    )
    with urlopen(address, timeout=2) as response:
        payload = json.loads(response.read())

    assert payload["ok"] is True
    assert payload["interval_ms"] == 500
    assert payload["quality"] > 80
    assert server.registry.latest().client_id == "iphone-test"

    server.set_alert("WAR MODE", 94)
    status_address = f"http://127.0.0.1:{port}/status?token={server.token}"
    with urlopen(status_address, timeout=2) as response:
        status = json.loads(response.read())
    assert status == {"alert": "WAR MODE", "quality": 94}
    server._server.shutdown()
