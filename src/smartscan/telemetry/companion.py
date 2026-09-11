"""Local browser companion link for iPhone and Android training devices."""

from __future__ import annotations

import json
import secrets
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen


@dataclass(frozen=True)
class CompanionStatus:
    connected: bool
    client_id: str
    latency_ms: float | None
    jitter_ms: float
    quality: int
    age_seconds: float | None
    sequence: int


def link_quality(
    latency_ms: float | None,
    jitter_ms: float,
    age_seconds: float,
    expected_interval_seconds: float = 0.1,
) -> int:
    """Calculate a bounded network-link score from observable transport health."""

    disconnect_after = max(2.0, expected_interval_seconds * 2.5)
    if latency_ms is None or age_seconds > disconnect_after:
        return 0
    # A healthy LAN commonly has 15-50 ms RTT plus browser scheduling jitter.
    # Treat that as a strong link and reserve large penalties for material delay.
    freshness_grace = max(0.3, expected_interval_seconds * 1.25)
    penalty = (
        max(0.0, latency_ms - 15.0) * 0.15
        + jitter_ms * 0.35
        + max(0.0, age_seconds - freshness_grace) * 12.0
    )
    return max(0, min(100, round(100 - penalty)))


class CompanionRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: dict[str, dict[str, float | int]] = {}
        self._interval_ms = 100

    @property
    def interval_ms(self) -> int:
        with self._lock:
            return self._interval_ms

    def set_interval_ms(self, interval_ms: int) -> None:
        with self._lock:
            self._interval_ms = max(100, min(5000, int(interval_ms)))

    def heartbeat(self, client_id: str, latency_ms: float) -> None:
        now = time.monotonic()
        with self._lock:
            previous = self._clients.get(client_id, {})
            prior_latency = float(previous.get("latency_ms", latency_ms))
            prior_jitter = float(previous.get("jitter_ms", 0.0))
            jitter = 0.75 * prior_jitter + 0.25 * abs(latency_ms - prior_latency)
            self._clients[client_id] = {
                "seen": now,
                "latency_ms": max(0.0, min(latency_ms, 5000.0)),
                "jitter_ms": jitter,
                "sequence": int(previous.get("sequence", 0)) + 1,
            }

    def latest(self) -> CompanionStatus:
        now = time.monotonic()
        with self._lock:
            if not self._clients:
                return CompanionStatus(False, "", None, 0.0, 0, None, 0)
            client_id, data = max(self._clients.items(), key=lambda item: float(item[1]["seen"]))
            age = now - float(data["seen"])
            latency = float(data["latency_ms"])
            jitter = float(data["jitter_ms"])
            interval_seconds = self._interval_ms / 1000.0
            return CompanionStatus(
                connected=age <= max(2.0, interval_seconds * 2.5),
                client_id=client_id,
                latency_ms=latency,
                jitter_ms=jitter,
                quality=link_quality(latency, jitter, age, interval_seconds),
                age_seconds=age,
                sequence=int(data["sequence"]),
            )


class CompanionServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self.token = secrets.token_urlsafe(9)
        self.registry = CompanionRegistry()
        self.error: str | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._alert_lock = threading.Lock()
        self._alert = "NORMAL"
        self._alert_quality = 0

    @property
    def local_ips(self) -> list[str]:
        """Return usable LAN addresses, with the most likely address first."""

        addresses: list[str] = []

        def add(address: str) -> None:
            address = address.strip()
            if address and not address.startswith("127.") and address not in addresses:
                addresses.append(address)

        # macOS commonly uses en0 for Wi-Fi, but USB tethering and different Mac
        # models may expose the active route on another interface.
        for interface in ("en0", "en1", "en2", "en3"):
            try:
                result = subprocess.run(
                    ["ipconfig", "getifaddr", interface],
                    capture_output=True,
                    text=True,
                    timeout=0.5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                break
            if result.returncode == 0:
                add(result.stdout)

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("8.8.8.8", 80))
                add(str(probe.getsockname()[0]))
        except OSError:
            pass
        try:
            for address in socket.gethostbyname_ex(socket.gethostname())[2]:
                add(address)
        except OSError:
            pass
        return addresses or ["localhost"]

    @property
    def local_ip(self) -> str:
        return self.local_ips[0]

    @property
    def url(self) -> str:
        return f"http://{self.local_ip}:{self.port}/?token={self.token}"

    @property
    def urls(self) -> list[str]:
        return [f"http://{address}:{self.port}/?token={self.token}" for address in self.local_ips]

    @property
    def status_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/status?token={self.token}"

    def set_alert(self, alert: str, quality: int) -> None:
        with self._alert_lock:
            self._alert = alert
            self._alert_quality = quality

    def set_interval_ms(self, interval_ms: int) -> None:
        self.registry.set_interval_ms(interval_ms)

    def alert_status(self) -> dict[str, str | int]:
        with self._alert_lock:
            return {"alert": self._alert, "quality": self._alert_quality}

    def self_test(self, timeout: float = 1.0) -> tuple[bool, str]:
        """Verify that the token-protected loopback endpoint is responding."""

        if self.error:
            return False, self.error
        try:
            with urlopen(self.status_url, timeout=timeout) as response:
                payload = json.loads(response.read())
        except (OSError, ValueError) as exc:
            return False, str(exc)
        if "alert" not in payload or "quality" not in payload:
            return False, "invalid status payload"
        return True, f"HTTP {response.status} on 127.0.0.1:{self.port}"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                del format, args

            def _authorized(self, query: dict[str, list[str]]) -> bool:
                return query.get("token", [""])[0] == owner.token

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                if not self._authorized(query):
                    self.send_error(403)
                    return
                if parsed.path == "/heartbeat":
                    client_id = query.get("client", ["PHONE"])[0][:64]
                    try:
                        latency = float(query.get("rtt", ["0"])[0])
                    except ValueError:
                        latency = 0.0
                    owner.registry.heartbeat(client_id, latency)
                    status = owner.registry.latest()
                    body = json.dumps(
                        {
                            "ok": True,
                            "server_ms": time.time_ns() // 1_000_000,
                            "quality": status.quality,
                            "interval_ms": owner.registry.interval_ms,
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/status":
                    body = json.dumps(owner.alert_status()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = _companion_html(owner.token).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        last_error: OSError | None = None
        candidates = [0] if self.port == 0 else range(self.port, self.port + 5)
        for candidate in candidates:
            try:
                self._server = ThreadingHTTPServer((self.host, candidate), Handler)
                self.port = int(self._server.server_port)
                break
            except OSError as exc:
                last_error = exc
        if self._server is None:
            self.error = str(last_error or "no local port available")
            return
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()


def _companion_html(token: str) -> str:
    return f"""<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartScan Phone Link</title><style>
body{{background:#020604;color:#bfffd6;font-family:monospace;margin:0;padding:28px}}
.box{{border:1px solid #35ff9a;padding:22px;max-width:520px;margin:auto}}
h1{{font-size:22px;letter-spacing:.08em}}#state{{font-size:34px;color:#35ff9a}}
.bar{{height:12px;border:1px solid #35ff9a;margin-top:18px}}.fill{{height:100%;background:#35ff9a;width:0}}
</style></head><body><div class="box"><div>SMARTSCAN // CONSENTED DEVICE LINK</div>
<h1>PHONE COMPANION ACTIVE</h1><div id="state">CONNECTING</div>
<p>Keep this page open while the Mac dashboard is running.</p>
<div class="bar"><div class="fill" id="fill"></div></div><p id="metric">RTT --</p></div>
<script>
const token={json.dumps(token)};
let id=localStorage.getItem('smartscan-client');
if(!id){{id=(crypto.randomUUID?crypto.randomUUID():String(Date.now()));localStorage.setItem('smartscan-client',id)}}
let rtt=0,intervalMs=100;
async function beat(){{const t=performance.now();try{{
 const response=await fetch('/heartbeat?token='+encodeURIComponent(token)+'&client='+encodeURIComponent(id)+'&rtt='+rtt.toFixed(1),{{cache:'no-store'}});
 const status=await response.json();intervalMs=status.interval_ms||100;
 rtt=performance.now()-t;document.getElementById('state').textContent='LINKED';
 const q=status.quality;
 document.getElementById('metric').textContent='ROUND TRIP '+rtt.toFixed(1)+' ms // QUALITY '+q+'%';
 document.getElementById('fill').style.width=q+'%';
}}catch(e){{document.getElementById('state').textContent='LINK LOST'}}setTimeout(beat,intervalMs)}}beat();
</script></body></html>"""


_SERVER: CompanionServer | None = None


def get_companion_server() -> CompanionServer:
    global _SERVER
    if _SERVER is None:
        _SERVER = CompanionServer()
        _SERVER.start()
    return _SERVER
