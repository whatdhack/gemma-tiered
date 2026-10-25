"""Live Device Panel: a browser view of the simulated hardware driven by the router.

Serves panel.html and streams every routed result to it via Server-Sent Events.
Standard library only; runs on a daemon thread so it exits with the router.
"""

import json
import queue
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PANEL_HTML = Path(__file__).with_name("panel.html")
HISTORY_LIMIT = 200


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass  # Browsers drop keep-alive sockets routinely; don't spray tracebacks on the router console


class DevicePanel:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self._clients = []
        self._history = []
        self._lock = threading.Lock()
        self._server = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Keep the router console clean

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = PANEL_HTML.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/events":
                    panel._stream(self)
                else:
                    self.send_error(404)

        try:
            self._server = _QuietServer((self.host, self.port), Handler)
        except OSError:
            # Preferred port busy -> let the OS pick a free one
            self._server = _QuietServer((self.host, 0), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def _stream(self, handler: BaseHTTPRequestHandler) -> None:
        q = queue.Queue()
        with self._lock:
            backlog = list(self._history)
            self._clients.append(q)

        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()
        try:
            # Replay past events so a freshly opened page shows the current state
            for ev in backlog:
                self._write(handler, ev, replay=True)
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    handler.wfile.write(b": ping\n\n")
                    handler.wfile.flush()
                    continue
                self._write(handler, ev)
        except OSError:
            pass  # Browser tab closed
        finally:
            with self._lock:
                self._clients.remove(q)

    @staticmethod
    def _write(handler: BaseHTTPRequestHandler, ev: dict, replay: bool = False) -> None:
        data = json.dumps({**ev, "replay": replay})
        handler.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        handler.wfile.flush()

    def publish(self, res) -> None:
        """Push a RouteResult to every connected panel."""
        payload = res.output_payload
        if res.tier_selected == "Tier 1 (Small Brain)":
            ev = {
                "type": "command",
                "command": payload.get("command", {}),
                "engine": payload.get("engine"),
            }
        else:
            out = payload.get("reflection_output", {})
            ev = {
                "type": "escalation",
                "reason": payload.get("abstention_reason"),
                "engine": out.get("engine"),
                "mode": out.get("mode", "diagnosis"),
                "diagnosis": out.get("report") or out.get("diagnosis", ""),
            }
        ev.update(prompt=res.prompt, tier=res.tier_selected, latency_ms=res.latency_ms, ts=time.time())

        with self._lock:
            self._history.append(ev)
            del self._history[:-HISTORY_LIMIT]
            clients = list(self._clients)
        for q in clients:
            q.put(ev)
