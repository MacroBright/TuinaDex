"""Loopback-only HTTP server for the guided stereo calibration interface."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .service import ServiceError


_GET_ROUTES = frozenset(("/", "/api/status", "/api/preview.jpg"))
_POST_ROUTES = frozenset(
    (
        "/api/capture",
        "/api/reject-last",
        "/api/calibrate",
        "/api/retry-cameras",
    )
)
_ALL_ROUTES = _GET_ROUTES | _POST_ROUTES
_INDEX_PATH = Path(__file__).with_name("static") / "index.html"


class _CalibrationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_server(host: str, port: int, service: Any) -> ThreadingHTTPServer:
    """Build a testable server, refusing any non-loopback bind address."""
    if host != "127.0.0.1":
        raise ValueError("web server must bind to exactly 127.0.0.1")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("web server port must be between 0 and 65535")
    server = _CalibrationHTTPServer((host, port), _handler_factory())
    server.calibration_service = service  # type: ignore[attr-defined]
    return server


def _handler_factory() -> type[BaseHTTPRequestHandler]:
    class CalibrationRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        @property
        def service(self) -> Any:
            return self.server.calibration_service  # type: ignore[attr-defined]

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlsplit(self.path).path
            if path in _POST_ROUTES:
                self._method_not_allowed("POST")
                return
            if path not in _GET_ROUTES:
                self._send_json(404, {"error": "not found"})
                return
            try:
                if path == "/":
                    body = _INDEX_PATH.read_bytes()
                    self._send_bytes(200, body, "text/html; charset=utf-8")
                elif path == "/api/status":
                    self._send_json(200, self.service.status())
                else:
                    payload = self.service.preview_jpeg()
                    if not isinstance(payload, bytes):
                        raise TypeError("preview payload is not bytes")
                    self._send_bytes(200, payload, "image/jpeg")
            except ServiceError as exc:
                self._send_json(409, {"error": _error_text(exc)})
            except Exception:
                self._send_internal_error()

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlsplit(self.path).path
            if path in _GET_ROUTES:
                self._method_not_allowed("GET")
                return
            if path not in _POST_ROUTES:
                self._send_json(404, {"error": "not found"})
                return
            if not self._origin_allowed():
                self._send_json(403, {"error": "forbidden origin"})
                return
            body_error = self._body_error()
            if body_error is not None:
                status, message = body_error
                self.close_connection = True
                self._send_json(status, {"error": message})
                return
            try:
                if path == "/api/capture":
                    saved = self.service.capture()
                    self._send_json(201, {"pair_id": saved.pair_id})
                elif path == "/api/reject-last":
                    rejected = self.service.reject_last()
                    self._send_json(200, {"pair_id": rejected.pair_id})
                elif path == "/api/retry-cameras":
                    self.service.retry_cameras()
                    self._send_json(200, {"status": "retry requested"})
                else:
                    job_id = self.service.start_calibration()
                    self._send_json(202, {"job_id": job_id})
            except ServiceError as exc:
                self._send_json(409, {"error": _error_text(exc)})
            except Exception:
                self._send_internal_error()

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlsplit(self.path).path
            if path in _ALL_ROUTES:
                allowed = "GET" if path in _GET_ROUTES else "POST"
                self._method_not_allowed(allowed)
            else:
                self._send_json(404, {"error": "not found"})

        def do_PUT(self) -> None:  # noqa: N802 - stdlib handler API
            self._unsupported_method()

        def do_DELETE(self) -> None:  # noqa: N802 - stdlib handler API
            self._unsupported_method()

        def do_PATCH(self) -> None:  # noqa: N802 - stdlib handler API
            self._unsupported_method()

        def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API
            self._unsupported_method()

        def do_CONNECT(self) -> None:  # noqa: N802 - stdlib handler API
            self._unsupported_method()

        def do_TRACE(self) -> None:  # noqa: N802 - stdlib handler API
            self._unsupported_method()

        def _unsupported_method(self) -> None:
            path = urlsplit(self.path).path
            self.close_connection = True
            if path in _ALL_ROUTES:
                allowed = "GET" if path in _GET_ROUTES else "POST"
                self._method_not_allowed(allowed)
            else:
                self._send_json(404, {"error": "not found"})

        def send_error(
            self,
            code: int,
            message: str | None = None,
            explain: str | None = None,
        ) -> None:
            """Keep stdlib-generated errors JSON-only with the security headers."""
            del message, explain
            path = urlsplit(getattr(self, "path", "")).path
            if code == HTTPStatus.NOT_IMPLEMENTED and path in _ALL_ROUTES:
                allowed = "GET" if path in _GET_ROUTES else "POST"
                self._method_not_allowed(allowed)
                return
            try:
                label = HTTPStatus(code).phrase.lower()
            except ValueError:
                label = "request error"
            self._send_json(code, {"error": label})

        def _origin_allowed(self) -> bool:
            origins = self.headers.get_all("Origin", [])
            if not origins:
                return True
            if len(origins) != 1:
                return False
            origin = origins[0]
            port = int(self.server.server_address[1])
            return origin in {
                f"http://localhost:{port}",
                f"http://127.0.0.1:{port}",
            }

        def _body_error(self) -> tuple[int, str] | None:
            if self.headers.get_all("Transfer-Encoding", []):
                return 400, "request bodies are not supported"
            raw_lengths = self.headers.get_all("Content-Length", [])
            if not raw_lengths:
                return None
            if len(raw_lengths) != 1:
                return 400, "invalid Content-Length"
            raw_length = raw_lengths[0]
            try:
                if not raw_length.isascii() or not raw_length.isdigit():
                    raise ValueError
                length = int(raw_length, 10)
            except (ValueError, OverflowError):
                return 400, "invalid Content-Length"
            if length != 0:
                return 413, "request body is not allowed"
            return None

        def _method_not_allowed(self, allowed: str) -> None:
            self._send_json(405, {"error": "method not allowed"}, {"Allow": allowed})

        def _send_json(
            self,
            status: int,
            payload: Any,
            headers: dict[str, str] | None = None,
        ) -> None:
            try:
                body = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError, OverflowError, UnicodeError):
                self._send_internal_error()
                return
            self._send_bytes(
                status,
                body,
                "application/json; charset=utf-8",
                headers,
            )

        def _send_internal_error(self) -> None:
            # The constant payload cannot recursively fail JSON serialization.
            body = b'{"error":"internal server error"}'
            self._send_bytes(500, body, "application/json; charset=utf-8")

        def _send_bytes(
            self,
            status: int,
            body: bytes,
            content_type: str,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def log_message(self, _format: str, *_args: Any) -> None:
            """Keep tests and the calibration console free of access-log noise."""

    return CalibrationRequestHandler


def _error_text(exc: BaseException) -> str:
    return (str(exc) or exc.__class__.__name__).replace("\r", " ").replace("\n", " ")[:500]
