"""Focused transport contract tests for per-request controls and telemetry."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fake_server import FakeServer
from httpbench.client import Target, parse_header, trace_text, validate_headers
from httpbench.metrics import StreamTrace, derive


def test_header_validation_and_parse_header():
    assert parse_header("X-Run=abc") == ("X-Run", "abc")
    assert validate_headers({"X-Run": "abc"}) == {"X-Run": "abc"}
    with pytest.raises(ValueError):
        parse_header("missing-equals")
    for bad in ("X\nRun=value", "X-Run=a\rb", "X-Run=a\x00b"):
        with pytest.raises(ValueError):
            parse_header(bad) if "=" not in bad else validate_headers({bad.split("=", 1)[0]: bad.split("=", 1)[1]})
    with pytest.raises(ValueError):
        validate_headers({"X-Run": "caf\N{euro sign}"})


def test_target_merges_default_global_and_request_headers():
    with FakeServer() as srv:
        target = Target(srv.url, api_key="secret", headers={"X-Global": "global", "Authorization": "global-auth"})
        status, _ = target.request_json("POST", "/tokenize", {"content": "hello"}, headers={"x-global": "request", "X-Request": "yes"})
        assert status == 200
        req = srv.state.requests[-1]
        assert req["headers"]["x-global"] == "request"
        assert req["headers"]["X-Request"] == "yes"
        assert req["headers"]["Authorization"] == "global-auth"


def test_stream_on_sent_and_optional_timeout_and_headers():
    sent = []
    with FakeServer() as srv:
        target = Target(srv.url, headers={"X-Global": "yes"}, timeout=2)
        body = {"model": "fake-model", "messages": [{"role": "user", "content": "hello"}], "stream": True}
        trace = target.stream(target.oai("/chat/completions"), body, headers={"X-Request": "yes"},
                              timeout=3, on_sent=lambda: sent.append(True))
        assert trace.status == 200 and trace.done and sent == [True]
        req = srv.state.requests[-1]
        assert req["headers"]["X-Global"] == "yes" and req["headers"]["X-Request"] == "yes"


def test_warnings_nested_timings_and_plain_reasoning_are_preserved():
    trace = StreamTrace(status=200, done=True, end_t=1.0)
    target = Target("http://127.0.0.1:1")
    payload = {"usage": {"prompt_tokens": 12, "sonder": {"timings": {"cache_n": 9, "prompt_n": 3}}},
               "sonder": {"warnings": ["thinking pinned"]},
               "timings": {"prompt_per_second": 100.0},
               "choices": [{"message": {"content": "answer", "reasoning_content": "reason"}, "finish_reason": "stop"}]}
    target._absorb_plain(trace, json.dumps(payload).encode(), "chat", 0.5)
    assert trace.extra_final["sonder_warnings"] == ["thinking pinned"]
    target._absorb_event(trace, json.dumps({"sonder": {"warnings": ["queue pressure"]}}), "chat", 0.6)
    assert trace.extra_final["sonder_warnings"] == ["thinking pinned", "queue pressure"]
    assert trace.timings == {"prompt_per_second": 100.0, "cache_n": 9, "prompt_n": 3}
    assert trace_text(trace) == ("answer", "reason")
    derived = derive(trace)
    assert derived["cache_n"] == 9 and derived["prompt_n"] == 3


class _SlowHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    mode = "delay"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.mode == "byte-trickle":
            try:
                for _ in range(35):
                    self.wfile.write(b":")
                    self.wfile.flush()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionError, OSError):
                pass
            return
        if self.mode == "delay":
            time.sleep(1.0)
            return
        self.wfile.write(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n")
        self.wfile.flush()
        time.sleep(1.0)


def test_stream_wall_timeout_aborts_delayed_response_and_cleans_transport():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.perf_counter()
        trace = Target(f"http://127.0.0.1:{server.server_port}").stream(
            "/v1/chat/completions", {"messages": [], "stream": True}, wall_timeout=0.1)
        elapsed = time.perf_counter() - started
        assert elapsed < 0.7 and trace.error
    finally:
        server.shutdown()
        server.server_close()


def test_stream_wall_timeout_aborts_trickling_response():
    _SlowHandler.mode = "trickle"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.perf_counter()
        trace = Target(f"http://127.0.0.1:{server.server_port}").stream(
            "/v1/chat/completions", {"messages": [], "stream": True}, wall_timeout=0.1)
        elapsed = time.perf_counter() - started
        assert elapsed < 0.7 and trace.error
    finally:
        server.shutdown()
        server.server_close()
        _SlowHandler.mode = "delay"


def test_wall_timeout_cannot_be_extended_by_bytes_without_newlines():
    _SlowHandler.mode = "byte-trickle"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.perf_counter()
        trace = Target(f"http://127.0.0.1:{server.server_port}").stream(
            "/v1/chat/completions", {"messages": [], "stream": True}, wall_timeout=0.1)
        assert time.perf_counter() - started < 0.5
        assert trace.error and not derive(trace)["ok"]
    finally:
        server.shutdown()
        server.server_close()
        _SlowHandler.mode = "delay"
