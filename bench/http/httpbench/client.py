"""Streaming HTTP client for OpenAI-compatible servers (stdlib http.client)."""

from __future__ import annotations

import http.client
import json
import math
import threading
import time
import urllib.parse
from collections.abc import Mapping
from typing import Any, Callable

from .metrics import Chunk, StreamTrace
from .sse import SSEParser


_HEADER_NAME_CHARS = frozenset("!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def validate_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Validate and copy HTTP field names and values before handing them to ``http.client``."""
    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        raise TypeError("headers must be a mapping")
    out: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(name, str) or not name or any(c not in _HEADER_NAME_CHARS for c in name):
            raise ValueError(f"invalid HTTP header name: {name!r}")
        if not isinstance(value, str) or any(ord(c) < 32 or ord(c) > 255 for c in value):
            raise ValueError(f"invalid HTTP header value for {name!r}")
        out[name] = value
    return out


def parse_header(spec: str) -> tuple[str, str]:
    """Parse one CLI-style ``K=V`` header specification."""
    if not isinstance(spec, str) or "=" not in spec:
        raise ValueError("header must use K=V")
    name, value = spec.split("=", 1)
    checked = validate_headers({name: value})
    return next(iter(checked.items()))


class Target:
    """A server under test.

    ``base_url`` may be the server root (``http://127.0.0.1:8080``) or an
    OpenAI prefix (``http://127.0.0.1:8080/v1``, ``http://host/api/v1``). The
    OpenAI routes hang off the prefix (``/v1`` is added to a bare root); the
    llama-server extras (``/props``, ``/tokenize``, ``/completion``,
    ``/health``) hang off the root.
    """

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 900.0,
                 headers: Mapping[str, str] | None = None):
        u = urllib.parse.urlsplit(base_url.rstrip("/"))
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError(f"base URL must be http(s)://host[:port][/prefix], got {base_url!r}")
        self.scheme = u.scheme
        self.host = u.hostname
        self.port = u.port or (443 if u.scheme == "https" else 80)
        path = u.path.rstrip("/")
        if path.endswith("/v1"):
            self.oai_prefix = path
            self.root_prefix = path[: -len("/v1")]
        elif path:
            self.oai_prefix = path
            self.root_prefix = path
        else:
            self.oai_prefix = "/v1"
            self.root_prefix = ""
        self.api_key = api_key
        self.timeout = timeout
        self.headers = validate_headers(headers)

    @property
    def display(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}{self.oai_prefix}"

    def oai(self, route: str) -> str:
        return self.oai_prefix + route

    def root(self, route: str) -> str:
        return self.root_prefix + route

    def _conn(self, timeout: float | None = None) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout if timeout is None else timeout)

    def _headers(self, body: bool, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
        h = {"Accept": "application/json, text/event-stream"}
        if body:
            h["Content-Type"] = "application/json"
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        for name, value in self.headers.items():
            h = {key: old for key, old in h.items() if key.lower() != name.lower()}
            h[name] = value
        for name, value in validate_headers(overrides).items():
            h = {key: old for key, old in h.items() if key.lower() != name.lower()}
            h[name] = value
        return h

    def request_json(self, method: str, path: str, body: Any = None, timeout: float = 10.0,
                     headers: Mapping[str, str] | None = None) -> tuple[int | None, Any]:
        """Small JSON request. Returns (status, parsed body or text); (None, error) on transport failure."""
        conn = self._conn(timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            conn.request(method, path, body=data, headers=self._headers(data is not None, headers))
            resp = conn.getresponse()
            raw = resp.read()
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, raw.decode("utf-8", errors="replace")[:2000]
        except (OSError, http.client.HTTPException) as e:
            return None, f"{type(e).__name__}: {e}"
        finally:
            conn.close()

    def tokenize(self, text: str) -> int | None:
        """Token count via llama-server's /tokenize; None when the server has no such route."""
        status, body = self.request_json("POST", self.root("/tokenize"), {"content": text}, timeout=120.0)
        if status == 200 and isinstance(body, dict) and isinstance(body.get("tokens"), list):
            return len(body["tokens"])
        return None

    # ------------------------------------------------------------------ streaming

    def stream(self, path: str, body: dict, api: str = "chat", headers: Mapping[str, str] | None = None,
               timeout: float | None = None, on_sent: Callable[[], None] | None = None,
               wall_timeout: float | None = None) -> StreamTrace:
        """POST a streaming request and record chunk arrival times."""
        trace = StreamTrace()
        data = json.dumps(body).encode()
        conn = self._conn(timeout)
        t0 = time.perf_counter()
        wall_timer: threading.Timer | None = None
        response_socket: list[Any] = [None]
        expired = threading.Event()
        resp = None

        def abort() -> None:
            expired.set()
            # Shutdown the socket before closing any buffered reader. Closing
            # HTTPResponse from this thread can block on readline's lock while
            # a peer trickles bytes without a newline, defeating the deadline.
            sock = response_socket[0] or getattr(conn, "sock", None)
            if sock is not None:
                try:
                    sock.shutdown(2)
                except OSError:
                    pass

        try:
            if wall_timeout is not None:
                if not math.isfinite(wall_timeout) or wall_timeout <= 0:
                    raise ValueError("wall_timeout must be finite and positive")
                conn.timeout = min(conn.timeout, wall_timeout) if conn.timeout is not None else wall_timeout
                wall_timer = threading.Timer(wall_timeout, abort)
                wall_timer.daemon = True
                wall_timer.start()
                conn.connect()
                # Retain the socket before getresponse(): HTTP/1.0 may detach
                # it from the connection while the response still owns it.
                response_socket[0] = conn.sock
                if expired.is_set():
                    raise TimeoutError("request wall deadline exceeded")
            conn.request("POST", path, body=data, headers=self._headers(True, headers))
            if on_sent is not None:
                on_sent()
            resp = conn.getresponse()
            trace.status = resp.status
            if resp.status != 200:
                raw = resp.read(65536).decode("utf-8", errors="replace")
                trace.error = f"HTTP {resp.status}: {raw[:500]}"
                trace.end_t = time.perf_counter() - t0
                return trace
            parser = SSEParser()
            ctype = resp.getheader("Content-Type", "") or ""
            if "event-stream" not in ctype and "ndjson" not in ctype:
                # Server ignored stream:true and answered with one JSON body.
                raw = resp.read()
                trace.end_t = time.perf_counter() - t0
                self._absorb_plain(trace, raw, api, trace.end_t)
                return trace
            while True:
                line = resp.readline()
                now = time.perf_counter() - t0
                if not line:
                    events = parser.flush()
                else:
                    events = parser.feed(line)
                for ev in events:
                    if ev.is_done:
                        trace.done = True
                        continue
                    self._absorb_event(trace, ev.data, api, now)
                if not line:
                    break
            trace.end_t = time.perf_counter() - t0
            if not trace.done and trace.error is None:
                trace.error = "stream ended without [DONE] / final event"
        except (OSError, http.client.HTTPException, AttributeError) as e:
            trace.error = f"{type(e).__name__}: {e}"
            trace.end_t = time.perf_counter() - t0
        finally:
            if wall_timer is not None:
                wall_timer.cancel()
                wall_timer.join()
            if resp is not None:
                resp.close()
            conn.close()
            if expired.is_set():
                trace.error = "request wall deadline exceeded"
                trace.end_t = time.perf_counter() - t0
        return trace

    @staticmethod
    def _absorb_event(trace: StreamTrace, data: str, api: str, now: float) -> None:
        try:
            obj = json.loads(data)
        except ValueError:
            return
        if not isinstance(obj, dict):
            return
        if "error" in obj and obj["error"]:
            err = obj["error"]
            trace.error = "stream error: " + (json.dumps(err)[:500] if not isinstance(err, str) else err[:500])
            return
        if isinstance(obj.get("usage"), dict):
            trace.usage = obj["usage"]
        Target._absorb_meta(trace, obj)
        if obj.get("model") and not trace.model:
            trace.model = obj["model"]
        if api == "completion":
            text = obj.get("content") or ""
            toks = obj.get("tokens")
            n = len(toks) if isinstance(toks, list) and toks else None
            if text or n:
                trace.chunks.append(Chunk(t=now, text=text, n_tokens=n))
            if obj.get("stop"):
                trace.done = True
                for k in ("tokens_predicted", "tokens_evaluated", "tokens_cached", "stop_type", "truncated"):
                    if k in obj:
                        trace.extra_final[k] = obj[k]
                trace.finish_reason = obj.get("stop_type") or trace.finish_reason
            return
        for ch in obj.get("choices") or []:
            delta = ch.get("delta") or {}
            text = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            tool = bool(delta.get("tool_calls"))
            if text or reasoning or tool:
                trace.chunks.append(Chunk(t=now, text=text if isinstance(text, str) else "",
                                          reasoning=reasoning if isinstance(reasoning, str) else "", tool=tool))
            if ch.get("finish_reason"):
                trace.finish_reason = ch["finish_reason"]

    @staticmethod
    def _absorb_plain(trace: StreamTrace, raw: bytes, api: str, now: float) -> None:
        try:
            obj = json.loads(raw)
        except ValueError:
            trace.error = "non-JSON, non-SSE response: " + raw[:200].decode("utf-8", errors="replace")
            return
        if isinstance(obj, dict) and obj.get("error"):
            trace.error = "error: " + json.dumps(obj["error"])[:500]
            return
        trace.usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
        Target._absorb_meta(trace, obj)
        reasoning = ""
        if api == "completion":
            text = obj.get("content") or ""
        else:
            msg = ((obj.get("choices") or [{}])[0] or {}).get("message") or {}
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            trace.finish_reason = ((obj.get("choices") or [{}])[0] or {}).get("finish_reason")
        if text or reasoning:
            trace.chunks.append(Chunk(t=now, text=text if isinstance(text, str) else "",
                                      reasoning=reasoning if isinstance(reasoning, str) else ""))
        trace.done = True
        trace.error = None
        # A non-streamed answer has no token timing; mark it so the caller can tell.
        trace.extra_final["not_streamed"] = True

    @staticmethod
    def _absorb_meta(trace: StreamTrace, obj: dict[str, Any]) -> None:
        """Collect additive Sonder metadata while retaining legacy top-level fields."""
        usage = obj.get("usage")
        nested = usage.get("sonder") if isinstance(usage, dict) else None
        nested_timings = nested.get("timings") if isinstance(nested, dict) else None
        top_timings = obj.get("timings")
        if isinstance(top_timings, dict) or isinstance(nested_timings, dict):
            merged = dict(trace.timings or {})
            if isinstance(nested_timings, dict):
                merged.update(nested_timings)
            if isinstance(top_timings, dict):
                merged.update(top_timings)
            trace.timings = merged
        sonder = obj.get("sonder")
        warnings = None
        if isinstance(sonder, dict):
            warnings = sonder.get("warnings")
        if warnings is None and isinstance(nested, dict):
            warnings = nested.get("warnings")
        if warnings:
            previous = trace.extra_final.get("sonder_warnings")
            if previous is None:
                trace.extra_final["sonder_warnings"] = warnings
            elif isinstance(previous, list):
                additions = warnings if isinstance(warnings, list) else [warnings]
                trace.extra_final["sonder_warnings"] = previous + [x for x in additions if x not in previous]


def trace_text(trace: StreamTrace) -> tuple[str, str]:
    """(content, reasoning) concatenated from the trace."""
    return "".join(c.text for c in trace.chunks), "".join(c.reasoning for c in trace.chunks)
