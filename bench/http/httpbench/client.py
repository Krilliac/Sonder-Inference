"""Streaming HTTP client for OpenAI-compatible servers (stdlib http.client)."""

from __future__ import annotations

import http.client
import json
import time
import urllib.parse
from typing import Any

from .metrics import Chunk, StreamTrace
from .sse import SSEParser


class Target:
    """A server under test.

    ``base_url`` may be the server root (``http://127.0.0.1:8080``) or an
    OpenAI prefix (``http://127.0.0.1:8080/v1``, ``http://host/api/v1``). The
    OpenAI routes hang off the prefix (``/v1`` is added to a bare root); the
    llama-server extras (``/props``, ``/tokenize``, ``/completion``,
    ``/health``) hang off the root.
    """

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 900.0):
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

    @property
    def display(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}{self.oai_prefix}"

    def oai(self, route: str) -> str:
        return self.oai_prefix + route

    def root(self, route: str) -> str:
        return self.root_prefix + route

    def _conn(self, timeout: float | None = None) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout or self.timeout)

    def _headers(self, body: bool) -> dict[str, str]:
        h = {"Accept": "application/json, text/event-stream"}
        if body:
            h["Content-Type"] = "application/json"
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def request_json(self, method: str, path: str, body: Any = None, timeout: float = 10.0) -> tuple[int | None, Any]:
        """Small JSON request. Returns (status, parsed body or text); (None, error) on transport failure."""
        conn = self._conn(timeout)
        try:
            data = json.dumps(body).encode() if body is not None else None
            conn.request(method, path, body=data, headers=self._headers(data is not None))
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

    def stream(self, path: str, body: dict, api: str = "chat") -> StreamTrace:
        """POST a streaming request and record chunk arrival times."""
        trace = StreamTrace()
        data = json.dumps(body).encode()
        conn = self._conn()
        t0 = time.perf_counter()
        try:
            conn.request("POST", path, body=data, headers=self._headers(True))
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
        except (OSError, http.client.HTTPException) as e:
            trace.error = f"{type(e).__name__}: {e}"
            trace.end_t = time.perf_counter() - t0
        finally:
            conn.close()
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
        if isinstance(obj.get("timings"), dict):
            trace.timings = obj["timings"]
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
        trace.timings = obj.get("timings") if isinstance(obj.get("timings"), dict) else None
        if api == "completion":
            text = obj.get("content") or ""
        else:
            msg = ((obj.get("choices") or [{}])[0] or {}).get("message") or {}
            text = msg.get("content") or ""
            trace.finish_reason = ((obj.get("choices") or [{}])[0] or {}).get("finish_reason")
        if text:
            trace.chunks.append(Chunk(t=now, text=text))
        trace.done = True
        trace.error = None
        # A non-streamed answer has no token timing; mark it so the caller can tell.
        trace.extra_final["not_streamed"] = True


def trace_text(trace: StreamTrace) -> tuple[str, str]:
    """(content, reasoning) concatenated from the trace."""
    return "".join(c.text for c in trace.chunks), "".join(c.reasoning for c in trace.chunks)
