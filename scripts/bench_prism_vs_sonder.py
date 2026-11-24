#!/usr/bin/env python3
"""PrismML llama-server vs Sonder-Inference Bonsai Head-to-Head Benchmark & Quality Suite.

Supports two execution modes (stdlib-only Python 3.10+, Windows & Linux):
  1. Attach mode:
       python scripts/bench_prism_vs_sonder.py \\
         --target prism=http://127.0.0.1:18080 \\
         --target sonder=http://127.0.0.1:18081 \\
         --model Bonsai-2-9B-MTP --disable-thinking \\
         --output-json out/h2h.json --output-md out/h2h.md

  2. Managed sequential mode (launches ONE server at a time, warms up, samples VRAM & RSS,
     runs all selected suites, and terminates the entire process tree before starting the
     next target so two 27B models never collide on 16 GiB VRAM):
       python scripts/bench_prism_vs_sonder.py --config bench_targets.json \\
         --output-json out/h2h.json --output-md out/h2h.md
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import hashlib
import http.client
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sonder.inference.bonsai_h2h/1"
ALL_SUITES = (
    "prefill",
    "prefix",
    "decode",
    "concurrency",
    "keepalive",
    "endpoints",
    "quality",
)
QUALITY_CATEGORIES = ("math", "python_code", "json_schema", "logic")
IS_WINDOWS = os.name == "nt"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = REPO_ROOT / "bench" / "corpus" / "bonsai_head2head.json"


# ---------------------------------------------------------------------------
# Statistics & Math Helpers
# ---------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """Convert int/float (excluding bool and non-finite floats) to float, else None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        fval = float(value)
        return fval if math.isfinite(fval) else None
    return None


def percentile(values: Iterable[float | None], p: float) -> float | None:
    """Linear-interpolated percentile (p in [0, 100]); None on empty input."""
    xs = sorted(float(v) for v in values if _num(v) is not None)  # type: ignore[arg-type]
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    p_clamped = min(max(float(p), 0.0), 100.0)
    rank = (len(xs) - 1) * (p_clamped / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def median_of(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if _num(v) is not None]  # type: ignore[arg-type]
    return float(statistics.median(xs)) if xs else None


def mean_of(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if _num(v) is not None]  # type: ignore[arg-type]
    return float(statistics.fmean(xs)) if xs else None


def summarize_metric(values: Iterable[float | None]) -> dict[str, Any]:
    xs = [float(v) for v in values if _num(v) is not None]  # type: ignore[arg-type]
    if not xs:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(xs),
        "median": float(statistics.median(xs)),
        "mean": float(statistics.fmean(xs)),
        "p50": percentile(xs, 50.0),
        "p95": percentile(xs, 95.0),
        "min": min(xs),
        "max": max(xs),
    }


def fnv1a_64_hex(data: bytes) -> str:
    h = 1469598103934665603
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


# ---------------------------------------------------------------------------
# Incremental SSE Parser
# ---------------------------------------------------------------------------


@dataclass
class SSEEvent:
    data: str
    event: str = "message"
    id: str | None = None

    @property
    def is_done(self) -> bool:
        return self.data.strip() == "[DONE]"


class SSEParser:
    """WHATWG Server-Sent Events incremental byte parser (LF, CR, and CRLF safe)."""

    def __init__(self) -> None:
        self._buf = b""
        self._data: list[str] = []
        self._event = ""
        self._id: str | None = None
        self._pending_cr = False

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        if self._pending_cr and chunk.startswith(b"\n"):
            chunk = chunk[1:]
        self._pending_cr = False
        self._buf += chunk
        events: list[SSEEvent] = []
        while True:
            idx_n = self._buf.find(b"\n")
            idx_r = self._buf.find(b"\r")
            if idx_n < 0 and idx_r < 0:
                break
            if idx_r >= 0 and (idx_n < 0 or idx_r < idx_n):
                line = self._buf[:idx_r]
                rest = self._buf[idx_r + 1 :]
                if rest.startswith(b"\n"):
                    rest = rest[1:]
                elif not rest:
                    self._pending_cr = True
            else:
                line = self._buf[:idx_n]
                rest = self._buf[idx_n + 1 :]
            self._buf = rest
            ev = self._consume_line(line.decode("utf-8", errors="replace"))
            if ev is not None:
                events.append(ev)
        return events

    def flush(self) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        if self._buf:
            ev = self._consume_line(self._buf.decode("utf-8", errors="replace"))
            self._buf = b""
            if ev is not None:
                events.append(ev)
        ev = self._consume_line("")
        if ev is not None:
            events.append(ev)
        return events

    def _consume_line(self, line: str) -> SSEEvent | None:
        if line == "":
            if not self._data:
                self._event = ""
                return None
            ev = SSEEvent(
                data="\n".join(self._data),
                event=self._event or "message",
                id=self._id,
            )
            self._data = []
            self._event = ""
            return ev
        if line.startswith(":"):
            return None
        field, sep, value = line.partition(":")
        if not sep:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value
        elif field == "id":
            self._id = value
        return None


# ---------------------------------------------------------------------------
# Host GPU VRAM & Process Tree RSS Sampling (Windows ctypes + Linux /proc)
# ---------------------------------------------------------------------------

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class _PROCESSENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * 260),
        ]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _k32.Process32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32)]
    _k32.Process32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32)]

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _PROCESS_QUERY_INFORMATION = 0x0400
    _PROCESS_VM_READ = 0x0010
    _TH32CS_SNAPPROCESS = 0x00000002
    _INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
    _STILL_ACTIVE = 259


def _descendant_pids(root_pid: int) -> list[int]:
    """Return [root_pid, ...descendants] across Windows or Linux."""
    if root_pid <= 0:
        return []
    children_map: dict[int, list[int]] = {}
    if IS_WINDOWS:
        snap = _k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        if snap and snap != _INVALID_HANDLE_VALUE:
            try:
                entry = _PROCESSENTRY32()
                entry.dwSize = ctypes.sizeof(_PROCESSENTRY32)
                if _k32.Process32First(snap, ctypes.byref(entry)):
                    while True:
                        pid = int(entry.th32ProcessID)
                        ppid = int(entry.th32ParentProcessID)
                        children_map.setdefault(ppid, []).append(pid)
                        if not _k32.Process32Next(snap, ctypes.byref(entry)):
                            break
            finally:
                _k32.CloseHandle(snap)
    else:
        proc_root = Path("/proc")
        if proc_root.is_dir():
            for child in proc_root.iterdir():
                if not child.name.isdigit():
                    continue
                try:
                    stat_text = (child / "stat").read_text(encoding="ascii", errors="replace")
                    rparen = stat_text.rfind(")")
                    if rparen > 0:
                        fields = stat_text[rparen + 2 :].split()
                        if len(fields) >= 2:
                            ppid = int(fields[1])
                            children_map.setdefault(ppid, []).append(int(child.name))
                except (OSError, ValueError):
                    continue

    visited: list[int] = []
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        visited.append(cur)
        for ch in children_map.get(cur, ()):
            if ch not in seen:
                stack.append(ch)
    return visited


def _single_pid_rss_bytes(pid: int) -> int | None:
    if pid <= 0:
        return None
    if IS_WINDOWS:
        handle = _k32.OpenProcess(
            _PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, pid
        )
        if not handle:
            handle = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            code = wintypes.DWORD(0)
            if not _k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != _STILL_ACTIVE:
                return None
            counters = _PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS_EX)
            if _k32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
            return None
        finally:
            _k32.CloseHandle(handle)

    try:
        status_text = Path(f"/proc/{pid}/status").read_text(encoding="ascii", errors="replace")
        for line in status_text.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def sample_process_rss_mib(pid: int | None) -> float | None:
    """Sum RSS (WorkingSetSize / VmRSS) in MiB across pid and its descendants, or None."""
    if pid is None or pid <= 0:
        return None
    total_bytes = 0
    any_alive = False
    for p in _descendant_pids(pid):
        rss = _single_pid_rss_bytes(p)
        if rss is not None:
            total_bytes += rss
            any_alive = True
    if not any_alive:
        return None
    return round(total_bytes / (1024.0 * 1024.0), 2)


def sample_gpu_vram_mib() -> float | None:
    """Query used GPU VRAM in MiB via nvidia-smi; gracefully returns None if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            return None
        vals = [float(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
        return round(sum(vals), 1) if vals else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------
# HTTP/1.1 Benchmark Client
# ---------------------------------------------------------------------------

_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def validate_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    if not headers:
        return {}
    out: dict[str, str] = {}
    for k, v in headers.items():
        if not isinstance(k, str) or not _HEADER_NAME_RE.match(k):
            raise ValueError(f"invalid HTTP header name: {k!r}")
        if not isinstance(v, str) or any(ord(c) < 32 or ord(c) > 255 for c in v):
            raise ValueError(f"invalid HTTP header value for {k!r}")
        out[k] = v
    return out


def strip_think_blocks(text: str) -> str:
    """Strip inline <think>...</think> reasoning blocks before quality grading."""
    if "<think>" not in text:
        return text
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Handle unclosed <think> if max_tokens truncated inside reasoning.
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned


class BenchmarkClient:
    """Stdlib HTTP/1.1 client supporting SSE streaming, keep-alive reuse, and endpoint probes."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 300.0,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url.rstrip("/"))
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"target URL must be http(s)://host[:port][/prefix], got {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            self.oai_prefix = path
            self.root_prefix = path[:-3]
        elif path:
            self.oai_prefix = path + "/v1"
            self.root_prefix = path
        else:
            self.oai_prefix = "/v1"
            self.root_prefix = ""
        self.api_key = api_key
        self.extra_headers = validate_headers(headers)
        self.timeout = float(timeout)

    def oai_path(self, route: str) -> str:
        if not route.startswith("/"):
            route = "/" + route
        if route.startswith("/v1/"):
            return self.root_prefix + route
        return self.oai_prefix + route

    def root_path(self, route: str) -> str:
        if not route.startswith("/"):
            route = "/" + route
        return self.root_prefix + route

    def new_connection(self, timeout: float | None = None) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout if timeout is None else float(timeout))

    def _build_headers(
        self,
        *,
        has_body: bool,
        accept: str = "application/json",
        overrides: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        hdrs: dict[str, str] = {"Accept": accept}
        if has_body:
            hdrs["Content-Type"] = "application/json"
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        for k, v in self.extra_headers.items():
            hdrs = {ek: ev for ek, ev in hdrs.items() if ek.lower() != k.lower()}
            hdrs[k] = v
        if overrides:
            for k, v in validate_headers(overrides).items():
                hdrs = {ek: ev for ek, ev in hdrs.items() if ek.lower() != k.lower()}
                hdrs[k] = v
        return hdrs

    def request_json(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int | None, Any, float, dict[str, str]]:
        """Execute one fresh-connection HTTP request. Returns (status, parsed_or_text, latency_ms, resp_headers)."""
        conn = self.new_connection(timeout=timeout)
        t0 = time.perf_counter()
        try:
            payload = json.dumps(body).encode("utf-8") if body is not None else None
            req_headers = self._build_headers(has_body=payload is not None, overrides=headers)
            conn.request(method, path, body=payload, headers=req_headers)
            resp = conn.getresponse()
            raw = resp.read()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            resp_hdrs = {k.lower(): v for k, v in resp.getheaders()}
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else None
            except ValueError:
                parsed = raw.decode("utf-8", errors="replace")[:2000]
            return resp.status, parsed, elapsed_ms, resp_hdrs
        except (OSError, http.client.HTTPException) as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return None, f"{type(exc).__name__}: {exc}", elapsed_ms, {}
        finally:
            conn.close()

    def discover_model(self, configured_model: str | None) -> str:
        """Resolve model ID from explicit CLI/config or GET /v1/models fallback."""
        if configured_model:
            return configured_model
        status, body, _, _ = self.request_json("GET", self.oai_path("/models"), timeout=10.0)
        if status == 200 and isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data and isinstance(data[0], dict):
                mid = data[0].get("id")
                if isinstance(mid, str) and mid:
                    return mid
        return "default"

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        seed: int | None = 42,
        disable_thinking: bool = False,
        response_format: dict[str, Any] | None = None,
        prompt_cache_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        on_ready: threading.Barrier | None = None,
    ) -> dict[str, Any]:
        """Run one SSE streaming POST /v1/chat/completions and derive exact timing & speculation metrics."""
        req_timeout = self.timeout if timeout is None else float(timeout)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if seed is not None:
            body["seed"] = int(seed)
        if disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if response_format is not None:
            body["response_format"] = response_format
        if prompt_cache_key is not None:
            body["prompt_cache_key"] = prompt_cache_key
        if extra_body:
            body.update(extra_body)

        encoded_body = json.dumps(body).encode("utf-8")
        req_headers = self._build_headers(
            has_body=True,
            accept="text/event-stream, application/json",
            overrides={"Connection": "close"},
        )
        path = self.oai_path("/chat/completions")

        if on_ready is not None:
            try:
                on_ready.wait(timeout=15.0)
            except threading.BrokenBarrierError:
                pass

        conn = self.new_connection(timeout=req_timeout)
        t0 = time.perf_counter()
        t_first: float | None = None
        token_chunks = 0
        saw_done = False
        finish_reason: str | None = None
        http_status: int | None = None
        error: str | None = None
        usage: dict[str, Any] | None = None
        timings_merged: dict[str, Any] = {}
        content_parts: list[str] = []
        reasoning_parts: list[str] = []

        def absorb_timings_and_usage(obj: dict[str, Any]) -> None:
            nonlocal usage
            u = obj.get("usage")
            if isinstance(u, dict):
                usage = u
                sonder_nested = u.get("sonder")
                if isinstance(sonder_nested, dict) and isinstance(sonder_nested.get("timings"), dict):
                    timings_merged.update(sonder_nested["timings"])
            top_t = obj.get("timings")
            if isinstance(top_t, dict):
                timings_merged.update(top_t)

        try:
            conn.connect()
            sock = conn.sock

            def arm_deadline() -> float:
                rem = req_timeout - (time.perf_counter() - t0)
                if rem <= 0:
                    raise TimeoutError(f"request exceeded {req_timeout:.1f}s deadline")
                if sock is not None:
                    sock.settimeout(rem)
                return rem

            arm_deadline()
            conn.request("POST", path, body=encoded_body, headers=req_headers)
            arm_deadline()
            resp = conn.getresponse()
            http_status = resp.status
            if resp.status != 200:
                arm_deadline()
                err_raw = resp.read(8192).decode("utf-8", errors="replace")
                error = f"HTTP {resp.status}: {err_raw.strip()[:300]}"
                total_s = time.perf_counter() - t0
                return _finalize_stream_metrics(
                    ok=False,
                    http_status=http_status,
                    error=error,
                    ttft_s=None,
                    total_s=total_s,
                    token_chunks=0,
                    usage=None,
                    raw_timings={},
                    content="",
                    reasoning="",
                    finish_reason=None,
                )

            ctype = (resp.getheader("Content-Type") or "").lower()
            if "event-stream" not in ctype:
                # Graceful fallback if server answered with a plain JSON completion body.
                arm_deadline()
                raw = resp.read()
                t_now = time.perf_counter()
                obj = json.loads(raw.decode("utf-8"))
                if isinstance(obj, dict):
                    absorb_timings_and_usage(obj)
                    choices = obj.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        msg = choices[0].get("message") or {}
                        c_txt = msg.get("content") or ""
                        r_txt = msg.get("reasoning_content") or msg.get("reasoning") or ""
                        if c_txt:
                            content_parts.append(str(c_txt))
                        if r_txt:
                            reasoning_parts.append(str(r_txt))
                        if c_txt or r_txt:
                            t_first = t_now
                            token_chunks = 1
                        finish_reason = choices[0].get("finish_reason")
                    saw_done = True
            else:
                parser = SSEParser()
                while True:
                    arm_deadline()
                    chunk = resp.read1(16384) if hasattr(resp, "read1") else resp.read(4096)
                    now = time.perf_counter()
                    events = parser.feed(chunk) if chunk else parser.flush()
                    for ev in events:
                        if ev.is_done:
                            saw_done = True
                            continue
                        try:
                            event_obj = json.loads(ev.data)
                        except ValueError:
                            error = f"malformed SSE JSON: {ev.data[:120]}"
                            break
                        if not isinstance(event_obj, dict):
                            continue
                        if event_obj.get("error"):
                            error = f"stream error: {json.dumps(event_obj['error'])[:300]}"
                            break
                        absorb_timings_and_usage(event_obj)
                        for ch in event_obj.get("choices") or []:
                            if not isinstance(ch, dict):
                                continue
                            delta = ch.get("delta") or {}
                            if isinstance(delta, dict):
                                c_piece = delta.get("content") or ""
                                r_piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                                if (isinstance(c_piece, str) and c_piece) or (
                                    isinstance(r_piece, str) and r_piece
                                ):
                                    if t_first is None:
                                        t_first = now
                                    token_chunks += 1
                                    if isinstance(c_piece, str) and c_piece:
                                        content_parts.append(c_piece)
                                    if isinstance(r_piece, str) and r_piece:
                                        reasoning_parts.append(r_piece)
                            if ch.get("finish_reason"):
                                finish_reason = ch["finish_reason"]
                    if error is not None or saw_done or not chunk:
                        break
        except (OSError, TimeoutError, http.client.HTTPException, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()

        t_end = time.perf_counter()
        total_s = max(t_end - t0, 0.0)
        ttft_s = (t_first - t0) if t_first is not None else None
        if error is None and not saw_done:
            error = "truncated SSE stream (missing [DONE])"
        ok = error is None and http_status == 200 and saw_done
        return _finalize_stream_metrics(
            ok=ok,
            http_status=http_status,
            error=error,
            ttft_s=ttft_s,
            total_s=total_s,
            token_chunks=token_chunks,
            usage=usage,
            raw_timings=timings_merged,
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
        )


def _finalize_stream_metrics(
    *,
    ok: bool,
    http_status: int | None,
    error: str | None,
    ttft_s: float | None,
    total_s: float,
    token_chunks: int,
    usage: dict[str, Any] | None,
    raw_timings: dict[str, Any],
    content: str,
    reasoning: str,
    finish_reason: str | None,
) -> dict[str, Any]:
    # Token count resolution: usage.completion_tokens -> timings.predicted_n -> SSE chunk count.
    tokens = token_chunks
    token_source = "chunks"
    if usage and _num(usage.get("completion_tokens")) is not None:
        tokens = int(usage["completion_tokens"])
        token_source = "usage"
    elif _num(raw_timings.get("predicted_n")) is not None:
        tokens = int(raw_timings["predicted_n"])
        token_source = "timings"

    # Cached prompt tokens: timings.cache_n -> usage.prompt_tokens_details.cached_tokens.
    cache_n_val = _num(raw_timings.get("cache_n"))
    if cache_n_val is None and usage:
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and _num(details.get("cached_tokens")) is not None:
            cache_n_val = _num(details.get("cached_tokens"))
    cache_n = int(cache_n_val) if cache_n_val is not None else None

    prompt_candidates: list[int] = []
    if usage and _num(usage.get("prompt_tokens")) is not None:
        prompt_candidates.append(int(usage["prompt_tokens"]))
    if _num(raw_timings.get("prompt_n")) is not None:
        prompt_candidates.append(int(raw_timings["prompt_n"]) + (cache_n or 0))
    prompt_tokens = max(prompt_candidates) if prompt_candidates else None

    draft_n_f = _num(raw_timings.get("draft_n"))
    draft_acc_f = _num(raw_timings.get("draft_n_accepted"))
    draft_n = int(draft_n_f) if draft_n_f is not None else None
    draft_n_accepted = int(draft_acc_f) if draft_acc_f is not None else None
    acceptance_ratio = (
        (float(draft_n_accepted) / float(draft_n))
        if (draft_n is not None and draft_n > 0 and draft_n_accepted is not None)
        else None
    )

    ttft_ms = (ttft_s * 1000.0) if ttft_s is not None else None
    total_ms = total_s * 1000.0

    wall_decode_tps: float | None = None
    if ttft_s is not None and tokens > 1 and total_s > ttft_s:
        wall_decode_tps = (tokens - 1) / (total_s - ttft_s)

    total_wall_tps: float | None = None
    if tokens > 0 and total_s > 0.0:
        total_wall_tps = tokens / total_s

    timings_out = {
        "prompt_ms": _num(raw_timings.get("prompt_ms")),
        "prompt_per_second": _num(raw_timings.get("prompt_per_second")),
        "predicted_ms": _num(raw_timings.get("predicted_ms")),
        "predicted_per_second": _num(raw_timings.get("predicted_per_second")),
        "cache_n": cache_n,
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "acceptance_ratio": acceptance_ratio,
    }

    clean_content = strip_think_blocks(content)
    return {
        "ok": ok,
        "http_status": http_status,
        "error": error,
        "finish_reason": finish_reason,
        "ttft_s": ttft_s,
        "ttft_ms": ttft_ms,
        "total_s": total_s,
        "total_ms": total_ms,
        "tokens": tokens,
        "token_source": token_source,
        "prompt_tokens": prompt_tokens,
        "wall_decode_tps": wall_decode_tps,
        "total_wall_tps": total_wall_tps,
        "timings": timings_out,
        "acceptance_ratio": acceptance_ratio,
        "content": content,
        "clean_content": clean_content,
        "reasoning_content": reasoning,
        "output_sha256": hashlib.sha256(clean_content.encode("utf-8")).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Quality Probe Graders (Math, Python Subprocess, JSON Schema, Logic Regex)
# ---------------------------------------------------------------------------


def grade_math(text: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Extract numeric answer from model output and check against expected within tolerance."""
    cleaned = strip_think_blocks(text).strip()
    expected = float(spec["expected"])
    tolerance = float(spec.get("tolerance", 1e-6))

    patterns = [
        r"FINAL_ANSWER\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"(?:final\s+answer|answer|total|sum|result)\s*(?:is|=|:)\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"([-+]?\d[\d,]*(?:\.\d+)?)",
    ]
    extracted_str: str | None = None
    for pat in patterns:
        matches = re.findall(pat, cleaned, flags=re.IGNORECASE)
        if matches:
            extracted_str = matches[-1]
            break

    if extracted_str is None:
        return {"passed": False, "extracted": None, "detail": "no numeric answer found"}

    try:
        numeric_val = float(extracted_str.replace(",", ""))
    except ValueError:
        return {"passed": False, "extracted": extracted_str, "detail": f"unparseable number {extracted_str!r}"}

    passed = math.isclose(numeric_val, expected, rel_tol=0.0, abs_tol=tolerance)
    detail = f"got {numeric_val:g} == {expected:g}" if passed else f"expected {expected:g}, got {numeric_val:g}"
    return {"passed": passed, "extracted": numeric_val, "detail": detail}


def extract_python_code(text: str, entry_point: str | None = None) -> str:
    """Extract Python code from markdown code fences or raw completion text."""
    cleaned = strip_think_blocks(text).strip()
    blocks = re.findall(r"```(?:python3?|py)?\s*\r?\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        if entry_point:
            for blk in blocks:
                if entry_point in blk:
                    return blk.strip()
        return "\n\n".join(b.strip() for b in blocks if b.strip())
    return cleaned


def grade_python_code(text: str, spec: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
    """Execute extracted Python candidate + unit test assertions in an isolated subprocess."""
    entry_point = spec.get("entry_point")
    candidate_code = extract_python_code(text, entry_point=entry_point)
    if not candidate_code:
        return {"passed": False, "extracted": "", "detail": "empty Python candidate"}

    test_code = str(spec.get("test_code", "")).strip()
    if not test_code:
        return {"passed": False, "extracted": candidate_code, "detail": "probe missing test_code"}

    effective_timeout = float(timeout_s if timeout_s is not None else spec.get("timeout_s", 3.0))
    combined_script = f"{candidate_code}\n\n# --- Harness Unit Assertions ---\n{test_code}\n"

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    with tempfile.TemporaryDirectory(prefix="bonsai_pygrade_") as tmp_cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", "-c", combined_script],
                input="",
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=tmp_cwd,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "extracted": candidate_code,
                "detail": f"timeout after {effective_timeout:.1f}s",
            }
        except OSError as exc:
            return {
                "passed": False,
                "extracted": candidate_code,
                "detail": f"subprocess launch failed: {exc}",
            }

    if proc.returncode == 0:
        return {"passed": True, "extracted": candidate_code, "detail": "all unit assertions passed"}

    stderr_lines = [ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip()]
    tail = stderr_lines[-1] if stderr_lines else f"exit code {proc.returncode}"
    return {
        "passed": False,
        "extracted": candidate_code,
        "detail": f"test failed: {tail[:240]}",
    }


def _matches_json_type(value: Any, type_spec: str) -> bool:
    t = type_spec.strip().lower()
    if t in ("str", "string"):
        return isinstance(value, str)
    if t in ("int", "integer"):
        return isinstance(value, int) and not isinstance(value, bool)
    if t in ("float", "number"):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if t in ("bool", "boolean"):
        return isinstance(value, bool)
    if t in ("dict", "object"):
        return isinstance(value, dict)
    if t in ("list", "array"):
        return isinstance(value, list)
    if t == "list[str]":
        return isinstance(value, list) and all(isinstance(x, str) for x in value)
    if t == "list[int]":
        return isinstance(value, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    if t == "list[number]":
        return isinstance(value, list) and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in value
        )
    raise ValueError(f"unsupported JSON schema type spec: {type_spec!r}")


def grade_json_schema(text: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Validate JSON object parseability, required keys, strict value types, and expected values."""
    cleaned = strip_think_blocks(text).strip()
    raw_candidate = cleaned
    if not raw_candidate:
        return {"passed": False, "extracted": None, "detail": "empty response"}

    obj: Any = None
    try:
        obj = json.loads(raw_candidate)
    except ValueError:
        fenced = re.findall(r"```(?:json)?\s*\r?\n(.*?)```", raw_candidate, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                obj = json.loads(fenced[0].strip())
            except ValueError as exc:
                return {"passed": False, "extracted": None, "detail": f"invalid JSON: {exc}"}
        else:
            return {"passed": False, "extracted": None, "detail": "invalid JSON payload"}

    if not isinstance(obj, dict):
        return {"passed": False, "extracted": obj, "detail": f"expected JSON object, got {type(obj).__name__}"}

    required_keys: dict[str, str] = spec.get("required_keys") or {}
    for key, type_spec in required_keys.items():
        if key not in obj:
            return {"passed": False, "extracted": obj, "detail": f"missing required key {key!r}"}
        if not _matches_json_type(obj[key], type_spec):
            return {
                "passed": False,
                "extracted": obj,
                "detail": f"key {key!r} expected {type_spec}, got {type(obj[key]).__name__} ({obj[key]!r})",
            }

    expected_values: dict[str, Any] = spec.get("expected_values") or {}
    for key, expected_val in expected_values.items():
        actual = obj.get(key)
        if isinstance(expected_val, bool):
            if actual is not expected_val:
                return {"passed": False, "extracted": obj, "detail": f"key {key!r}: expected {expected_val}, got {actual!r}"}
        elif isinstance(expected_val, (int, float)):
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), float(expected_val), rel_tol=0.0, abs_tol=1e-6
            ):
                return {"passed": False, "extracted": obj, "detail": f"key {key!r}: expected {expected_val}, got {actual!r}"}
        elif actual != expected_val:
            return {"passed": False, "extracted": obj, "detail": f"key {key!r}: expected {expected_val!r}, got {actual!r}"}

    return {"passed": True, "extracted": obj, "detail": f"valid JSON object ({len(required_keys)} keys verified)"}


def grade_logic(text: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Deterministic exact / regex / line-structure checker for reasoning & instruction-following."""
    cleaned = strip_think_blocks(text).strip()
    if not cleaned:
        return {"passed": False, "extracted": "", "detail": "empty response"}

    if "exact" in spec and spec["exact"] is not None:
        want = str(spec["exact"]).strip()
        if cleaned != want:
            return {"passed": False, "extracted": cleaned, "detail": f"exact mismatch: expected {want!r}"}

    lines = cleaned.splitlines()
    if spec.get("no_blank_lines"):
        if any(not ln.strip() for ln in lines):
            return {"passed": False, "extracted": cleaned, "detail": "blank line forbidden"}

    non_empty_lines = [ln for ln in lines if ln.strip()]
    if "non_empty_line_count" in spec and spec["non_empty_line_count"] is not None:
        want_n = int(spec["non_empty_line_count"])
        if len(non_empty_lines) != want_n:
            return {
                "passed": False,
                "extracted": cleaned,
                "detail": f"expected {want_n} non-empty lines, got {len(non_empty_lines)}",
            }

    if spec.get("line_regex"):
        line_pat = re.compile(str(spec["line_regex"]))
        for idx, ln in enumerate(non_empty_lines, start=1):
            if not line_pat.search(ln):
                return {
                    "passed": False,
                    "extracted": cleaned,
                    "detail": f"line {idx} failed pattern {spec['line_regex']!r}: {ln[:80]!r}",
                }

    for pat_str in spec.get("regex_all") or []:
        if not re.search(pat_str, cleaned):
            return {"passed": False, "extracted": cleaned, "detail": f"missing required pattern {pat_str!r}"}

    for pat_str in spec.get("regex_none") or []:
        m = re.search(pat_str, cleaned)
        if m:
            return {"passed": False, "extracted": cleaned, "detail": f"matched forbidden pattern {pat_str!r}: {m.group(0)!r}"}

    return {"passed": True, "extracted": cleaned, "detail": "all logic/instruction checks passed"}


def grade_probe(text: str, probe: dict[str, Any]) -> dict[str, Any]:
    spec = probe.get("grader") or {}
    gtype = spec.get("type") or probe.get("category")
    if gtype == "math":
        return grade_math(text, spec)
    if gtype == "python_code":
        return grade_python_code(text, spec)
    if gtype == "json_schema":
        return grade_json_schema(text, spec)
    if gtype == "logic":
        return grade_logic(text, spec)
    raise ValueError(f"unknown grader type {gtype!r} in probe {probe.get('id')!r}")


# ---------------------------------------------------------------------------
# Benchmark Suites (Prefill, Prefix, Decode, Concurrency, KeepAlive, Endpoints, Quality)
# ---------------------------------------------------------------------------


def _summarize_stream_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok_runs = [r for r in runs if r.get("ok")]
    return {
        "ok_count": len(ok_runs),
        "total_count": len(runs),
        "errors": [r.get("error") for r in runs if not r.get("ok")][:3],
        "ttft_ms": summarize_metric(r.get("ttft_ms") for r in ok_runs),
        "wall_decode_tps": summarize_metric(r.get("wall_decode_tps") for r in ok_runs),
        "total_wall_tps": summarize_metric(r.get("total_wall_tps") for r in ok_runs),
        "total_ms": summarize_metric(r.get("total_ms") for r in ok_runs),
        "tokens": summarize_metric(r.get("tokens") for r in ok_runs),
        "prompt_tokens": summarize_metric(r.get("prompt_tokens") for r in ok_runs),
        "server_prompt_ms": summarize_metric((r.get("timings") or {}).get("prompt_ms") for r in ok_runs),
        "server_prompt_tps": summarize_metric((r.get("timings") or {}).get("prompt_per_second") for r in ok_runs),
        "server_decode_ms": summarize_metric((r.get("timings") or {}).get("predicted_ms") for r in ok_runs),
        "server_decode_tps": summarize_metric((r.get("timings") or {}).get("predicted_per_second") for r in ok_runs),
        "cache_n": summarize_metric((r.get("timings") or {}).get("cache_n") for r in ok_runs),
        "draft_n": summarize_metric((r.get("timings") or {}).get("draft_n") for r in ok_runs),
        "draft_n_accepted": summarize_metric((r.get("timings") or {}).get("draft_n_accepted") for r in ok_runs),
        "acceptance_ratio": summarize_metric(r.get("acceptance_ratio") for r in ok_runs),
        "runs": [
            {k: v for k, v in r.items() if k not in ("content", "clean_content", "reasoning_content")}
            for r in runs
        ],
    }


def run_prefill_suite(
    client: BenchmarkClient,
    corpus: dict[str, Any],
    *,
    model: str,
    repeats: int,
    max_tokens: int,
    max_tokens_explicit: bool,
    disable_thinking: bool,
) -> dict[str, Any]:
    cases_out: dict[str, Any] = {}
    for item in corpus.get("prefill") or []:
        cid = str(item["id"])
        cap = int(max_tokens) if max_tokens_explicit else int(item.get("max_tokens", min(max_tokens, 32)))
        runs: list[dict[str, Any]] = []
        for r_idx in range(repeats):
            # Deterministic front-loaded hex nonce busts LCP right after the ChatML role header
            # so repeat 2+ never hits KV prefix cache from repeat 1.
            nonce = hashlib.sha256(f"cold:{cid}:{r_idx + 1}".encode("utf-8")).hexdigest()[:12]
            cold_prompt = f"[{nonce} run={r_idx + 1}]\n{item['prompt']}"
            res = client.stream_chat(
                [{"role": "user", "content": cold_prompt}],
                model=model,
                max_tokens=cap,
                temperature=0.0,
                seed=42,
                disable_thinking=disable_thinking,
            )
            runs.append(res)
        summary = _summarize_stream_runs(runs)
        summary["id"] = cid
        summary["label"] = item.get("label", cid)
        summary["target_tokens"] = item.get("target_tokens")
        summary["approx_tokens"] = item.get("approx_tokens")
        cases_out[cid] = summary
    return {"cases": cases_out}


def run_prefix_suite(
    client: BenchmarkClient,
    corpus: dict[str, Any],
    *,
    model: str,
    repeats: int,
    max_tokens: int,
    max_tokens_explicit: bool,
    disable_thinking: bool,
) -> dict[str, Any]:
    pcfg = corpus["prefix_cache"]
    cap = int(max_tokens) if max_tokens_explicit else int(pcfg.get("max_tokens", min(max_tokens, 64)))
    turns_cfg = pcfg["turns"]

    cold_runs: list[dict[str, Any]] = []
    warm_runs: list[dict[str, Any]] = []
    turn_runs: dict[int, list[dict[str, Any]]] = {int(t["turn"]): [] for t in turns_cfg}
    per_repeat_speedups: list[float] = []

    for r_idx in range(repeats):
        session_tag = f"bonsai-prefix-{r_idx + 1:04d}"
        nonce = hashlib.sha256(session_tag.encode("utf-8")).hexdigest()[:12]
        sys_text = f"[{nonce} session={session_tag}]\n{pcfg['system_prefix']}"
        messages: list[dict[str, str]] = [{"role": "system", "content": sys_text}]
        repeat_cold_ttft: float | None = None
        repeat_warm_ttfts: list[float] = []

        for t_cfg in turns_cfg:
            t_num = int(t_cfg["turn"])
            messages.append({"role": "user", "content": str(t_cfg["user"])})
            res = client.stream_chat(
                list(messages),
                model=model,
                max_tokens=cap,
                temperature=0.0,
                seed=42,
                disable_thinking=disable_thinking,
                prompt_cache_key=session_tag,
            )
            turn_runs[t_num].append(res)
            if t_num == 1:
                cold_runs.append(res)
                if res.get("ok") and _num(res.get("ttft_ms")) is not None:
                    repeat_cold_ttft = float(res["ttft_ms"])
            else:
                warm_runs.append(res)
                if res.get("ok") and _num(res.get("ttft_ms")) is not None:
                    repeat_warm_ttfts.append(float(res["ttft_ms"]))

            reply_text = (res.get("clean_content") or "").strip() or str(t_cfg.get("assistant_fallback", "OK."))
            messages.append({"role": "assistant", "content": reply_text})

        if repeat_cold_ttft is not None and repeat_warm_ttfts:
            warm_avg = statistics.fmean(repeat_warm_ttfts)
            if warm_avg > 0:
                per_repeat_speedups.append(repeat_cold_ttft / warm_avg)

    cold_summary = _summarize_stream_runs(cold_runs)
    warm_summary = _summarize_stream_runs(warm_runs)
    cold_med = cold_summary["ttft_ms"]["median"]
    warm_med = warm_summary["ttft_ms"]["median"]
    cold_mean = cold_summary["ttft_ms"]["mean"]
    warm_mean = warm_summary["ttft_ms"]["mean"]

    speedup_from_median = (cold_med / warm_med) if (cold_med is not None and warm_med and warm_med > 0) else None
    speedup_from_mean = (cold_mean / warm_mean) if (cold_mean is not None and warm_mean and warm_mean > 0) else None

    return {
        "id": pcfg.get("id", "prefix_3turn"),
        "target_shared_tokens": pcfg.get("target_shared_tokens"),
        "cold": cold_summary,
        "warm": warm_summary,
        "turns": {f"turn_{t}": _summarize_stream_runs(rs) for t, rs in sorted(turn_runs.items())},
        "cold_ttft_ms": cold_summary["ttft_ms"],
        "warm_ttft_ms": warm_summary["ttft_ms"],
        "warm_cache_n": warm_summary["cache_n"],
        "cold_cache_n": cold_summary["cache_n"],
        "speedup": summarize_metric(per_repeat_speedups),
        "speedup_median_ratio": speedup_from_median,
        "speedup_mean_ratio": speedup_from_mean,
    }


def run_decode_suite(
    client: BenchmarkClient,
    corpus: dict[str, Any],
    *,
    model: str,
    repeats: int,
    max_tokens: int,
    disable_thinking: bool,
) -> dict[str, Any]:
    regimes_out: dict[str, Any] = {}
    for item in corpus.get("decode_regimes") or []:
        rid = str(item["id"])
        resp_fmt = item.get("response_format")
        runs: list[dict[str, Any]] = []
        for _ in range(repeats):
            res = client.stream_chat(
                [{"role": "user", "content": str(item["prompt"])}],
                model=model,
                max_tokens=int(max_tokens),
                temperature=0.0,
                seed=42,
                disable_thinking=disable_thinking,
                response_format=resp_fmt,
            )
            runs.append(res)
        summary = _summarize_stream_runs(runs)
        summary["id"] = rid
        summary["regime"] = item.get("regime", rid)
        summary["response_format"] = resp_fmt
        regimes_out[rid] = summary
    return {"regimes": regimes_out}


def run_concurrency_suite(
    client: BenchmarkClient,
    corpus: dict[str, Any],
    *,
    model: str,
    levels: list[int],
    repeats: int,
    max_tokens: int,
    max_tokens_explicit: bool,
    disable_thinking: bool,
) -> dict[str, Any]:
    ccfg = corpus.get("concurrency") or {}
    prompts: list[str] = list(ccfg.get("prompts") or ["Explain KV cache reuse in three sentences."])
    cap = int(max_tokens) if max_tokens_explicit else int(ccfg.get("max_tokens", min(max_tokens, 96)))

    levels_out: dict[str, Any] = {}
    for c_level in levels:
        c = max(1, int(c_level))
        wave_agg_tps: list[float] = []
        wave_makespans_ms: list[float] = []
        all_reqs: list[dict[str, Any]] = []

        for r_idx in range(repeats):
            barrier = threading.Barrier(c) if c > 1 else None

            def _worker(slot_idx: int) -> dict[str, Any]:
                base_p = prompts[slot_idx % len(prompts)]
                tagged_p = f"[Concurrency-C{c}-R{r_idx + 1}-W{slot_idx + 1}]\n{base_p}"
                return client.stream_chat(
                    [{"role": "user", "content": tagged_p}],
                    model=model,
                    max_tokens=cap,
                    temperature=0.0,
                    seed=42,
                    disable_thinking=disable_thinking,
                    on_ready=barrier,
                )

            t_wave0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                futs = [pool.submit(_worker, i) for i in range(c)]
                wave_results = [f.result() for f in futs]
            makespan_s = max(time.perf_counter() - t_wave0, 1e-9)
            wave_makespans_ms.append(makespan_s * 1000.0)
            ok_wave = [w for w in wave_results if w.get("ok")]
            total_toks = sum(int(w.get("tokens") or 0) for w in ok_wave)
            if ok_wave and total_toks > 0:
                wave_agg_tps.append(total_toks / makespan_s)
            all_reqs.extend(wave_results)

        ok_reqs = [r for r in all_reqs if r.get("ok")]
        ttft_stats = summarize_metric(r.get("ttft_ms") for r in ok_reqs)
        levels_out[str(c)] = {
            "concurrency": c,
            "repeats": repeats,
            "success_count": len(ok_reqs),
            "total_count": len(all_reqs),
            "aggregate_tps": summarize_metric(wave_agg_tps),
            "makespan_ms": summarize_metric(wave_makespans_ms),
            "ttft_ms": ttft_stats,
            "ttft_p50_ms": ttft_stats["p50"],
            "ttft_p95_ms": ttft_stats["p95"],
            "per_stream_decode_tps": summarize_metric(r.get("wall_decode_tps") for r in ok_reqs),
            "errors": [r.get("error") for r in all_reqs if not r.get("ok")][:3],
        }
    return {"levels": levels_out}


def run_keepalive_suite(
    client: BenchmarkClient,
    corpus: dict[str, Any] | None = None,
    *,
    num_requests: int | None = None,
) -> dict[str, Any]:
    """Compare N requests over 1 persistent HTTPConnection vs N fresh connections."""
    kcfg = (corpus or {}).get("keepalive") or {}
    n_reqs = int(num_requests if num_requests is not None else kcfg.get("requests", 10))
    n_reqs = max(1, n_reqs)
    raw_route = str(kcfg.get("path", "/v1/models"))
    path = client.oai_path(raw_route) if raw_route.startswith("/v1") else client.root_path(raw_route)
    method = str(kcfg.get("method", "GET")).upper()

    # 1. Persistent HTTP/1.1 connection reuse
    ka_latencies_ms: list[float] = []
    ka_ok = 0
    connections_opened = 0
    reused_requests = 0
    ka_headers = client._build_headers(has_body=False, overrides={"Connection": "keep-alive"})
    conn = client.new_connection(timeout=min(client.timeout, 30.0))
    try:
        for i in range(n_reqs):
            t0 = time.perf_counter()
            reused_this_turn = False
            try:
                if conn.sock is None:
                    conn.connect()
                    connections_opened += 1
                else:
                    reused_this_turn = i > 0
                conn.request(method, path, headers=ka_headers)
                resp = conn.getresponse()
                _ = resp.read()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status == 200:
                    ka_ok += 1
                    ka_latencies_ms.append(elapsed_ms)
                if reused_this_turn:
                    reused_requests += 1
                conn_hdr = (resp.getheader("Connection") or "").lower()
                if resp.will_close or "close" in conn_hdr:
                    conn.close()
            except (http.client.RemoteDisconnected, http.client.BadStatusLine, ConnectionError, OSError):
                # Server closed idle keep-alive socket without warning; reconnect cleanly once.
                conn.close()
                t_retry = time.perf_counter()
                try:
                    conn.connect()
                    connections_opened += 1
                    conn.request(method, path, headers=ka_headers)
                    resp = conn.getresponse()
                    _ = resp.read()
                    elapsed_ms = (time.perf_counter() - t_retry) * 1000.0
                    if resp.status == 200:
                        ka_ok += 1
                        ka_latencies_ms.append(elapsed_ms)
                    if resp.will_close or "close" in (resp.getheader("Connection") or "").lower():
                        conn.close()
                except (OSError, http.client.HTTPException):
                    conn.close()
    finally:
        conn.close()

    # 2. Fresh TCP socket per request
    fresh_latencies_ms: list[float] = []
    fresh_ok = 0
    fresh_connections = 0
    fresh_headers = client._build_headers(has_body=False, overrides={"Connection": "close"})
    for _ in range(n_reqs):
        fconn = client.new_connection(timeout=min(client.timeout, 30.0))
        t0 = time.perf_counter()
        try:
            fconn.connect()
            fresh_connections += 1
            fconn.request(method, path, headers=fresh_headers)
            resp = fconn.getresponse()
            _ = resp.read()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status == 200:
                fresh_ok += 1
                fresh_latencies_ms.append(elapsed_ms)
        except (OSError, http.client.HTTPException):
            pass
        finally:
            fconn.close()

    ka_stats = summarize_metric(ka_latencies_ms)
    fresh_stats = summarize_metric(fresh_latencies_ms)
    ka_mean = ka_stats["mean"]
    fresh_mean = fresh_stats["mean"]
    ka_med = ka_stats["median"]
    fresh_med = fresh_stats["median"]

    speedup_mean = (fresh_mean / ka_mean) if (ka_mean and ka_mean > 0 and fresh_mean is not None) else None
    speedup_median = (fresh_med / ka_med) if (ka_med and ka_med > 0 and fresh_med is not None) else None
    saved_ms = (fresh_mean - ka_mean) if (ka_mean is not None and fresh_mean is not None) else None

    return {
        "path": path,
        "requests_per_mode": n_reqs,
        "keepalive": {
            "ok_count": ka_ok,
            "total_count": n_reqs,
            "connections_opened": connections_opened,
            "reused_requests": reused_requests,
            "latency_ms": ka_stats,
            "total_ms": sum(ka_latencies_ms) if ka_latencies_ms else None,
        },
        "fresh_socket": {
            "ok_count": fresh_ok,
            "total_count": n_reqs,
            "connections_opened": fresh_connections,
            "reused_requests": 0,
            "latency_ms": fresh_stats,
            "total_ms": sum(fresh_latencies_ms) if fresh_latencies_ms else None,
        },
        "speedup_mean": speedup_mean,
        "speedup_median": speedup_median,
        "latency_saved_mean_ms": saved_ms,
    }


def run_endpoints_suite(
    client: BenchmarkClient,
    *,
    model: str,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    """Probe OpenAI & Sonder endpoint parity matrix and verify HTTP status + payload validity."""
    probes: list[dict[str, Any]] = []

    # 1. GET /v1/models
    st, body, ms, _ = client.request_json("GET", client.oai_path("/models"), timeout=15.0)
    ok_models = st == 200 and isinstance(body, dict) and isinstance(body.get("data"), list)
    probes.append({
        "id": "models",
        "endpoint": "GET /v1/models",
        "http_status": st,
        "supported": ok_models,
        "latency_ms": ms,
    })

    # 2. POST /v1/chat/completions (stream=True)
    stream_res = client.stream_chat(
        [{"role": "user", "content": "Reply with OK."}],
        model=model,
        max_tokens=8,
        temperature=0.0,
        seed=42,
        disable_thinking=disable_thinking,
        timeout=30.0,
    )
    probes.append({
        "id": "chat_stream",
        "endpoint": "POST /v1/chat/completions (stream)",
        "http_status": stream_res["http_status"],
        "supported": bool(stream_res["ok"]),
        "latency_ms": stream_res["total_ms"],
    })

    # 3. POST /v1/chat/completions (non-stream)
    chat_body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 8,
        "temperature": 0.0,
        "seed": 42,
        "stream": False,
    }
    if disable_thinking:
        chat_body["chat_template_kwargs"] = {"enable_thinking": False}
    st, body, ms, _ = client.request_json("POST", client.oai_path("/chat/completions"), chat_body, timeout=30.0)
    ok_chat_ns = st == 200 and isinstance(body, dict) and isinstance(body.get("choices"), list) and len(body["choices"]) > 0
    probes.append({
        "id": "chat_non_stream",
        "endpoint": "POST /v1/chat/completions (non-stream)",
        "http_status": st,
        "supported": ok_chat_ns,
        "latency_ms": ms,
    })

    # 4. POST /v1/chat/completions (response_format: json_object)
    json_chat_body = dict(chat_body)
    json_chat_body["messages"] = [{"role": "user", "content": 'Return JSON {"ok": true}.'}]
    json_chat_body["max_tokens"] = 32
    json_chat_body["response_format"] = {"type": "json_object"}
    st, body, ms, _ = client.request_json("POST", client.oai_path("/chat/completions"), json_chat_body, timeout=30.0)
    ok_json_fmt = False
    if st == 200 and isinstance(body, dict) and isinstance(body.get("choices"), list) and body["choices"]:
        msg_content = ((body["choices"][0] or {}).get("message") or {}).get("content") or ""
        try:
            parsed_json = json.loads(strip_think_blocks(str(msg_content)).strip())
            ok_json_fmt = isinstance(parsed_json, dict)
        except ValueError:
            ok_json_fmt = False
    probes.append({
        "id": "chat_json_object",
        "endpoint": "POST /v1/chat/completions (json_object)",
        "http_status": st,
        "supported": ok_json_fmt,
        "latency_ms": ms,
    })

    # 5. POST /v1/completions
    cmpl_body = {
        "model": model,
        "prompt": "2 + 2 =",
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False,
    }
    st, body, ms, _ = client.request_json("POST", client.oai_path("/completions"), cmpl_body, timeout=30.0)
    ok_cmpl = st == 200 and isinstance(body, dict) and isinstance(body.get("choices"), list) and len(body["choices"]) > 0
    probes.append({
        "id": "completions",
        "endpoint": "POST /v1/completions",
        "http_status": st,
        "supported": ok_cmpl,
        "latency_ms": ms,
    })

    # 6. POST /v1/embeddings
    emb_body = {"model": model, "input": "Bonsai ternary inference benchmark"}
    st, body, ms, _ = client.request_json("POST", client.oai_path("/embeddings"), emb_body, timeout=30.0)
    ok_emb = (
        st == 200
        and isinstance(body, dict)
        and isinstance(body.get("data"), list)
        and len(body["data"]) > 0
        and isinstance((body["data"][0] or {}).get("embedding"), list)
    )
    probes.append({
        "id": "embeddings",
        "endpoint": "POST /v1/embeddings",
        "http_status": st,
        "supported": ok_emb,
        "latency_ms": ms,
    })

    # 7. POST /v1/tokenize
    tok_body = {"model": model, "content": "Hello Bonsai benchmark"}
    st, body, ms, _ = client.request_json("POST", client.oai_path("/tokenize"), tok_body, timeout=15.0)
    ok_tok = st == 200 and isinstance(body, dict) and isinstance(body.get("tokens"), list)
    probes.append({
        "id": "tokenize",
        "endpoint": "POST /v1/tokenize",
        "http_status": st,
        "supported": ok_tok,
        "latency_ms": ms,
    })

    # 8. GET /v1/sonder/health
    st, body, ms, _ = client.request_json("GET", client.root_path("/v1/sonder/health"), timeout=15.0)
    ok_shealth = st == 200 and isinstance(body, dict)
    probes.append({
        "id": "sonder_health",
        "endpoint": "GET /v1/sonder/health",
        "http_status": st,
        "supported": ok_shealth,
        "latency_ms": ms,
    })

    supported_count = sum(1 for p in probes if p["supported"])
    return {
        "supported_count": supported_count,
        "total_count": len(probes),
        "probes": probes,
        "by_id": {p["id"]: p for p in probes},
    }


def run_quality_suite(
    client: BenchmarkClient,
    corpus: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
    max_tokens_explicit: bool,
    disable_thinking: bool,
) -> dict[str, Any]:
    probes_out: list[dict[str, Any]] = []
    by_category: dict[str, dict[str, Any]] = {
        cat: {"passed": 0, "total": 0, "pass_rate_pct": 0.0} for cat in QUALITY_CATEGORIES
    }

    for probe in corpus.get("quality_probes") or []:
        pid = str(probe["id"])
        cat = str(probe["category"])
        by_category.setdefault(cat, {"passed": 0, "total": 0, "pass_rate_pct": 0.0})
        cap = int(max_tokens) if max_tokens_explicit else int(probe.get("max_tokens", max_tokens))
        resp_fmt = probe.get("response_format")

        res = client.stream_chat(
            [{"role": "user", "content": str(probe["prompt"])}],
            model=model,
            max_tokens=cap,
            temperature=float(probe.get("temperature", 0.0)),
            seed=int(probe.get("seed", 42)),
            disable_thinking=disable_thinking,
            response_format=resp_fmt,
        )
        if not res["ok"]:
            grade = {"passed": False, "extracted": None, "detail": f"request error: {res['error']}"}
        else:
            grade = grade_probe(res["clean_content"], probe)

        passed = bool(grade["passed"])
        by_category[cat]["total"] += 1
        if passed:
            by_category[cat]["passed"] += 1

        probes_out.append({
            "id": pid,
            "category": cat,
            "passed": passed,
            "detail": grade["detail"],
            "ttft_ms": res["ttft_ms"],
            "total_ms": res["total_ms"],
            "tokens": res["tokens"],
            "wall_decode_tps": res["wall_decode_tps"],
            "clean_content": res["clean_content"],
            "output_sha256": res["output_sha256"],
        })

    for stats in by_category.values():
        stats["pass_rate_pct"] = (
            (100.0 * stats["passed"] / stats["total"]) if stats["total"] > 0 else 0.0
        )

    total_passed = sum(1 for p in probes_out if p["passed"])
    total_probes = len(probes_out)
    overall_pct = (100.0 * total_passed / total_probes) if total_probes > 0 else 0.0

    return {
        "passed": total_passed,
        "total": total_probes,
        "pass_rate_pct": overall_pct,
        "categories": by_category,
        "probes": probes_out,
        "by_id": {p["id"]: p for p in probes_out},
    }


# ---------------------------------------------------------------------------
# Cross-Target Output Parity Analysis
# ---------------------------------------------------------------------------


def token_similarity_ratio(a_text: str, b_text: str) -> float:
    """Word/token-level SequenceMatcher similarity in [0.0, 1.0]."""
    a_norm = a_text.strip()
    b_norm = b_text.strip()
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return difflib.SequenceMatcher(None, a_norm.split(), b_norm.split()).ratio()


def compute_cross_target_parity(target_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compare quality outputs of all challenger targets against baseline (first target with quality)."""
    qual_targets = [
        t for t in target_results if isinstance((t.get("suites") or {}).get("quality"), dict)
    ]
    if len(qual_targets) < 2:
        return None

    base = qual_targets[0]
    base_name = str(base["name"])
    base_qual = base["suites"]["quality"]
    base_by_id: dict[str, dict[str, Any]] = base_qual.get("by_id") or {
        p["id"]: p for p in base_qual.get("probes") or []
    }

    comparisons: list[dict[str, Any]] = []
    for cand in qual_targets[1:]:
        cand_name = str(cand["name"])
        cand_qual = cand["suites"]["quality"]
        cand_by_id: dict[str, dict[str, Any]] = cand_qual.get("by_id") or {
            p["id"]: p for p in cand_qual.get("probes") or []
        }

        common_ids = [pid for pid in base_by_id if pid in cand_by_id]
        exact_matches = 0
        outcome_agreements = 0
        similarities: list[float] = []
        per_probe: list[dict[str, Any]] = []

        for pid in common_ids:
            bp = base_by_id[pid]
            cp = cand_by_id[pid]
            b_txt = str(bp.get("clean_content") or "").strip()
            c_txt = str(cp.get("clean_content") or "").strip()
            is_exact = b_txt == c_txt
            if is_exact:
                exact_matches += 1
            if bool(bp.get("passed")) == bool(cp.get("passed")):
                outcome_agreements += 1
            sim_pct = token_similarity_ratio(b_txt, c_txt) * 100.0
            similarities.append(sim_pct)
            per_probe.append({
                "id": pid,
                "category": bp.get("category"),
                "baseline_passed": bool(bp.get("passed")),
                "candidate_passed": bool(cp.get("passed")),
                "exact_match": is_exact,
                "token_similarity_pct": sim_pct,
            })

        n = len(common_ids)
        comparisons.append({
            "baseline": base_name,
            "target": cand_name,
            "probes_compared": n,
            "exact_match_count": exact_matches,
            "exact_match_pct": (100.0 * exact_matches / n) if n > 0 else 0.0,
            "token_similarity_pct": float(statistics.fmean(similarities)) if similarities else 0.0,
            "baseline_pass_rate_pct": float(base_qual.get("pass_rate_pct", 0.0)),
            "target_pass_rate_pct": float(cand_qual.get("pass_rate_pct", 0.0)),
            "pass_rate_delta_pct": float(cand_qual.get("pass_rate_pct", 0.0))
            - float(base_qual.get("pass_rate_pct", 0.0)),
            "outcome_agreement_pct": (100.0 * outcome_agreements / n) if n > 0 else 0.0,
            "per_probe": per_probe,
        })

    return {
        "baseline": base_name,
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Managed Sequential Process Lifecycle
# ---------------------------------------------------------------------------


def terminate_process_tree(proc: subprocess.Popen[Any], grace_s: float = 4.0) -> None:
    """Cleanly terminate a managed server and all child processes before starting the next target."""
    if proc.poll() is not None:
        return
    pid = proc.pid
    try:
        if IS_WINDOWS:
            proc.terminate()
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                return
    except OSError:
        pass

    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass

    if IS_WINDOWS:
        # Always reap any child processes (e.g., sonder-infer's llamaserver child) via taskkill /T /F.
        taskkill = shutil.which("taskkill") or "taskkill"
        try:
            subprocess.run(
                [taskkill, "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=10.0,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            try:
                proc.kill()
            except OSError:
                pass
    elif proc.poll() is None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass

    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


class ManagedServer:
    """Context manager launching ONE server process at a time, polling health, and reaping its tree."""

    def __init__(self, spec: dict[str, Any], log_fn: Any = None) -> None:
        self.name = str(spec["name"])
        self.url = str(spec["url"])
        self.health_path = str(spec.get("health_path") or "/health")
        self.startup_timeout_s = float(spec.get("startup_timeout_s", 120.0))
        self.cwd = spec.get("cwd")
        self.env_overrides = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
        raw_cmd = spec["cmd"]
        if isinstance(raw_cmd, list):
            self.cmd = [str(x) for x in raw_cmd]
        elif isinstance(raw_cmd, str):
            self.cmd = shlex.split(raw_cmd, posix=not IS_WINDOWS)
        else:
            raise ValueError(f"target {self.name!r}: 'cmd' must be a list of args or command string")
        if not self.cmd:
            raise ValueError(f"target {self.name!r}: 'cmd' cannot be empty")
        self.log_fn = log_fn or (lambda _msg: None)
        self.proc: subprocess.Popen[Any] | None = None
        self._log_file: Any = None
        self.log_path: str | None = None
        self.startup_ms: float | None = None

    def __enter__(self) -> ManagedServer:
        env = dict(os.environ)
        env.update(self.env_overrides)
        fd, log_path = tempfile.mkstemp(prefix=f"bonsai_srv_{self.name}_", suffix=".log")
        os.close(fd)
        self.log_path = log_path
        self._log_file = open(log_path, "w", encoding="utf-8", errors="replace")

        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": self._log_file,
            "stderr": subprocess.STDOUT,
            "cwd": self.cwd,
            "env": env,
        }
        if IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        self.log_fn(f"[{self.name}] launching managed server: {self.cmd}")
        t0 = time.perf_counter()
        try:
            self.proc = subprocess.Popen(self.cmd, **popen_kwargs)
            self._wait_for_health(t0)
            self.startup_ms = (time.perf_counter() - t0) * 1000.0
            return self
        except Exception:
            self.stop()
            raise

    def _wait_for_health(self, t0: float) -> None:
        assert self.proc is not None
        probe_client = BenchmarkClient(self.url, timeout=3.0)
        path = (
            probe_client.root_path(self.health_path)
            if self.health_path.startswith("/")
            else "/" + self.health_path
        )
        deadline = t0 + self.startup_timeout_s
        last_detail = "no response yet"
        while time.perf_counter() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                tail = self._read_log_tail()
                raise RuntimeError(
                    f"managed target {self.name!r} exited early with code {rc} before health ready. Log tail:\n{tail}"
                )
            st, body, _, _ = probe_client.request_json("GET", path, timeout=2.0)
            if st == 200:
                return
            last_detail = f"HTTP {st}: {str(body)[:120]}"
            time.sleep(0.15)
        tail = self._read_log_tail()
        raise TimeoutError(
            f"managed target {self.name!r} did not return HTTP 200 on {path} within {self.startup_timeout_s:.1f}s "
            f"(last={last_detail}). Log tail:\n{tail}"
        )

    def _read_log_tail(self, max_chars: int = 1200) -> str:
        if self._log_file is not None:
            try:
                self._log_file.flush()
            except OSError:
                pass
        if not self.log_path:
            return ""
        try:
            txt = Path(self.log_path).read_text(encoding="utf-8", errors="replace")
            return txt[-max_chars:]
        except OSError:
            return ""

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None

    def stop(self) -> None:
        if self.proc is not None:
            self.log_fn(f"[{self.name}] terminating process tree (pid={self.proc.pid})")
            terminate_process_tree(self.proc)
            self.proc = None
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None
        if self.log_path:
            try:
                os.remove(self.log_path)
            except OSError:
                pass
            self.log_path = None

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Markdown Side-by-Side Report Generator
# ---------------------------------------------------------------------------


def _fmt(val: Any, digits: int = 1, suffix: str = "") -> str:
    n = _num(val)
    if n is None:
        return "—"
    return f"{n:.{digits}f}{suffix}"


def _fmt_med_mean(stat: dict[str, Any] | None, digits: int = 1) -> str:
    if not isinstance(stat, dict) or not stat.get("n"):
        return "—"
    return f"{_fmt(stat.get('median'), digits)} / {_fmt(stat.get('mean'), digits)}"


def _delta_verdict(
    base_val: Any,
    cand_val: Any,
    *,
    lower_is_better: bool,
    tie_pct: float = 1.0,
) -> tuple[str, str]:
    """Return ('+12.3%', 'WIN'|'TIE'|'LOSS'|'—') comparing candidate vs baseline."""
    b = _num(base_val)
    c = _num(cand_val)
    if b is None or c is None or b == 0.0:
        return "—", "—"
    pct = ((c - b) / abs(b)) * 100.0
    pct_str = f"{pct:+.1f}%"
    if abs(pct) < tie_pct:
        return pct_str, "TIE"
    improved = (c < b) if lower_is_better else (c > b)
    return pct_str, ("WIN" if improved else "LOSS")


def render_markdown_report(report: dict[str, Any], *, tie_pct: float = 1.0) -> str:
    """Render side-by-side Markdown report with % delta vs baseline (first target)."""
    lines: list[str] = []
    targets: list[dict[str, Any]] = report.get("targets") or []
    corpus_meta = report.get("corpus") or {}
    cfg_meta = report.get("config") or {}

    lines.append("# Bonsai Head-to-Head Benchmark Report (PrismML vs Sonder-Inference)")
    lines.append("")
    lines.append(
        f"- **Corpus**: `{corpus_meta.get('name')}` (`v{corpus_meta.get('version')}`, fnv1a `{corpus_meta.get('fnv1a')}`)"
    )
    lines.append(
        f"- **Settings**: mode=`{cfg_meta.get('mode')}`, suites=`{','.join(cfg_meta.get('suites') or [])}`, "
        f"repeats=`{cfg_meta.get('repeats')}`, max_tokens=`{cfg_meta.get('max_tokens')}`, "
        f"disable_thinking=`{cfg_meta.get('disable_thinking')}`, tie_band=`±{tie_pct:.1f}%`"
    )
    lines.append(f"- **Generated**: `{report.get('timestamp')}` on `{report.get('platform')}`")
    lines.append("")

    # Target Overview & Resource Footprint
    lines.append("## Targets & Resource Footprint")
    lines.append("")
    lines.append("| Target | Role | URL | Model | Startup ms | Peak VRAM MiB | Peak RSS MiB |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for idx, t in enumerate(targets):
        res_fp = t.get("resources") or {}
        role = "Baseline" if idx == 0 else "Challenger"
        lines.append(
            f"| `{t['name']}` | {role} | `{t['url']}` | `{t.get('model')}` | "
            f"{_fmt(t.get('startup_ms'), 0)} | {_fmt(res_fp.get('peak_vram_mib'), 1)} | "
            f"{_fmt(res_fp.get('peak_rss_mib'), 1)} |"
        )
    lines.append("")

    if not targets:
        return "\n".join(lines) + "\n"

    base = targets[0]
    bname = base["name"]
    challengers = targets[1:]

    # Helper to emit pairwise or single-target sections cleanly
    comparison_pairs = [(base, ch) for ch in challengers] if challengers else [(base, None)]

    for base_t, cand_t in comparison_pairs:
        cname = cand_t["name"] if cand_t else None
        if cand_t:
            lines.append(f"## Comparison: `{cname}` vs `{bname}` (Baseline)")
            lines.append("")

        bsuites = base_t.get("suites") or {}
        csuites = (cand_t.get("suites") if cand_t else None) or {}

        # 1. Cold Prefill
        if "prefill" in bsuites:
            lines.append("### 1. Cold Prefill Latency (TTFT ms — lower is better)")
            lines.append("")
            if cand_t:
                lines.append(
                    f"| Context | `{bname}` TTFT med/mean (ms) | `{cname}` TTFT med/mean (ms) | `% Delta` (med) | Verdict | "
                    f"Server Prefill tok/s (`{bname}` / `{cname}`) |"
                )
                lines.append("|---|---:|---:|---:|---|---:|")
                bcases = bsuites["prefill"].get("cases") or {}
                ccases = (csuites.get("prefill") or {}).get("cases") or {}
                for cid, bcase in bcases.items():
                    ccase = ccases.get(cid) or {}
                    d_str, verd = _delta_verdict(
                        bcase["ttft_ms"]["median"], ccase.get("ttft_ms", {}).get("median"), lower_is_better=True, tie_pct=tie_pct
                    )
                    lines.append(
                        f"| `{bcase.get('label', cid)}` | {_fmt_med_mean(bcase.get('ttft_ms'))} | "
                        f"{_fmt_med_mean(ccase.get('ttft_ms'))} | {d_str} | **{verd}** | "
                        f"{_fmt(bcase.get('server_prompt_tps', {}).get('median'))} / "
                        f"{_fmt(ccase.get('server_prompt_tps', {}).get('median'))} |"
                    )
            else:
                lines.append(f"| Context | `{bname}` TTFT med/mean (ms) | Server Prefill tok/s |")
                lines.append("|---|---:|---:|")
                for cid, bcase in (bsuites["prefill"].get("cases") or {}).items():
                    lines.append(
                        f"| `{bcase.get('label', cid)}` | {_fmt_med_mean(bcase.get('ttft_ms'))} | "
                        f"{_fmt(bcase.get('server_prompt_tps', {}).get('median'))} |"
                    )
            lines.append("")

        # 2. Warm Multi-Turn Prefix Cache
        if "prefix" in bsuites:
            lines.append("### 2. Multi-Turn KV Prefix Cache Reuse (~1500-tok shared prefix)")
            lines.append("")
            bp = bsuites["prefix"]
            cp = csuites.get("prefix") or {}
            if cand_t:
                lines.append(f"| Metric | `{bname}` | `{cname}` | `% Delta` | Verdict |")
                lines.append("|---|---:|---:|---:|---|")
                rows_spec = [
                    ("Cold Turn-1 TTFT med (ms)", bp["cold_ttft_ms"]["median"], cp.get("cold_ttft_ms", {}).get("median"), True, 1),
                    ("Warm Turn-2/3 TTFT med (ms)", bp["warm_ttft_ms"]["median"], cp.get("warm_ttft_ms", {}).get("median"), True, 1),
                    ("Prefix TTFT Speedup (x)", bp.get("speedup_median_ratio"), cp.get("speedup_median_ratio"), False, 2),
                    ("Warm Cached Tokens (`cache_n`)", bp["warm_cache_n"]["median"], cp.get("warm_cache_n", {}).get("median"), False, 0),
                ]
                for label, bv, cv, lower_better, digs in rows_spec:
                    d_str, verd = _delta_verdict(bv, cv, lower_is_better=lower_better, tie_pct=tie_pct)
                    lines.append(f"| {label} | {_fmt(bv, digs)} | {_fmt(cv, digs)} | {d_str} | **{verd}** |")
            else:
                lines.append(f"| Metric | `{bname}` |")
                lines.append("|---|---:|")
                lines.append(f"| Cold Turn-1 TTFT med/mean (ms) | {_fmt_med_mean(bp.get('cold_ttft_ms'))} |")
                lines.append(f"| Warm Turn-2/3 TTFT med/mean (ms) | {_fmt_med_mean(bp.get('warm_ttft_ms'))} |")
                lines.append(f"| Prefix TTFT Speedup (x) | {_fmt(bp.get('speedup_median_ratio'), 2)}x |")
                lines.append(f"| Warm Cached Tokens (`cache_n`) | {_fmt(bp['warm_cache_n']['median'], 0)} |")
            lines.append("")

        # 3. Decode Speed Regimes
        if "decode" in bsuites:
            lines.append("### 3. Decode Speed Regimes (Wall Decode tok/s — higher is better)")
            lines.append("")
            bregs = bsuites["decode"].get("regimes") or {}
            cregs = (csuites.get("decode") or {}).get("regimes") or {}
            if cand_t:
                lines.append(
                    f"| Regime | `{bname}` Wall tok/s (med/mean) | `{cname}` Wall tok/s (med/mean) | `% Delta` (med) | Verdict | "
                    f"Server tok/s (`{bname}` / `{cname}`) | Spec Acc (`{bname}` / `{cname}`) |"
                )
                lines.append("|---|---:|---:|---:|---|---:|---:|")
                for rid, br in bregs.items():
                    cr = cregs.get(rid) or {}
                    d_str, verd = _delta_verdict(
                        br["wall_decode_tps"]["median"],
                        cr.get("wall_decode_tps", {}).get("median"),
                        lower_is_better=False,
                        tie_pct=tie_pct,
                    )
                    b_acc = _num(br.get("acceptance_ratio", {}).get("mean"))
                    c_acc = _num(cr.get("acceptance_ratio", {}).get("mean"))
                    b_acc_s = f"{b_acc * 100:.1f}%" if b_acc is not None else "—"
                    c_acc_s = f"{c_acc * 100:.1f}%" if c_acc is not None else "—"
                    lines.append(
                        f"| `{rid}` | {_fmt_med_mean(br.get('wall_decode_tps'))} | "
                        f"{_fmt_med_mean(cr.get('wall_decode_tps'))} | {d_str} | **{verd}** | "
                        f"{_fmt(br.get('server_decode_tps', {}).get('median'))} / "
                        f"{_fmt(cr.get('server_decode_tps', {}).get('median'))} | {b_acc_s} / {c_acc_s} |"
                    )
            else:
                lines.append(f"| Regime | `{bname}` Wall tok/s (med/mean) | Server tok/s | Spec Acc |")
                lines.append("|---|---:|---:|---:|")
                for rid, br in bregs.items():
                    b_acc = _num(br.get("acceptance_ratio", {}).get("mean"))
                    b_acc_s = f"{b_acc * 100:.1f}%" if b_acc is not None else "—"
                    lines.append(
                        f"| `{rid}` | {_fmt_med_mean(br.get('wall_decode_tps'))} | "
                        f"{_fmt(br.get('server_decode_tps', {}).get('median'))} | {b_acc_s} |"
                    )
            lines.append("")

        # 4. Concurrency Scaling
        if "concurrency" in bsuites:
            lines.append("### 4. Concurrency Scaling (Aggregate tok/s — higher is better)")
            lines.append("")
            blevs = bsuites["concurrency"].get("levels") or {}
            clevs = (csuites.get("concurrency") or {}).get("levels") or {}
            if cand_t:
                lines.append(
                    f"| Level | `{bname}` Agg tok/s (med) | `{cname}` Agg tok/s (med) | `% Delta` | Verdict | "
                    f"`{bname}` p50/p95 TTFT ms | `{cname}` p50/p95 TTFT ms | OK (`{bname}` / `{cname}`) |"
                )
                lines.append("|---|---:|---:|---:|---|---:|---:|---:|")
                for lk, bl in blevs.items():
                    cl = clevs.get(lk) or {}
                    d_str, verd = _delta_verdict(
                        bl["aggregate_tps"]["median"],
                        cl.get("aggregate_tps", {}).get("median"),
                        lower_is_better=False,
                        tie_pct=tie_pct,
                    )
                    lines.append(
                        f"| `C={lk}` | {_fmt(bl['aggregate_tps']['median'])} | "
                        f"{_fmt(cl.get('aggregate_tps', {}).get('median'))} | {d_str} | **{verd}** | "
                        f"{_fmt(bl.get('ttft_p50_ms'))} / {_fmt(bl.get('ttft_p95_ms'))} | "
                        f"{_fmt(cl.get('ttft_p50_ms'))} / {_fmt(cl.get('ttft_p95_ms'))} | "
                        f"{bl.get('success_count')}/{bl.get('total_count')} vs "
                        f"{cl.get('success_count', 0)}/{cl.get('total_count', 0)} |"
                    )
            else:
                lines.append(f"| Level | `{bname}` Agg tok/s (med) | p50 / p95 TTFT ms | OK |")
                lines.append("|---|---:|---:|---:|")
                for lk, bl in blevs.items():
                    lines.append(
                        f"| `C={lk}` | {_fmt(bl['aggregate_tps']['median'])} | "
                        f"{_fmt(bl.get('ttft_p50_ms'))} / {_fmt(bl.get('ttft_p95_ms'))} | "
                        f"{bl.get('success_count')}/{bl.get('total_count')} |"
                    )
            lines.append("")

        # 5. Keep-Alive vs Fresh Socket Overhead
        if "keepalive" in bsuites:
            lines.append("### 5. HTTP Keep-Alive vs Fresh-Socket Overhead (ms — lower is better)")
            lines.append("")
            bka = bsuites["keepalive"]
            lines.append("| Target | Keep-Alive med/mean (ms) | Fresh-Socket med/mean (ms) | Reuse Speedup | Reused / Conns |")
            lines.append("|---|---:|---:|---:|---:|")
            for t_obj in ([base_t, cand_t] if cand_t else [base_t]):
                assert t_obj is not None
                ka_s = (t_obj.get("suites") or {}).get("keepalive")
                if not ka_s:
                    continue
                ka_part = ka_s["keepalive"]
                fr_part = ka_s["fresh_socket"]
                lines.append(
                    f"| `{t_obj['name']}` | {_fmt_med_mean(ka_part['latency_ms'], 2)} | "
                    f"{_fmt_med_mean(fr_part['latency_ms'], 2)} | {_fmt(ka_s.get('speedup_mean'), 2)}x | "
                    f"{ka_part['reused_requests']} reused / {ka_part['connections_opened']} opened |"
                )
            if cand_t and "keepalive" in csuites:
                cka = csuites["keepalive"]
                d_str, verd = _delta_verdict(
                    bka["keepalive"]["latency_ms"]["median"],
                    cka["keepalive"]["latency_ms"]["median"],
                    lower_is_better=True,
                    tie_pct=tie_pct,
                )
                lines.append("")
                lines.append(f"- Keep-alive median latency delta (`{cname}` vs `{bname}`): `{d_str}` (**{verd}**)")
            lines.append("")

        # 6. Endpoint Parity Matrix
        if "endpoints" in bsuites:
            lines.append("### 6. Endpoint Parity Matrix")
            lines.append("")
            beps = bsuites["endpoints"].get("probes") or []
            ceps_by_id = ((csuites.get("endpoints") or {}).get("by_id")) or {}
            if cand_t:
                lines.append(f"| Endpoint | `{bname}` | `{cname}` |")
                lines.append("|---|---|---|")
                for bp_ep in beps:
                    cp_ep = ceps_by_id.get(bp_ep["id"]) or {}
                    b_cell = f"{'PASS' if bp_ep['supported'] else 'UNSUPPORTED'} (`{bp_ep.get('http_status')}`)"
                    c_cell = f"{'PASS' if cp_ep.get('supported') else 'UNSUPPORTED'} (`{cp_ep.get('http_status')}`)"
                    lines.append(f"| `{bp_ep['endpoint']}` | {b_cell} | {c_cell} |")
            else:
                lines.append(f"| Endpoint | `{bname}` |")
                lines.append("|---|---|")
                for bp_ep in beps:
                    b_cell = f"{'PASS' if bp_ep['supported'] else 'UNSUPPORTED'} (`{bp_ep.get('http_status')}`)"
                    lines.append(f"| `{bp_ep['endpoint']}` | {b_cell} |")
            lines.append("")

        # 7. Quality & Cross-Target Output Parity
        if "quality" in bsuites:
            lines.append("### 7. Quality & Cross-Target Output Parity (`temperature=0, seed=42`)")
            lines.append("")
            bq = bsuites["quality"]
            cq = csuites.get("quality") or {}
            if cand_t and cq:
                lines.append(f"| Category | `{bname}` Pass | `{cname}` Pass | Pass-Rate Delta |")
                lines.append("|---|---:|---:|---:|")
                for cat in QUALITY_CATEGORIES:
                    bc = (bq.get("categories") or {}).get(cat) or {"passed": 0, "total": 0, "pass_rate_pct": 0.0}
                    cc = (cq.get("categories") or {}).get(cat) or {"passed": 0, "total": 0, "pass_rate_pct": 0.0}
                    diff = cc["pass_rate_pct"] - bc["pass_rate_pct"]
                    lines.append(
                        f"| `{cat}` | {bc['passed']}/{bc['total']} ({bc['pass_rate_pct']:.1f}%) | "
                        f"{cc['passed']}/{cc['total']} ({cc['pass_rate_pct']:.1f}%) | {diff:+.1f}% |"
                    )
                tot_diff = float(cq.get("pass_rate_pct", 0.0)) - float(bq.get("pass_rate_pct", 0.0))
                lines.append(
                    f"| **OVERALL** | **{bq['passed']}/{bq['total']} ({bq['pass_rate_pct']:.1f}%)** | "
                    f"**{cq['passed']}/{cq['total']} ({cq['pass_rate_pct']:.1f}%)** | **{tot_diff:+.1f}%** |"
                )
            else:
                lines.append(f"| Category | `{bname}` Pass |")
                lines.append("|---|---:|")
                for cat in QUALITY_CATEGORIES:
                    bc = (bq.get("categories") or {}).get(cat) or {"passed": 0, "total": 0, "pass_rate_pct": 0.0}
                    lines.append(f"| `{cat}` | {bc['passed']}/{bc['total']} ({bc['pass_rate_pct']:.1f}%) |")
                lines.append(f"| **OVERALL** | **{bq['passed']}/{bq['total']} ({bq['pass_rate_pct']:.1f}%)** |")
            lines.append("")

    parity = report.get("cross_target_parity")
    if isinstance(parity, dict) and parity.get("comparisons"):
        lines.append("### Cross-Target Output Parity Summary")
        lines.append("")
        lines.append(
            "| Baseline | Challenger | Exact Output Match | Token Similarity (SequenceMatcher) | "
            "Outcome Agreement | Pass-Rate Delta |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        for comp in parity["comparisons"]:
            lines.append(
                f"| `{comp['baseline']}` | `{comp['target']}` | "
                f"{comp['exact_match_count']}/{comp['probes_compared']} ({comp['exact_match_pct']:.1f}%) | "
                f"{comp['token_similarity_pct']:.1f}% | {comp['outcome_agreement_pct']:.1f}% | "
                f"{comp['pass_rate_delta_pct']:+.1f}% |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Target Execution Orchestrator & CLI
# ---------------------------------------------------------------------------


def parse_suites(spec: str) -> list[str]:
    items = [s.strip().lower() for s in spec.split(",") if s.strip()]
    if not items or "all" in items:
        return list(ALL_SUITES)
    unknown = [s for s in items if s not in ALL_SUITES]
    if unknown:
        raise ValueError(f"unknown suite(s): {unknown}; valid suites are all or {','.join(ALL_SUITES)}")
    # Preserve canonical execution order without duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for s in ALL_SUITES:
        if s in items and s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def parse_concurrency(spec: str) -> list[int]:
    vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals or any(v <= 0 for v in vals):
        raise ValueError(f"--concurrency must be comma-separated positive integers, got {spec!r}")
    return vals


def parse_kv_pair(raw: str, flag_name: str) -> tuple[str, str]:
    name, sep, val = raw.partition("=")
    if not sep or not name.strip() or not val.strip():
        raise ValueError(f"{flag_name} must be NAME=VALUE, got {raw!r}")
    return name.strip(), val.strip()


def load_corpus_file(path: str | Path) -> tuple[dict[str, Any], str]:
    p = Path(path)
    raw = p.read_bytes()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"corpus {p} must be a JSON object")
    return obj, fnv1a_64_hex(raw)


def load_managed_config(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    targets_raw: Any = raw.get("targets") if isinstance(raw, dict) and "targets" in raw else raw
    targets: list[dict[str, Any]] = []
    if isinstance(targets_raw, dict):
        for name, cfg in targets_raw.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"managed target {name!r} must be an object")
            entry = dict(cfg)
            entry.setdefault("name", name)
            targets.append(entry)
    elif isinstance(targets_raw, list):
        for idx, cfg in enumerate(targets_raw):
            if not isinstance(cfg, dict) or not cfg.get("name"):
                raise ValueError(f"managed targets[{idx}] must be an object with 'name'")
            targets.append(dict(cfg))
    else:
        raise ValueError("managed config must contain a 'targets' list or object")

    if not targets:
        raise ValueError("managed config defines zero targets")
    for t in targets:
        if not t.get("url") or not t.get("cmd"):
            raise ValueError(f"managed target {t.get('name')!r} requires both 'url' and 'cmd'")
    return targets


def execute_single_target(
    target_spec: dict[str, Any],
    corpus: dict[str, Any],
    *,
    suites: list[str],
    repeats: int,
    max_tokens: int,
    max_tokens_explicit: bool,
    concurrency_levels: list[int],
    disable_thinking: bool,
    keepalive_requests: int | None,
    timeout: float,
    log_fn: Any,
) -> dict[str, Any]:
    name = str(target_spec["name"])
    url = str(target_spec["url"])
    managed = bool(target_spec.get("cmd"))
    api_key = target_spec.get("api_key")
    headers = target_spec.get("headers")
    pid_hint: int | None = target_spec.get("pid")

    vram_samples: list[float] = []
    rss_samples: list[float] = []

    def record_resources(active_pid: int | None) -> None:
        v = sample_gpu_vram_mib()
        if v is not None:
            vram_samples.append(v)
        r = sample_process_rss_mib(active_pid)
        if r is not None:
            rss_samples.append(r)

    def run_against_live_server(active_pid: int | None, startup_ms: float | None) -> dict[str, Any]:
        client = BenchmarkClient(url, api_key=api_key, headers=headers, timeout=timeout)
        resolved_model = client.discover_model(target_spec.get("model"))
        log_fn(f"[{name}] benchmarking {url} (model={resolved_model!r}, suites={suites})")

        # Warm up 1 tiny request before timed measurements.
        warmup_res = client.stream_chat(
            [{"role": "user", "content": "Say OK."}],
            model=resolved_model,
            max_tokens=4,
            temperature=0.0,
            seed=42,
            disable_thinking=disable_thinking,
            timeout=min(timeout, 60.0),
        )
        record_resources(active_pid)

        t_suite0 = time.perf_counter()
        suites_out: dict[str, Any] = {}
        if "prefill" in suites:
            suites_out["prefill"] = run_prefill_suite(
                client,
                corpus,
                model=resolved_model,
                repeats=repeats,
                max_tokens=max_tokens,
                max_tokens_explicit=max_tokens_explicit,
                disable_thinking=disable_thinking,
            )
            record_resources(active_pid)
        if "prefix" in suites:
            suites_out["prefix"] = run_prefix_suite(
                client,
                corpus,
                model=resolved_model,
                repeats=repeats,
                max_tokens=max_tokens,
                max_tokens_explicit=max_tokens_explicit,
                disable_thinking=disable_thinking,
            )
            record_resources(active_pid)
        if "decode" in suites:
            suites_out["decode"] = run_decode_suite(
                client,
                corpus,
                model=resolved_model,
                repeats=repeats,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
            record_resources(active_pid)
        if "concurrency" in suites:
            suites_out["concurrency"] = run_concurrency_suite(
                client,
                corpus,
                model=resolved_model,
                levels=concurrency_levels,
                repeats=repeats,
                max_tokens=max_tokens,
                max_tokens_explicit=max_tokens_explicit,
                disable_thinking=disable_thinking,
            )
            record_resources(active_pid)
        if "keepalive" in suites:
            suites_out["keepalive"] = run_keepalive_suite(
                client,
                corpus,
                num_requests=keepalive_requests,
            )
        if "endpoints" in suites:
            suites_out["endpoints"] = run_endpoints_suite(
                client,
                model=resolved_model,
                disable_thinking=disable_thinking,
            )
        if "quality" in suites:
            suites_out["quality"] = run_quality_suite(
                client,
                corpus,
                model=resolved_model,
                max_tokens=max_tokens,
                max_tokens_explicit=max_tokens_explicit,
                disable_thinking=disable_thinking,
            )
            record_resources(active_pid)

        wall_s = time.perf_counter() - t_suite0
        return {
            "name": name,
            "mode": "managed" if managed else "attach",
            "url": url,
            "model": resolved_model,
            "startup_ms": startup_ms,
            "warmup_ok": bool(warmup_res.get("ok")),
            "wall_s": wall_s,
            "resources": {
                "pid": active_pid,
                "vram_samples_mib": vram_samples,
                "peak_vram_mib": max(vram_samples) if vram_samples else None,
                "rss_samples_mib": rss_samples,
                "peak_rss_mib": max(rss_samples) if rss_samples else None,
            },
            "suites": suites_out,
        }

    if managed:
        with ManagedServer(target_spec, log_fn=log_fn) as srv:
            return run_against_live_server(srv.pid, srv.startup_ms)
    return run_against_live_server(pid_hint, None)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bench_prism_vs_sonder.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="Attach mode target, repeatable (e.g., --target prism=http://127.0.0.1:18080)",
    )
    ap.add_argument("--model", default=None, help="Default model ID across targets")
    ap.add_argument(
        "--target-model",
        action="append",
        default=[],
        metavar="NAME=MODEL",
        help="Per-target model override (repeatable)",
    )
    ap.add_argument(
        "--target-pid",
        action="append",
        default=[],
        metavar="NAME=PID",
        help="Optional per-target PID for RSS sampling in attach mode",
    )
    ap.add_argument(
        "--config",
        default=None,
        metavar="JSON_PATH",
        help="Managed sequential mode config JSON (launches ONE server at a time)",
    )
    ap.add_argument(
        "--corpus",
        default=str(DEFAULT_CORPUS_PATH),
        help="Path to head-to-head workload & quality corpus JSON",
    )
    ap.add_argument(
        "--suites",
        default="all",
        help="all or comma-separated subset of: prefill,prefix,decode,concurrency,keepalive,endpoints,quality",
    )
    ap.add_argument("--repeats", type=int, default=2, help="Timed repeats per benchmark case (default: 2)")
    ap.add_argument("--max-tokens", type=int, default=None, help="Max generated tokens (default: 256)")
    ap.add_argument("--concurrency", default="1,2,4", help="Comma-separated concurrency levels (default: 1,2,4)")
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help='Send chat_template_kwargs={"enable_thinking":false} for apples-to-apples non-thinking runs',
    )
    ap.add_argument(
        "--keepalive-requests",
        type=int,
        default=None,
        help="Number of requests per mode in keepalive suite (default: 10 from corpus)",
    )
    ap.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds (default: 300)")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_HTTP_API_KEY"), help="Optional Bearer token")
    ap.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="K=V",
        help="Extra HTTP header (repeatable)",
    )
    ap.add_argument("--output-json", default=None, help="Write structured benchmark report JSON to PATH")
    ap.add_argument("--output-md", default=None, help="Write side-by-side Markdown report to PATH")
    ap.add_argument("--quiet", action="store_true", help="Suppress stderr progress messages")
    return ap


def run_benchmark_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.config) == bool(args.target):
        raise ValueError("specify exactly one of --config <json> (managed mode) or --target NAME=URL (attach mode)")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    max_tokens_explicit = args.max_tokens is not None
    effective_max_tokens = int(args.max_tokens) if max_tokens_explicit else 256
    if effective_max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1")

    suites = parse_suites(args.suites)
    concurrency_levels = parse_concurrency(args.concurrency)

    cli_headers: dict[str, str] = {}
    for h_raw in args.header or []:
        hk, hv = parse_kv_pair(h_raw, "--header")
        cli_headers[hk] = hv

    per_target_models: dict[str, str] = {}
    for tm_raw in args.target_model or []:
        tname, tmod = parse_kv_pair(tm_raw, "--target-model")
        per_target_models[tname] = tmod

    per_target_pids: dict[str, int] = {}
    for tp_raw in args.target_pid or []:
        tname, tpid = parse_kv_pair(tp_raw, "--target-pid")
        per_target_pids[tname] = int(tpid)

    target_specs: list[dict[str, Any]] = []
    if args.config:
        mode = "managed"
        for entry in load_managed_config(args.config):
            spec = dict(entry)
            tname = str(spec["name"])
            if tname in per_target_models:
                spec["model"] = per_target_models[tname]
            elif not spec.get("model") and args.model:
                spec["model"] = args.model
            merged_hdrs = dict(cli_headers)
            merged_hdrs.update(spec.get("headers") or {})
            spec["headers"] = merged_hdrs
            if not spec.get("api_key") and args.api_key:
                spec["api_key"] = args.api_key
            target_specs.append(spec)
    else:
        mode = "attach"
        seen_names: set[str] = set()
        for t_raw in args.target:
            tname, turl = parse_kv_pair(t_raw, "--target")
            if tname in seen_names:
                raise ValueError(f"duplicate target name {tname!r}")
            seen_names.add(tname)
            target_specs.append({
                "name": tname,
                "url": turl,
                "model": per_target_models.get(tname, args.model),
                "pid": per_target_pids.get(tname),
                "api_key": args.api_key,
                "headers": dict(cli_headers),
            })

    corpus, corpus_fnv1a = load_corpus_file(args.corpus)
    log_fn = (lambda _m: None) if args.quiet else (lambda m: print(m, file=sys.stderr, flush=True))

    target_results: list[dict[str, Any]] = []
    for t_spec in target_specs:
        res = execute_single_target(
            t_spec,
            corpus,
            suites=suites,
            repeats=args.repeats,
            max_tokens=effective_max_tokens,
            max_tokens_explicit=max_tokens_explicit,
            concurrency_levels=concurrency_levels,
            disable_thinking=bool(args.disable_thinking),
            keepalive_requests=args.keepalive_requests,
            timeout=float(args.timeout),
            log_fn=log_fn,
        )
        target_results.append(res)

    parity = compute_cross_target_parity(target_results)
    report: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "corpus": {
            "path": str(args.corpus),
            "name": corpus.get("name"),
            "version": corpus.get("version"),
            "fnv1a": corpus_fnv1a,
        },
        "config": {
            "mode": mode,
            "suites": suites,
            "repeats": args.repeats,
            "max_tokens": effective_max_tokens,
            "max_tokens_explicit": max_tokens_explicit,
            "concurrency": concurrency_levels,
            "disable_thinking": bool(args.disable_thinking),
        },
        "targets": target_results,
        "cross_target_parity": parity,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark_from_args(args)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    md = render_markdown_report(report)
    if args.output_json:
        out_j = Path(args.output_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        out_j.write_bytes((json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    if args.output_md:
        out_m = Path(args.output_md)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        out_m.write_bytes(md.encode("utf-8"))

    if not args.quiet or (not args.output_json and not args.output_md):
        print(md, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
