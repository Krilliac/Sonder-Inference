#!/usr/bin/env python3
"""Reliability run driver for an OpenAI-compatible endpoint (stdlib only).

Drives POST /v1/chat/completions (streaming, temperature 0) with fixed prompts
for a fixed duration and records:

  requests.jsonl  one row per request: TTFT, decode tok/s, token counts and
                  where they came from, output hash, errors
  samples.jsonl   host/process metrics at a fixed interval (default 1 Hz)
  events.jsonl    scheduled fault injections that ran, restarts observed
  summary.json    totals, percentiles, criteria and the verdict

Evidence rules (docs/BENCHMARK_PLAN.md "Evidence rules"):
  * A run that completed zero requests is never a success.
  * A run whose host sampler produced nothing is never a success.
  * A scheduled fault that never fired fails the run (the N > 0 gate).
  * Output hashes at temperature 0 are compared per prompt; a prompt whose
    hash changes is reported as a correctness failure, not hidden.

Host metrics: Windows via ctypes (GlobalMemoryStatusEx, GetProcessMemoryInfo,
GetProcessIoCounters); Linux via /proc. VRAM comes from nvidia-smi when it is on
PATH, otherwise it is recorded as null. A value that cannot be read is null,
never zero.

    python scripts/reliability_run.py --base-url http://127.0.0.1:8080 \
        --model m --duration 3600 --pid-file server.pid --out-dir run1
    python scripts/reliability_run.py --self-test
"""
import argparse
import hashlib
import http.client
import http.server
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

DEFAULT_PROMPTS = [
    "Count from one to twelve in English words, separated by commas.",
    "List the planets of the solar system in order from the sun.",
    "Write the first ten Fibonacci numbers separated by spaces.",
    "Spell the word 'reliability' one letter per line.",
]

IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# Host and process metrics
# ---------------------------------------------------------------------------
if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

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

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
    _k32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    _k32.GetProcessIoCounters.argtypes = [wintypes.HANDLE, ctypes.POINTER(_IO_COUNTERS)]
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]

    _PROCESS_QUERY_INFORMATION = 0x0400
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _PROCESS_VM_READ = 0x0010
    _STILL_ACTIVE = 259


def host_memory():
    """Returns dict(avail_bytes, commit_used_bytes, commit_limit_bytes); None per unreadable field."""
    out = {"avail_bytes": None, "commit_used_bytes": None, "commit_limit_bytes": None}
    if IS_WINDOWS:
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if _k32.GlobalMemoryStatusEx(ctypes.byref(status)):
            out["avail_bytes"] = int(status.ullAvailPhys)
            out["commit_limit_bytes"] = int(status.ullTotalPageFile)
            out["commit_used_bytes"] = int(status.ullTotalPageFile - status.ullAvailPageFile)
        return out
    try:
        fields = {}
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    fields[name] = int(parts[0]) * 1024
        out["avail_bytes"] = fields.get("MemAvailable")
        out["commit_limit_bytes"] = fields.get("CommitLimit")
        out["commit_used_bytes"] = fields.get("Committed_AS")
    except (OSError, ValueError):
        pass
    return out


def process_counters(pid):
    """Returns dict(alive, private_bytes, working_set_bytes, process_read_bytes_total); None per unreadable field.

    process_read_bytes_total is OS-specific: Windows ReadTransferCount = all I/O reads of the process
    (including sockets); Linux /proc/<pid>/io read_bytes = bytes read from the storage layer.
    private_bytes is also OS-specific: Windows PrivateUsage (commit charge), Linux resident private pages."""
    out = {"alive": False, "private_bytes": None, "working_set_bytes": None, "process_read_bytes_total": None}
    if pid is None or pid <= 0:
        return out
    if IS_WINDOWS:
        handle = _k32.OpenProcess(_PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, pid)
        if not handle:
            handle = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return out
        try:
            code = wintypes.DWORD(0)
            if not _k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != _STILL_ACTIVE:
                return out
            out["alive"] = True
            counters = _PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS_EX)
            if _k32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                out["private_bytes"] = int(counters.PrivateUsage)
                out["working_set_bytes"] = int(counters.WorkingSetSize)
            # GetProcessIoCounters counts ALL I/O the process performs (files, pipes, sockets,
            # devices), so this is NOT disk traffic: an HTTP server's socket reads land here.
            io = _IO_COUNTERS()
            if _k32.GetProcessIoCounters(handle, ctypes.byref(io)):
                out["process_read_bytes_total"] = int(io.ReadTransferCount)
        finally:
            _k32.CloseHandle(handle)
        return out
    try:
        with open("/proc/%d/stat" % pid, "r", encoding="ascii", errors="replace") as handle:
            text = handle.read()
        state = text[text.rfind(")") + 2:text.rfind(")") + 3]
        if state in ("Z", "X"):
            return out
        out["alive"] = True
    except OSError:
        return out
    try:
        rss_anon = None
        with open("/proc/%d/status" % pid, "r", encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    out["working_set_bytes"] = int(line.split()[1]) * 1024
                elif line.startswith("RssAnon:"):
                    rss_anon = int(line.split()[1]) * 1024
        out["private_bytes"] = rss_anon
        try:
            private = 0
            with open("/proc/%d/smaps_rollup" % pid, "r", encoding="ascii", errors="replace") as handle:
                for line in handle:
                    if line.startswith(("Private_Clean:", "Private_Dirty:")):
                        private += int(line.split()[1]) * 1024
            out["private_bytes"] = private
        except OSError:
            pass
    except (OSError, ValueError):
        pass
    try:
        with open("/proc/%d/io" % pid, "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("read_bytes:"):  # bytes fetched from the storage layer
                    out["process_read_bytes_total"] = int(line.split()[1])
    except (OSError, ValueError):
        pass
    return out


def vram_used_mib():
    """Sum of used VRAM across GPUs from nvidia-smi, or None when unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        run = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3)
        if run.returncode != 0:
            return None
        values = [float(line.strip()) for line in run.stdout.splitlines() if line.strip()]
        return sum(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def read_pid_file(path):
    try:
        with open(path, "r", encoding="ascii") as handle:
            return int(handle.read().strip().split()[0])
    except (OSError, ValueError, IndexError):
        return None


class Sampler(threading.Thread):
    """Samples host and process metrics every `interval` seconds on a fixed schedule."""

    def __init__(self, out_path, interval, pids, pid_file, t0):
        super().__init__(name="sampler", daemon=True)
        self.interval = interval
        self.pids = list(pids)
        self.pid_file = pid_file
        self.t0 = t0
        self.stop_event = threading.Event()
        self.out = open(out_path, "w", encoding="utf-8", newline="\n")
        self.rows = []
        self.restarts = 0
        self.pid_missing_samples = 0
        self._last_file_pid = None
        self._last_read = {}

    def run(self):
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            self.sample_once()
            next_tick += self.interval
            delay = next_tick - time.monotonic()
            if delay < 0:
                next_tick = time.monotonic()
                delay = 0
            self.stop_event.wait(delay)

    def sample_once(self):
        # Every relative value and rate uses the monotonic clock (self.t0 is a time.monotonic() origin), so a
        # wall-clock step cannot reorder or negate them. time.time() is kept only for the absolute "t" field.
        wall = time.time()
        now = time.monotonic()
        targets = list(self.pids)
        if self.pid_file:
            file_pid = read_pid_file(self.pid_file)
            if file_pid is not None:
                if self._last_file_pid is not None and file_pid != self._last_file_pid:
                    self.restarts += 1
                self._last_file_pid = file_pid
                targets.append(file_pid)
            else:
                self.pid_missing_samples += 1
        row = {"t": round(wall, 3), "t_rel": round(now - self.t0, 3), "host": host_memory(),
               "vram_used_mib": vram_used_mib(), "processes": []}
        for pid in targets:
            counters = process_counters(pid)
            counters["pid"] = pid
            previous = self._last_read.get(pid)
            total = counters["process_read_bytes_total"]
            counters["process_read_bytes_per_s"] = None
            if total is not None and previous is not None and now > previous[0] and total >= previous[1]:
                counters["process_read_bytes_per_s"] = round((total - previous[1]) / (now - previous[0]), 1)
            if total is not None:
                self._last_read[pid] = (now, total)
            if not counters["alive"]:
                self.pid_missing_samples += 1
            row["processes"].append(counters)
        row["restarts"] = self.restarts
        self.rows.append(row)
        self.out.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.out.flush()

    def stop(self):
        self.stop_event.set()
        self.join(timeout=self.interval + 5)
        self.sample_once()  # closing sample so a short run always has a final point
        self.out.close()


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
def run_request(base_url, model, prompt, max_tokens, timeout, headers):
    """One streaming chat request. Never raises; returns a result dict.

    `timeout` is a total deadline for the whole request, not just a per-socket-operation timeout.
    It is exact over the response body: the socket timeout is shrunk to the time remaining before every
    body read, so a server that trickles bytes (or never finishes a line) still fails the request.
    Before the body (send, status line and headers) each socket operation is bounded by the time
    remaining, so only a server that trickles header bytes can overrun by up to the per-operation wait.
    The base URL may or may not end in /v1. Header values never appear in error text.
    """
    parts = urlsplit(base_url)
    base_path = parts.path.rstrip("/")
    if base_path.endswith("/v1"):
        base_path = base_path[:-3]
    result = {"ok": False, "error": None, "http_status": None, "ttft_s": None, "total_s": None,
              "prompt_tokens": None, "completion_tokens": None, "token_count_source": None,
              "decode_tps": None, "finish_reason": None, "output_sha256": None}
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
        "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
    })
    request_headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    request_headers.update(headers)
    connection_class = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = None
    timed_out = threading.Event()
    pending = bytearray()
    start = time.perf_counter()
    first = None
    last = None
    chunks = 0
    saw_done = False
    usage = None
    text = []
    try:
        conn = connection_class(parts.hostname, parts.port, timeout=timeout)
        conn.connect()
        sock = conn.sock  # http.client drops conn.sock for close-delimited responses

        def arm_deadline():
            remaining_now = timeout - (time.perf_counter() - start)
            if remaining_now <= 0:
                timed_out.set()
                raise socket.timeout("deadline")
            sock.settimeout(remaining_now)

        arm_deadline()
        conn.request("POST", base_path + "/v1/chat/completions", body, request_headers)
        arm_deadline()
        response = conn.getresponse()
        result["http_status"] = response.status
        if time.perf_counter() - start >= timeout:
            timed_out.set()
            result["total_s"] = round(time.perf_counter() - start, 4)
            result["error"] = "deadline_exceeded: request outlasted %.1f s" % timeout
            return result
        if response.status != 200:
            arm_deadline()
            detail = response.read(512).decode("utf-8", "replace")
            result["error"] = "http_%d: %s" % (response.status, detail.strip()[:200])
            return result
        while True:
            remaining = timeout - (time.perf_counter() - start)
            if remaining <= 0 or timed_out.is_set():
                timed_out.set()
                break
            newline = pending.find(b"\n")
            if newline < 0:
                sock.settimeout(remaining)
                chunk = response.read1(65536)  # returns after one socket read, so the deadline is rechecked
                if not chunk:
                    break
                pending += chunk
                continue
            raw = bytes(pending[:newline + 1])
            del pending[:newline + 1]
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            try:
                event = json.loads(payload)
            except ValueError:
                result["error"] = "malformed_sse_json"
                return result
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    chunks += 1
                    text.append(piece)
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
        end = time.perf_counter()
        result["total_s"] = round(end - start, 4)
        if not saw_done and timed_out.is_set():
            result["error"] = "deadline_exceeded: request outlasted %.1f s" % timeout
            return result
        if not saw_done:
            result["error"] = "truncated_stream"
            return result
        if chunks == 0:
            result["error"] = "empty_completion"
            return result
        result["ttft_s"] = round(first - start, 4)
        if usage and isinstance(usage.get("completion_tokens"), int):
            result["completion_tokens"] = usage["completion_tokens"]
            result["token_count_source"] = "usage"
            if isinstance(usage.get("prompt_tokens"), int):
                result["prompt_tokens"] = usage["prompt_tokens"]
        else:
            result["completion_tokens"] = chunks
            result["token_count_source"] = "chunks"
        if result["completion_tokens"] > 1 and last is not None and last > first:
            result["decode_tps"] = round((result["completion_tokens"] - 1) / (last - first), 3)
        result["output_sha256"] = hashlib.sha256("".join(text).encode("utf-8")).hexdigest()
        result["ok"] = True
        return result
    except ValueError:
        # http.client rejects illegal header names/values with a message that quotes the value
        # (it may be a bearer token), so the text is deliberately not recorded.
        result["error"] = "ValueError: invalid request (message withheld; check --header)"
        result["total_s"] = round(time.perf_counter() - start, 4)
        return result
    except (OSError, http.client.HTTPException) as exc:
        if (timed_out.is_set() or isinstance(exc, socket.timeout)
                or time.perf_counter() - start >= timeout):
            result["error"] = "deadline_exceeded: request outlasted %.1f s" % timeout
        else:
            result["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
        result["total_s"] = round(time.perf_counter() - start, 4)
        return result
    finally:
        if conn is not None:
            conn.close()


PROCESS_READ_KIND = "all_io_reads" if IS_WINDOWS else "storage_reads"


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int(-(-q * len(ordered) // 100)))  # nearest rank, ceil
    return ordered[min(rank, len(ordered)) - 1]


def slope_per_hour(points):
    """Least-squares slope of (seconds, value) in value-units per hour, or None."""
    if len(points) < 3:
        return None
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    denominator = sum((p[0] - mean_x) ** 2 for p in points)
    if denominator == 0:
        return None
    return sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / denominator * 3600.0


def growth_after_warmup(points, warmup_s):
    """slope_per_hour over the points taken at or after warmup_s seconds into the run."""
    return slope_per_hour([p for p in points if p[0] >= warmup_s])


def in_fault_window(row, windows, grace_s):
    """True when the request overlapped a fault window [start, end + grace_s]."""
    start = row["t_rel"]
    end = row.get("end_rel", start)
    return any(start <= w_end + grace_s and end >= w_start for w_start, w_end in windows)


def unrecovered_faults(windows, rows, budget_s):
    """Fault windows with no successful request that started after the fault and finished within budget_s."""
    missing = []
    for w_start, w_end in windows:
        recovered = any(r["ok"] and r["t_rel"] >= w_end and r.get("end_rel", r["t_rel"]) <= w_end + budget_s
                        for r in rows)
        if not recovered:
            missing.append(w_start)
    return missing


def expected_sample_count(duration_s, elapsed_s, interval_s):
    """Fewest host samples a run must have: 80 % of the sampling ticks in the time it ran, never under 1.

    duration_s is a ceiling: a run ended early only owes samples for elapsed_s, and a run that overran
    (a last request still in flight) owes none for the overrun."""
    return max(1, int(min(duration_s, elapsed_s) / interval_s * 0.8))


def host_samples_ok(sample_count, duration_s, elapsed_s, interval_s):
    """The host_samples criterion: (passed, minimum sample count)."""
    expected = expected_sample_count(duration_s, elapsed_s, interval_s)
    return sample_count >= expected, expected


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def parse_fault(text):
    seconds, sep, command = text.partition(":")
    if not sep or not command.strip():
        raise ValueError("--fault must look like SECONDS:COMMAND, got %r" % text)
    return float(seconds), command.strip()


def parse_headers(items, env_items, environ):
    """Builds the request header dict from --header K=V and --header-env K=ENVVAR items.

    Raises ValueError on a malformed item. The message never contains a header value or the raw item
    (it may be a bearer token); an environment variable NAME may be shown, never its value."""
    headers = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError("--header must look like K=V (missing '=' or empty name)")
        headers[key.strip()] = value
    for item in env_items or []:
        key, sep, var = item.partition("=")
        if not sep or not key.strip() or not var.strip():
            raise ValueError("--header-env must look like K=ENVVAR (missing '=' or empty name)")
        value = environ.get(var.strip())
        if not value:
            raise ValueError("--header-env %s: environment variable %s is not set or empty"
                             % (key.strip(), var.strip()))
        headers[key.strip()] = value
    return headers


def run(args, stop_event=None, seams=None):
    """Executes one run. Returns (summary, exit_code); exit_code 0 only on success.

    stop_event (threading.Event, optional) ends the request loops early; args.duration is then only a ceiling.
    The self-test uses it to end a run on an observed event instead of a wall-clock guess.

    seams (dict, optional) is for the self-test only. "sampler_class" replaces Sampler. Any of the names
    "fault_command_returned", "fault_window_recorded", "fault_event_logged", "request_stamped" and
    "request_returned" maps to a callable run at that point of the fault or worker thread, so a test can
    stall the thread there. A normal run passes nothing and behaves exactly as without seams."""
    seams = seams or {}

    def seam(name):
        callback = seams.get(name)
        if callback is not None:
            callback()

    prompts = list(DEFAULT_PROMPTS)
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as handle:
            prompts = [line.rstrip("\n") for line in handle if line.strip()]
        if not prompts:
            raise SystemExit("prompts file is empty")
    os.makedirs(args.out_dir, exist_ok=True)
    headers = parse_headers(args.header, getattr(args, "header_env", None), os.environ)
    faults = sorted(parse_fault(item) for item in (args.fault or []))

    t0 = time.monotonic()  # single origin for every relative timestamp below, the sampler's included
    sampler = seams.get("sampler_class", Sampler)(os.path.join(args.out_dir, "samples.jsonl"),
                                                  args.sample_interval, args.pid or [], args.pid_file, t0)
    sampler.start()
    deadline = time.monotonic() + args.duration
    lock = threading.Lock()
    results = []
    counter = [0]
    requests_out = open(os.path.join(args.out_dir, "requests.jsonl"), "w", encoding="utf-8", newline="\n")
    events_out = open(os.path.join(args.out_dir, "events.jsonl"), "w", encoding="utf-8", newline="\n")
    faults_fired = [0]
    # (start_rel, end_rel) of each fault command that exited 0, rounded like request rows so that
    # events.jsonl, summary.json and the criteria all compare the same numbers.
    fault_windows = []

    def log_event(kind, **fields):
        row = {"t_rel": round(time.monotonic() - t0, 3), "event": kind}
        row.update(fields)
        with lock:
            events_out.write(json.dumps(row, separators=(",", ":")) + "\n")
            events_out.flush()

    def stopped():
        return time.monotonic() >= deadline or (stop_event is not None and stop_event.is_set())

    def worker():
        while not stopped():
            with lock:
                index = counter[0]
                counter[0] += 1
            prompt_id = index % len(prompts)
            sent = time.monotonic()
            seam("request_stamped")
            result = run_request(args.base_url, args.model, prompts[prompt_id], args.max_tokens,
                                 args.request_timeout, headers)
            seam("request_returned")
            result.update({"index": index, "prompt_id": prompt_id, "t_rel": round(sent - t0, 3),
                           "end_rel": round(time.monotonic() - t0, 3)})
            with lock:
                results.append(result)
                requests_out.write(json.dumps(result, separators=(",", ":")) + "\n")
                requests_out.flush()
            if not result["ok"]:
                # Back off briefly so a dead endpoint is not hammered in a tight loop.
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))

    def fault_runner():
        for offset, command in faults:
            wait = offset - (time.monotonic() - t0)
            if wait > 0 and wait > args.duration:
                return  # scheduled past the end: it never fires, the gate will fail the run
            end = time.monotonic() + max(0.0, wait)
            while time.monotonic() < end:
                time.sleep(0.05)
            if time.monotonic() >= deadline:
                return
            try:
                fault_start = round(time.monotonic() - t0, 3)
                run_cmd = subprocess.run(shlex.split(command), capture_output=True,
                                         text=True, timeout=60)
                seam("fault_command_returned")
                fault_end = round(time.monotonic() - t0, 3)
                # Record first, log second. The event row carries the window, so whoever reads the row
                # sees exactly the window the criteria will use, however long this thread is stalled
                # before, between or after these statements.
                if run_cmd.returncode == 0:
                    with lock:
                        fault_windows.append((fault_start, fault_end))
                        faults_fired[0] += 1
                seam("fault_window_recorded")
                log_event("fault", command=command, returncode=run_cmd.returncode,
                          start_rel=fault_start, end_rel=fault_end)
                seam("fault_event_logged")
            except (OSError, subprocess.SubprocessError) as exc:
                log_event("fault_error", command=command, error=str(exc)[:200])

    threads = [threading.Thread(target=worker, name="worker-%d" % i, daemon=True)
               for i in range(max(1, args.concurrency))]
    fault_thread = threading.Thread(target=fault_runner, name="faults", daemon=True)
    for thread in threads:
        thread.start()
    fault_thread.start()
    for thread in threads:
        thread.join()
    fault_thread.join(timeout=70)
    sampler.stop()
    requests_out.close()
    elapsed = time.monotonic() - t0

    completed = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    expected_failed = [r for r in failed if in_fault_window(r, fault_windows, args.fault_window)]
    unexpected_failed = [r for r in failed if not in_fault_window(r, fault_windows, args.fault_window)]
    errors = {}
    for row in failed:
        kind = (row["error"] or "unknown").split(":")[0]
        errors[kind] = errors.get(kind, 0) + 1
    ttfts = [r["ttft_s"] for r in completed]
    tps = [r["decode_tps"] for r in completed if r["decode_tps"] is not None]
    hashes = {}
    for row in completed:
        hashes.setdefault(row["prompt_id"], set()).add(row["output_sha256"])
    nondeterministic = sorted(pid for pid, seen in hashes.items() if len(seen) > 1)

    avail = [s["host"]["avail_bytes"] for s in sampler.rows if s["host"]["avail_bytes"] is not None]
    commit = [s["host"]["commit_used_bytes"] for s in sampler.rows if s["host"]["commit_used_bytes"] is not None]
    vram = [s["vram_used_mib"] for s in sampler.rows if s["vram_used_mib"] is not None]
    private_series = {}
    for row in sampler.rows:
        for proc in row["processes"]:
            if proc["private_bytes"] is not None:
                private_series.setdefault(proc["pid"], []).append((row["t_rel"], proc["private_bytes"]))
    growth = {}
    for pid, points in private_series.items():
        value = growth_after_warmup(points, args.warmup_s)
        growth[str(pid)] = None if value is None else round(value / (1024 * 1024), 3)
    read_rates = [p["process_read_bytes_per_s"] for s in sampler.rows for p in s["processes"]
                  if p["process_read_bytes_per_s"] is not None]

    # A run ended early by stop_event only owes samples for the time it actually ran (duration is a ceiling).
    samples_ok, expected_samples = host_samples_ok(len(sampler.rows), args.duration, elapsed, args.sample_interval)
    criteria = []

    def criterion(name, ok, detail):
        criteria.append({"name": name, "ok": bool(ok), "detail": detail})

    criterion("requests_completed", len(completed) > 0,
              "%d completed of %d attempted (zero completed is never a success)" % (len(completed), len(results)))
    outside_total = len(results) - len(expected_failed)
    error_rate = (len(unexpected_failed) / outside_total) if outside_total else 1.0
    criterion("error_rate", error_rate <= args.max_error_rate,
              "%.4f observed outside fault windows (%d failures of %d requests; %d failures inside fault "
              "windows [fault start, end + %.1fs] are expected and listed in expected_fault_failures), "
              "limit %.4f" % (error_rate, len(unexpected_failed), outside_total, len(expected_failed),
                              args.fault_window, args.max_error_rate))
    if args.require_usage_tokens:
        token_sources = sorted({r["token_count_source"] for r in completed})
        criterion("exact_token_counts", bool(completed) and token_sources == ["usage"],
                  "token count sources over completed requests: %s (must be exactly ['usage']; 'chunks' is an "
                  "approximation of one token per streamed chunk)" % token_sources)
    criterion("host_samples", samples_ok,
              "%d samples, expected at least %d at %.2fs interval" % (len(sampler.rows), expected_samples,
                                                                        args.sample_interval))
    if faults:
        criterion("faults_fired", faults_fired[0] == len(faults),
                  "%d of %d scheduled faults fired (exit 0 only; without --pid-file/--expected-restarts "
                  "this does not prove the fault had any effect)" % (faults_fired[0], len(faults)))
        missing = unrecovered_faults(fault_windows, results, args.recovery_budget_s)
        criterion("fault_recovery", not missing,
                  "faults at t=%s s with no successful request started after the fault and finished within "
                  "%.0f s of it: %s" % ([round(w[0], 1) for w in fault_windows], args.recovery_budget_s,
                                        [round(m, 1) for m in missing] or "none"))
    criterion("deterministic_output", args.allow_nondeterminism or not nondeterministic,
              "prompts with changing output hash at temperature 0: %s" % (nondeterministic or "none"))
    if args.max_ttft_p95 is not None:
        p95 = percentile(ttfts, 95)
        criterion("ttft_p95", p95 is not None and p95 <= args.max_ttft_p95,
                  "p95 %s s, limit %s s" % (p95, args.max_ttft_p95))
    if args.pid_file:
        expected = args.expected_restarts or 0
        criterion("restarts_explained", sampler.restarts == expected,
                  "%d pid changes observed in --pid-file, schedule expects exactly %d (--expected-restarts)"
                  % (sampler.restarts, expected))
    if args.min_available_mib is not None:
        criterion("host_available_floor", bool(avail) and min(avail) >= args.min_available_mib * 1048576,
                  "minimum available %s MiB, floor %s MiB" % (
                      round(min(avail) / 1048576, 1) if avail else None, args.min_available_mib))
    if args.min_commit_headroom_mib is not None:
        headroom = [s["host"]["commit_limit_bytes"] - s["host"]["commit_used_bytes"] for s in sampler.rows
                    if s["host"]["commit_limit_bytes"] is not None and s["host"]["commit_used_bytes"] is not None]
        criterion("host_commit_headroom_floor",
                  bool(headroom) and min(headroom) >= args.min_commit_headroom_mib * 1048576,
                  "minimum commit headroom %s MiB, floor %s MiB" % (
                      round(min(headroom) / 1048576, 1) if headroom else None, args.min_commit_headroom_mib))
    if args.max_restarts is not None:
        criterion("restarts", sampler.restarts <= args.max_restarts,
                  "%d observed, limit %d" % (sampler.restarts, args.max_restarts))
    if args.max_private_growth_mib_per_hour is not None:
        observed = [v for v in growth.values() if v is not None]
        criterion("private_bytes_growth",
                  bool(observed) and max(observed) <= args.max_private_growth_mib_per_hour,
                  "slopes MiB/h after the first %.0f s %s, limit %s (needs at least 3 samples after warm-up)"
                  % (args.warmup_s, growth, args.max_private_growth_mib_per_hour))

    success = all(item["ok"] for item in criteria)
    summary = {
        "success": success,
        "failed_criteria": [item["name"] for item in criteria if not item["ok"]],
        "criteria": criteria,
        "base_url": args.base_url, "model": args.model,
        "duration_requested_s": args.duration, "duration_actual_s": round(elapsed, 3),
        "platform": platform.platform(),
        "requests": {"attempted": len(results), "completed": len(completed), "failed": len(failed),
                     "errors_by_kind": errors, "failed_outside_fault_windows": len(unexpected_failed),
                     "expected_fault_failures": [
                         {"index": r["index"], "t_rel": r["t_rel"], "error": r["error"]} for r in expected_failed]},
        "tokens": {"completion_total": sum(r["completion_tokens"] for r in completed),
                   "prompt_total": sum(r["prompt_tokens"] for r in completed if r["prompt_tokens"] is not None),
                   "sources": sorted({r["token_count_source"] for r in completed})},
        "ttft_s": {"p50": percentile(ttfts, 50), "p95": percentile(ttfts, 95), "max": max(ttfts) if ttfts else None},
        "decode_tps": {"p50": percentile(tps, 50), "mean": round(sum(tps) / len(tps), 3) if tps else None},
        "correctness": {"nondeterministic_prompt_ids": nondeterministic,
                        "output_hashes_per_prompt": {str(k): sorted(v) for k, v in sorted(hashes.items())}},
        "host": {"samples": len(sampler.rows), "samples_expected_min": expected_samples,
                 "sample_interval_s": args.sample_interval,
                 "min_available_bytes": min(avail) if avail else None,
                 "max_commit_used_bytes": max(commit) if commit else None,
                 "max_vram_used_mib": max(vram) if vram else None,
                 "max_process_read_bytes_per_s": max(read_rates) if read_rates else None,
                 "process_read_bytes_kind": PROCESS_READ_KIND,
                 "private_bytes_warmup_s": args.warmup_s,
                 "private_bytes_growth_mib_per_hour": growth,
                 "restarts_observed": sampler.restarts,
                 "pid_missing_samples": sampler.pid_missing_samples},
        "faults": {"scheduled": len(faults), "fired": faults_fired[0],
                   "windows_s": [[round(a, 3), round(b, 3)] for a, b in fault_windows]},
        "out_dir": os.path.abspath(args.out_dir),
    }
    events_out.close()
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary, (0 if success else 2)


# ---------------------------------------------------------------------------
# Built-in fake server and self-test
# ---------------------------------------------------------------------------
class _FakeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # body ends at close; no chunk framing needed

    def log_message(self, fmt, *args):  # silence
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length) or b"{}"
        if self.path.startswith("/__flaky/"):
            # Event-driven failure control (no clock): begin/end switch failing on and off, arm does nothing
            # (the test harness switches failing itself), wait blocks until a request has been failed,
            # ok_wait blocks until a request has been answered in full. The 50 s limits only stop a broken
            # test from hanging: when one is hit the reply is 504, the fault command exits non-zero, the
            # fault does not count as fired and the control fails.
            action = self.path[len("/__flaky/"):]
            status = 200
            if action == "begin":
                self.server.failing = True
            elif action == "end":
                self.server.failing = False
            elif action == "wait":
                give_up = time.monotonic() + 50.0
                while self.server.failed_served == 0 and time.monotonic() < give_up:
                    time.sleep(0.01)
                if self.server.failed_served == 0:
                    status = 504
            elif action == "ok_wait":
                give_up = time.monotonic() + 50.0
                while self.server.ok_served == 0 and time.monotonic() < give_up:
                    time.sleep(0.01)
                if self.server.ok_served == 0:
                    status = 504
            elif action != "arm":
                status = 404
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.loads(raw_body)
        mode = self.server.mode
        if self.server.failing:
            mode = "fail"
            self.server.failed_served += 1
        self.server.hits += 1
        if mode == "slow_headers":  # never replies: holds the connection until the client closes it
            try:
                self.connection.settimeout(60.0)  # only so a broken client cannot hang the test
                self.connection.recv(1)
            except socket.timeout:
                self.server.gave_up += 1  # the client never closed: its deadline did not work
            except OSError:
                pass
            return
        if mode == "trickle":  # headers, then one byte at a time with no newline until the client closes
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for _ in range(600):  # 60 s at most, only so a broken client cannot hang the test
                    self.wfile.write(b"d")
                    self.wfile.flush()
                    time.sleep(0.1)
                self.server.gave_up += 1  # the client never closed: its deadline did not work
            except OSError:
                pass
            return
        if mode == "fail" or self.path != "/v1/chat/completions":
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        prompt = body["messages"][-1]["content"]
        count = min(int(body.get("max_tokens") or 16), 12)
        seed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.02)  # fake prefill
        for i in range(count):
            event = {"choices": [{"delta": {"content": "t%s_%d " % (seed[:4], i)}, "finish_reason": None}]}
            self.wfile.write(("data: %s\n\n" % json.dumps(event)).encode())
            self.wfile.flush()
            time.sleep(0.004)
        final = {"choices": [{"delta": {}, "finish_reason": "length"}]}
        self.wfile.write(("data: %s\n\n" % json.dumps(final)).encode())
        if mode != "nousage":
            usage = {"choices": [], "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": count}}
            self.wfile.write(("data: %s\n\n" % json.dumps(usage)).encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.server.ok_served += 1


def _start_fake_server(mode):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    server.mode = mode
    server.hits = 0
    server.failing = False
    server.failed_served = 0
    server.ok_served = 0
    server.gave_up = 0
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _closed_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _namespace(**overrides):
    values = dict(base_url="http://127.0.0.1:0", model="fake", duration=2.0, prompts_file=None, max_tokens=12,
                  request_timeout=120.0, header=None, fault=None, out_dir=None, sample_interval=1.0,
                  pid=[os.getpid()], pid_file=None, concurrency=1, max_error_rate=0.0, max_ttft_p95=None,
                  max_restarts=None, max_private_growth_mib_per_hour=None, allow_nondeterminism=False,
                  self_test=False, fault_window=30.0, recovery_budget_s=120.0, warmup_s=600.0,
                  expected_restarts=None, min_available_mib=None, min_commit_headroom_mib=None,
                  header_env=None, require_usage_tokens=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def _read_jsonl(path):
    """Rows of a JSON-lines file that is still being written; a torn last line is skipped."""
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def _run_until(args, until, on_window=None, seams=None):
    """Runs run(args) while a watcher thread reads the files the run is writing, and ends the run on what it sees.

    until(request_rows, window) gets the rows of requests.jsonl written so far. In a run with no fault,
    window is None. In a run with a fault nothing is evaluated until the fault's event row exists; window is
    then (start_rel, end_rel) copied from that row. run() records the fault window before it writes the row,
    so the pair is exactly the window the criteria will use. on_window(window), if given, is called once at
    that point, before until. The run is stopped when until returns true, or, unwitnessed, when the fault
    command failed (no window will ever exist). args.duration is only a ceiling.

    No decision here waits a fixed time for something to have happened: each one is taken on a row that is
    already in a file. The 20 ms sleep is the polling period and nothing else.

    Returns (summary, code, window or None, witnessed)."""
    stop = threading.Event()
    seen = {"window": None, "witnessed": False}
    events_path = os.path.join(args.out_dir, "events.jsonl")
    requests_path = os.path.join(args.out_dir, "requests.jsonl")

    def watch():
        released = False
        while not stop.is_set():
            time.sleep(0.02)
            if args.fault:
                events = [e for e in _read_jsonl(events_path) if e.get("event") in ("fault", "fault_error")]
                if not events:
                    continue
                if events[0]["event"] != "fault" or events[0].get("returncode") != 0:
                    stop.set()  # the fault command failed: there is no window to wait for
                    return
                seen["window"] = (events[0]["start_rel"], events[0]["end_rel"])
                if on_window is not None and not released:
                    on_window(seen["window"])
                    released = True
            if until(_read_jsonl(requests_path), seen["window"]):
                seen["witnessed"] = True
                stop.set()

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        summary, code = run(args, stop_event=stop, seams=seams)
    finally:
        stop.set()
        thread.join(timeout=5)
    return summary, code, seen["window"], seen["witnessed"]


def _ok_after(rows, window):
    """A successful request that started at or after the recorded end of the fault window."""
    return any(r["ok"] and r["t_rel"] >= window[1] for r in rows)


def _failed_after(rows, window):
    """A failed request that started strictly after the recorded end of the fault window."""
    return any(not r["ok"] and r["t_rel"] > window[1] for r in rows)


def _passed(summary, code):
    """True only for a run that succeeded outright: exit code 0, success true, no failed criterion."""
    return code == 0 and summary["success"] is True and summary["failed_criteria"] == []


def _passed_or_sampler_starved(summary, code):
    """True when the run succeeded outright, or when host_samples is its ONLY failed criterion and the
    summary's own numbers show a real shortfall (fewer sample rows than the stated minimum, exit code 2).

    Only the healthy run (section 1) uses this: it samples at 1 Hz on the real host, and a sampler that is
    starved of CPU can truly fall under 80 % of its ticks. The criterion itself is checked without a clock
    in section 1b. Every other run in the self-test must satisfy _passed."""
    if _passed(summary, code):
        return True
    host = summary["host"]
    return (code == 2 and summary["success"] is False and summary["failed_criteria"] == ["host_samples"]
            and host["samples"] < host["samples_expected_min"])


def _window_in_event_row(out_dir, summary):
    """True when events.jsonl has one fault row per fired fault and each carries the window in the summary."""
    rows = [e for e in _read_jsonl(os.path.join(out_dir, "events.jsonl"))
            if e.get("event") == "fault" and e.get("returncode") == 0]
    return (len(rows) == summary["faults"]["fired"] > 0
            and [[e["start_rel"], e["end_rel"]] for e in rows] == summary["faults"]["windows_s"]
            and all(e["start_rel"] <= e["end_rel"] <= e["t_rel"] for e in rows))


class _StarvedSampler(Sampler):
    """A sampler whose thread never gets to take a periodic sample. Only stop()'s closing sample is recorded,
    so a run has exactly one sample row however long it lasts."""

    def run(self):
        self.stop_event.wait()


class _Stalls:
    """Seams for run() that put a thread to sleep once at each named point, standing in for a harness thread
    the host did not schedule. The worker points only stall once the fault command has returned, so the
    stalls land around the fault window rather than on the first request of the run."""

    FAULT_POINTS = ("fault_command_returned", "fault_window_recorded", "fault_event_logged")
    WORKER_POINTS = ("request_stamped", "request_returned")

    def __init__(self, seconds):
        self.seconds = seconds
        self.slept = set()
        self._armed = threading.Event()
        self._lock = threading.Lock()

    def seams(self):
        return {name: (lambda name=name: self._at(name)) for name in self.FAULT_POINTS + self.WORKER_POINTS}

    def _at(self, name):
        if name in self.FAULT_POINTS:
            self._armed.set()
        elif not self._armed.is_set():
            return
        with self._lock:
            if name in self.slept:
                return
            self.slept.add(name)
        time.sleep(self.seconds)

    def missing(self):
        return sorted(set(self.FAULT_POINTS + self.WORKER_POINTS) - self.slept)


def _fault_window_controls(root, tag, lag_s, stall_s=0.0):
    """Positive and negative controls for fault windows, error exemption and recovery.

    Returns a list of problem strings. lag_s is slept by the fault subprocess before its first request and
    again before it exits. stall_s, when positive, is slept once by the fault thread after the command
    returns, after the window is recorded and after the event row is written, and once by a worker after it
    stamps a request and after the request returns. The controls must give the same answers whatever the
    two are: every run ends on rows observed in its files, and every expectation follows from the order of
    events, not from how long anything took."""
    problems = []

    def expect(condition, message):
        if not condition:
            problems.append(message)

    def start():
        server = _start_fake_server("ok")
        return server, server.server_address[1]

    def command(port, steps):
        code = ("import time, urllib.request as u; time.sleep(%r); p='http://127.0.0.1:%d/__flaky/'; "
                "[u.urlopen(u.Request(p+s, data=b'x'), timeout=55) for s in %r]; time.sleep(%r)"
                % (lag_s, port, tuple(steps), lag_s))
        return '"%s" -I -c "%s"' % (sys.executable.replace("\\", "/"), code)

    ceiling = 60.0 + 2 * lag_s + 6 * stall_s  # only bounds a broken run; reaching it fails the control

    def go(name, steps, until, on_window=None, **overrides):
        out = os.path.join(root, name + "_" + tag)
        stalls = _Stalls(stall_s) if stall_s > 0 else None
        # sample_interval == duration: the run owes exactly one host sample, and the closing sample is
        # always taken, so host_samples cannot fail here whatever the host load is.
        args = _namespace(base_url="http://127.0.0.1:%d" % port, duration=ceiling, sample_interval=ceiling,
                          out_dir=out, fault=["0.3:" + command(port, steps)], **overrides)
        summary, code, window, witnessed = _run_until(args, until, on_window=on_window,
                                                      seams=stalls.seams() if stalls else None)
        expect(window is not None and summary["faults"]["fired"] == 1,
               "%s: fault did not fire: %s" % (name, summary["faults"]))
        expect(witnessed, "%s: control inconclusive, the run ended before its witness row was seen" % name)
        expect(window is None or _window_in_event_row(out, summary),
               "%s: the fault event row does not carry the recorded window: %s" % (name, summary["faults"]))
        if stalls is not None:
            expect(not stalls.missing(), "%s: control inconclusive, no stall was injected at %s"
                   % (name, stalls.missing()))
        return summary, code, window or (None, None), _read_jsonl(os.path.join(out, "requests.jsonl"))

    # Positive: failures begin inside the fault window (the command waits until one was served) and stop
    # before the command exits. With no grace at all they are exempt, recovery is seen and the run passes.
    server, port = start()
    try:
        summary, code, window, rows = go("flaky", ("begin", "wait", "end"), _ok_after, fault_window=0.0)
        failed = [r for r in rows if not r["ok"]]
        expect(bool(failed) and summary["requests"]["failed"] == len(failed)
               and summary["requests"]["failed_outside_fault_windows"] == 0
               and len(summary["requests"]["expected_fault_failures"]) == len(failed),
               "flaky: failures inside the fault window were not exempted: %s" % summary["requests"])
        expect(_passed(summary, code), "flaky: run did not pass: %s" % summary["failed_criteria"])
    finally:
        server.shutdown()
        server.server_close()

    # Negative: the command only arms. The harness switches failures on once the fault row, and with it the
    # recorded window end, is in events.jsonl; the run then goes on until a failure that started after that
    # end is in requests.jsonl. With no grace such a failure must fail error_rate. The late list is worked
    # out from requests.jsonl here, not taken from run().
    server, port = start()
    try:
        def release(window):
            server.failing = True

        summary, code, window, rows = go("flaky0", ("arm",), _failed_after, on_window=release, fault_window=0.0)
        late = [r for r in rows if not r["ok"] and window[1] is not None and r["t_rel"] > window[1]]
        expect(bool(late), "flaky0: control inconclusive, no failure started after the fault window ended")
        expect(code == 2 and "error_rate" in summary["failed_criteria"]
               and summary["requests"]["failed_outside_fault_windows"] == len(late),
               "flaky0: failures outside the fault window did not fail error_rate: %s %s %s"
               % (summary["failed_criteria"], summary["requests"], summary["faults"]))
    finally:
        server.shutdown()
        server.server_close()

    # Never recovers: one request is answered in full first (ok_wait), then failures begin in the window and
    # never end. The run goes on until a failure that started after the window end is in requests.jsonl.
    # Every failure is exempt (unbounded grace), so fault_recovery must be the only failed criterion.
    server, port = start()
    try:
        summary, code, window, rows = go("norecover", ("ok_wait", "begin", "wait"), _failed_after,
                                         fault_window=1e9)
        expect(window[1] is not None and not any(r["ok"] and r["t_rel"] >= window[1] for r in rows),
               "norecover: control inconclusive, a request succeeded after the fault window ended")
        expect(code == 2 and summary["failed_criteria"] == ["fault_recovery"],
               "norecover: expected exactly fault_recovery to fail: %s %s"
               % (summary["failed_criteria"], summary["requests"]))
    finally:
        server.shutdown()
        server.server_close()
    return problems


def self_test():
    """Runs positive and negative controls against a built-in fake server. Returns 0 on success.

    Runs that need something to have happened end on rows observed in their own output files
    (_run_until); their duration is a 60 s ceiling that only bounds a broken run."""
    problems = []

    def expect(condition, message):
        if not condition:
            problems.append(message)

    def some_ok(count):
        return lambda rows, window: sum(1 for r in rows if r["ok"]) >= count

    def any_row(rows, window):
        return len(rows) >= 1

    root = tempfile.mkdtemp(prefix="sonder-reliability-selftest-")
    try:
        # 1. Healthy server: a real run at 1 Hz sampling must succeed and its files must agree. It ends once
        #    three requests have completed and three sample rows (the first and two periodic ones) exist.
        server = _start_fake_server("ok")
        try:
            base = "http://127.0.0.1:%d" % server.server_address[1]
            out = os.path.join(root, "ok")
            summary, code, _, witnessed = _run_until(
                _namespace(base_url=base, duration=60.0, out_dir=out),
                lambda rows, window: (some_ok(3)(rows, window)
                                      and len(_read_jsonl(os.path.join(out, "samples.jsonl"))) >= 3))
            expect(witnessed, "healthy run reached its ceiling before 3 requests and 3 samples were seen")
            expect(_passed_or_sampler_starved(summary, code),
                   "healthy run did not succeed: code %s, %s" % (code, summary["failed_criteria"]))
            if not _passed(summary, code):
                print("self-test note: the sampler of the healthy run fell behind on this host (%d samples, "
                      "minimum %d); host_samples failed for that reason alone" % (
                          summary["host"]["samples"], summary["host"]["samples_expected_min"]), file=sys.stderr)
            expect(summary["requests"]["completed"] >= 3, "too few completed requests: %s" % summary["requests"])
            expect(summary["requests"]["failed"] == 0, "healthy run had failures: %s" % summary["requests"])
            expect(server.hits >= summary["requests"]["attempted"] > 0, "fake server saw fewer hits than requests")
            expect(summary["tokens"]["completion_total"] > 0 and summary["tokens"]["sources"] == ["usage"],
                   "token counts not taken from usage: %s" % summary["tokens"])
            expect(summary["ttft_s"]["p50"] is not None and summary["ttft_s"]["p50"] > 0, "no TTFT recorded")
            expect(summary["decode_tps"]["p50"] is not None and summary["decode_tps"]["p50"] > 0,
                   "no decode tok/s recorded")
            expect(summary["host"]["samples"] >= 4, "sampler recorded %d samples" % summary["host"]["samples"])
            expect(summary["host"]["min_available_bytes"] is not None, "host available memory not sampled")
            with open(os.path.join(out, "requests.jsonl"), encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            expect(len(rows) == summary["requests"]["attempted"], "requests.jsonl row count differs from summary")
            with open(os.path.join(out, "samples.jsonl"), encoding="utf-8") as handle:
                samples = [json.loads(line) for line in handle if line.strip()]
            expect(len(samples) == summary["host"]["samples"], "samples.jsonl row count differs from summary")
            own = [p for s in samples for p in s["processes"] if p["pid"] == os.getpid()]
            expect(bool(own) and own[0]["alive"] and own[0]["working_set_bytes"],
                   "own process counters not sampled: %s" % own[:1])

            # 1b. The host_samples criterion without a clock. First its arithmetic on fixed numbers:
            #     (sample rows, duration, elapsed, interval) gives (passes, minimum).
            for case, want in (
                    ((8, 10.0, 10.0, 1.0), (True, 8)),       # 80 % of 10 ticks
                    ((7, 10.0, 10.0, 1.0), (False, 8)),      # one short
                    ((3, 3600.0, 4.0, 1.0), (True, 3)),      # ended early: owes for the 4 s it ran
                    ((2, 3600.0, 4.0, 1.0), (False, 3)),
                    ((8, 10.0, 25.0, 1.0), (True, 8)),       # overran: duration is the cap
                    ((7, 10.0, 25.0, 1.0), (False, 8)),
                    ((4, 10.0, 10.0, 2.0), (True, 4)),       # interval scales the count
                    ((3, 10.0, 10.0, 2.0), (False, 4)),
                    ((1, 60.0, 0.4, 1.0), (True, 1)),        # never under one
                    ((0, 60.0, 0.4, 1.0), (False, 1))):      # no samples is never a pass
                got = host_samples_ok(*case)
                expect(got == want, "host_samples_ok%r returned %r, expected %r" % (case, got, want))
            #     Then through run() with a sampler that only ever records the closing sample (one row).
            #     Owing one sample (interval == duration) the run passes; owing more it fails on host_samples
            #     and nothing else. Five completed requests take at least 0.34 s of server-side sleeps and
            #     two samples are owed from 0.125 s on at a 0.05 s interval; a slower host only owes more.
            starved = {"sampler_class": _StarvedSampler}
            summary, code, _, witnessed = _run_until(
                _namespace(base_url=base, duration=60.0, sample_interval=60.0,
                           out_dir=os.path.join(root, "starved_owes_one")), some_ok(3), seams=starved)
            expect(witnessed and _passed(summary, code) and summary["host"]["samples"] == 1
                   and summary["host"]["samples_expected_min"] == 1,
                   "one sample, one owed, did not pass: %s %s" % (summary["host"], summary["failed_criteria"]))
            summary, code, _, witnessed = _run_until(
                _namespace(base_url=base, duration=60.0, sample_interval=0.05,
                           out_dir=os.path.join(root, "starved_owes_more")), some_ok(5), seams=starved)
            expect(witnessed and code == 2 and summary["success"] is False
                   and summary["failed_criteria"] == ["host_samples"] and summary["host"]["samples"] == 1
                   and summary["host"]["samples_expected_min"] >= 2,
                   "one sample with several owed did not fail host_samples alone: %s %s"
                   % (summary["host"], summary["failed_criteria"]))
        finally:
            server.shutdown()
            server.server_close()

        # 2. Positive fault gate: a scheduled fault that runs counts as fired. From here on runs that are
        #    not about the sampler use sample_interval == duration, so exactly one sample is owed and the
        #    closing sample always covers it: host_samples cannot fail and every such run must pass outright.
        server = _start_fake_server("ok")
        try:
            base = "http://127.0.0.1:%d" % server.server_address[1]
            command = '"%s" -c "pass"' % sys.executable.replace("\\", "/")
            # Ends on an observed row: an ok request that started at or after the recorded window end.
            out = os.path.join(root, "fault_ok")
            summary, code, _, witnessed = _run_until(
                _namespace(base_url=base, duration=60.0, sample_interval=60.0, out_dir=out,
                           fault=["0.3:" + command]), _ok_after)
            expect(witnessed and _passed(summary, code) and summary["faults"]["fired"] == 1
                   and _window_in_event_row(out, summary),
                   "scheduled fault did not fire/succeed: %s %s" % (summary["faults"], summary["failed_criteria"]))
            # 3. Negative control: a fault scheduled past the end never fires, so the run must fail.
            summary, code = run(_namespace(base_url=base, duration=1.0, out_dir=os.path.join(root, "fault_late"),
                                           fault=["500:" + command]))
            expect(code != 0 and not summary["success"] and "faults_fired" in summary["failed_criteria"],
                   "a fault that never fired did not fail the run: %s" % summary["failed_criteria"])
        finally:
            server.shutdown()
            server.server_close()

        # 3b. Fault windows three times: with prompt fault commands, with the fault subprocess made slow to
        #     start and slow to exit, and with the fault and worker threads of the harness itself stalled for
        #     0.8 s around the moment the window is recorded. Each control has its own server and ends on
        #     observed rows, so the answers must be the same all three times.
        for lag, stall, tag in ((0.0, 0.0, "fast"), (2.0, 0.0, "lag"), (0.0, 0.8, "stall")):
            for message in _fault_window_controls(root, tag, lag, stall):
                problems.append("[fault controls, %s] %s" % (tag, message))
        # Pure helpers: growth is measured after warm-up, windows and recovery behave on synthetic rows.
        ramp = [(float(t), 100.0 + (50.0 if t < 10 else 0.0)) for t in range(0, 20)]
        expect(abs(growth_after_warmup(ramp, 10.0)) < 1e-6, "warm-up samples leaked into the growth fit")
        expect(growth_after_warmup(ramp, 0.0) < 0, "control: the fit over the whole run should show the step")
        rows = [{"ok": False, "t_rel": 5.0, "end_rel": 6.0}, {"ok": True, "t_rel": 20.0, "end_rel": 21.0}]
        expect(in_fault_window(rows[0], [(5.5, 5.6)], 0.0) and not in_fault_window(rows[1], [(5.5, 5.6)], 2.0),
               "in_fault_window wrong")
        late_row = {"ok": False, "t_rel": 7.0, "end_rel": 7.5}
        expect(in_fault_window(late_row, [(5.5, 5.6)], 1.5) and not in_fault_window(late_row, [(5.5, 5.6)], 1.0)
               and not in_fault_window({"ok": False, "t_rel": 1.0, "end_rel": 5.0}, [(5.5, 5.6)], 30.0),
               "in_fault_window grace or start edge wrong")
        expect(unrecovered_faults([(5.5, 5.6)], rows, 30.0) == [] and unrecovered_faults([(5.5, 5.6)], rows, 5.0) == [5.5],
               "unrecovered_faults wrong")

        # 4. Negative control: every request fails, zero completed, never a success. Ends on the first
        #    request row.
        server = _start_fake_server("fail")
        try:
            base = "http://127.0.0.1:%d" % server.server_address[1]
            summary, code, _, witnessed = _run_until(
                _namespace(base_url=base, duration=60.0, sample_interval=60.0,
                           out_dir=os.path.join(root, "http500")), any_row)
            expect(witnessed and code != 0 and not summary["success"], "HTTP 500 run reported success")
            expect(summary["requests"]["completed"] == 0, "HTTP 500 run completed requests")
            expect("requests_completed" in summary["failed_criteria"], "zero-completed criterion did not fail")
            expect(server.hits > 0, "failing server was never contacted: the control injected nothing")
        finally:
            server.shutdown()
            server.server_close()

        # 4b. Total deadline: a server trickling bytes must fail the request at --request-timeout (a
        #     per-socket timeout alone never fires because every byte arrives in time). The server keeps
        #     trickling until the client closes, so only the deadline of the client can end the request;
        #     gave_up counts a server that stopped first.
        server = _start_fake_server("trickle")
        try:
            base = "http://127.0.0.1:%d" % server.server_address[1]
            outcome = run_request(base, "fake", "hello", 8, 0.8, {})
            expect(not outcome["ok"] and (outcome["error"] or "").startswith("deadline_exceeded"),
                   "trickling server did not trip the total deadline: %s" % outcome)
            expect(server.gave_up == 0, "the trickling server stopped before the client deadline did")
            expect(server.hits == 1, "trickle control never contacted the server")
        finally:
            server.shutdown()
            server.server_close()
        # 4b2. Deadline before the body: a server that never replies must trip a 0.8 s deadline. The server
        #      holds the connection until the client closes it.
        server = _start_fake_server("slow_headers")
        try:
            base = "http://127.0.0.1:%d" % server.server_address[1]
            outcome = run_request(base, "fake", "hello", 8, 0.8, {})
            expect(not outcome["ok"] and (outcome["error"] or "").startswith("deadline_exceeded"),
                   "slow-header server did not trip the total deadline: %s" % outcome)
            expect(server.gave_up == 0, "the slow-header server stopped before the client deadline did")
            expect(server.hits == 1, "slow-header control never contacted the server")
        finally:
            server.shutdown()
            server.server_close()
        # 4b3. Base URL ending in /v1 reaches /v1/chat/completions (not /v1/v1/...).
        server = _start_fake_server("ok")
        try:
            base = "http://127.0.0.1:%d" % server.server_address[1]
            for suffix in ("/v1", "/v1/"):
                outcome = run_request(base + suffix, "fake", "hello", 8, 120.0, {})
                expect(outcome["ok"], "base URL ending in %s failed: %s" % (suffix, outcome))
        finally:
            server.shutdown()
            server.server_close()
        # 4b4. Exact token counts: with no usage event the run still succeeds by default (source "chunks"),
        #      and --require-usage-tokens fails it on exact_token_counts alone. Both end on the first ok row.
        server = _start_fake_server("nousage")
        try:
            base = "http://127.0.0.1:%d" % server.server_address[1]
            summary, code, _, witnessed = _run_until(
                _namespace(base_url=base, duration=60.0, sample_interval=60.0,
                           out_dir=os.path.join(root, "nousage")), some_ok(1))
            expect(witnessed and _passed(summary, code) and summary["tokens"]["sources"] == ["chunks"],
                   "no-usage run without the flag: %s %s" % (summary["tokens"], summary["failed_criteria"]))
            summary, code, _, witnessed = _run_until(
                _namespace(base_url=base, duration=60.0, sample_interval=60.0,
                           out_dir=os.path.join(root, "nousage_req"), require_usage_tokens=True), some_ok(1))
            expect(witnessed and code == 2 and summary["failed_criteria"] == ["exact_token_counts"],
                   "--require-usage-tokens did not fail a no-usage run: %s" % summary["failed_criteria"])
        finally:
            server.shutdown()
            server.server_close()
        # 4b5. Header parsing: malformed items are rejected without echoing the secret.
        secret = "S3CRET-TOKEN"
        for items, env_items, environ in (
                (["Authorization" + secret], None, {}),        # missing '='
                (["=" + secret], None, {}),                    # empty name
                (None, ["Authorization=MISSING_VAR"], {})):    # unset env var
            try:
                parse_headers(items, env_items, environ)
                expect(False, "parse_headers accepted %r %r" % (items, env_items))
            except ValueError as exc:
                expect(secret not in str(exc), "parse_headers error leaked the secret: %s" % exc)
        try:
            ok_headers = parse_headers(["X-A=1"], ["Authorization=SELFTEST_TOKEN_VAR"],
                                       {"SELFTEST_TOKEN_VAR": secret})
            expect(ok_headers == {"X-A": "1", "Authorization": secret},
                   "parse_headers result wrong: %s" % list(ok_headers))
        except ValueError as exc:
            expect(False, "parse_headers rejected valid input: %s" % exc)
        # 4c. Fault evidence: a fault command that exits 0 but never changes --pid-file, with a restart
        #     expected, must fail on restarts_explained alone; the same run expecting 0 restarts must pass.
        server = _start_fake_server("ok")
        try:
            base = "http://127.0.0.1:%d" % server.server_address[1]
            outcome = run_request(base, "fake", "x", 8, 2.0, {"X-A": "bad\nSECRET"})
            expect(not outcome["ok"] and "SECRET" not in (outcome["error"] or "")
                   and (outcome["error"] or "").startswith("ValueError"), "header value leaked: %s" % outcome)
            pid_path = os.path.join(root, "server.pid")
            with open(pid_path, "w", encoding="ascii") as handle:
                handle.write(str(os.getpid()))
            noop = '"%s" -c "pass"' % sys.executable.replace("\\", "/")
            summary, code, _, witnessed = _run_until(
                _namespace(base_url=base, duration=60.0, sample_interval=60.0,
                           out_dir=os.path.join(root, "noeffect"),
                           fault=["0.3:" + noop], pid_file=pid_path, expected_restarts=1), _ok_after)
            expect(witnessed and code == 2 and summary["failed_criteria"] == ["restarts_explained"],
                   "a fault that restarted nothing passed with --expected-restarts 1: %s" % summary["failed_criteria"])
            summary, code, _, witnessed = _run_until(
                _namespace(base_url=base, duration=60.0, sample_interval=60.0,
                           out_dir=os.path.join(root, "noeffect0"),
                           fault=["0.3:" + noop], pid_file=pid_path, expected_restarts=0), _ok_after)
            expect(witnessed and _passed(summary, code),
                   "control: unchanged pid with 0 expected restarts failed: %s" % summary["failed_criteria"])
            # Lag proof for sections 2 and 4c: a fault command that takes 2.5 s to start and exit (a loaded
            # host) must give the same answers as the prompt one above.
            slow = '"%s" -I -c "import time; time.sleep(2.5)"' % sys.executable.replace("\\", "/")
            for tag, extra in (("slow_fault_ok", {}), ("slow_noeffect0", {"pid_file": pid_path, "expected_restarts": 0})):
                summary, code, window, witnessed = _run_until(
                    _namespace(base_url=base, duration=60.0, sample_interval=60.0,
                               out_dir=os.path.join(root, tag), fault=["0.3:" + slow], **extra), _ok_after)
                expect(window is not None and witnessed and _passed(summary, code)
                       and summary["faults"]["fired"] == 1,
                       "slow fault (%s) did not pass: %s %s" % (tag, summary["faults"], summary["failed_criteria"]))
            # 4d. Wall-clock step: time.time() jumps back 100 s while a fault command is running (WSL does
            #     this). Every relative value comes from the monotonic clock, so the window and the request
            #     rows must stay ordered and non-negative.
            real_time = time.time
            stepped_at = real_time() + 0.8

            def stepping_time():
                now = real_time()
                return now - 100.0 if now > stepped_at else now

            time.time = stepping_time
            try:
                summary, code, window, witnessed = _run_until(
                    _namespace(base_url=base, duration=60.0, sample_interval=60.0,
                               out_dir=os.path.join(root, "clockstep"), fault=["0.3:" + slow]), _ok_after)
            finally:
                time.time = real_time
            step_out = os.path.join(root, "clockstep")
            step_rows = _read_jsonl(os.path.join(step_out, "requests.jsonl"))
            expect(window is not None and witnessed and summary["faults"]["fired"] == 1
                   and window[0] <= window[1] and window[0] >= 0 and _window_in_event_row(step_out, summary),
                   "wall-clock step disordered the fault window: %s" % summary["faults"])
            expect(bool(step_rows) and all(0 <= r["t_rel"] <= r["end_rel"] for r in step_rows),
                   "wall-clock step produced a negative or reversed request time")
        finally:
            server.shutdown()
            server.server_close()

        # 5. Negative control: nothing listening, zero completed, never a success. Ends on the first
        #    request row.
        base = "http://127.0.0.1:%d" % _closed_port()
        summary, code, _, witnessed = _run_until(
            _namespace(base_url=base, duration=60.0, sample_interval=60.0,
                       out_dir=os.path.join(root, "refused")), any_row)
        expect(witnessed and code != 0 and not summary["success"], "connection-refused run reported success")
        expect(summary["requests"]["completed"] == 0 and summary["requests"]["attempted"] > 0,
               "connection-refused run did not attempt requests: %s" % summary["requests"])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if problems:
        for problem in problems:
            print("SELF-TEST FAIL: " + problem, file=sys.stderr)
        return 1
    print("self-test ok")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="run against a built-in fake server and exit")
    parser.add_argument("--base-url", help="endpoint root, e.g. http://127.0.0.1:8080 (a trailing /v1 is accepted)")
    parser.add_argument("--model", default="default")
    parser.add_argument("--duration", type=float, default=3600.0, help="seconds to keep issuing requests")
    parser.add_argument("--prompts-file", help="one fixed prompt per line (default: built-in set)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--header", action="append", metavar="K=V")
    parser.add_argument("--header-env", action="append", metavar="K=ENVVAR",
                        help="header whose value is read from environment variable ENVVAR (keeps tokens off the "
                             "command line); repeatable")
    parser.add_argument("--require-usage-tokens", action="store_true",
                        help="fail unless every completed request got its token counts from the server's usage "
                             "event (otherwise the fallback is one token per streamed chunk)")
    parser.add_argument("--out-dir", default="reliability-run")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--pid", type=int, action="append", help="process id to sample (repeatable)")
    parser.add_argument("--pid-file", help="file holding the current server pid; a changed pid counts as a restart (see --expected-restarts)")
    parser.add_argument("--fault", action="append", metavar="SECONDS:COMMAND",
                        help="run COMMAND (shlex split, no shell; use forward slashes on Windows) SECONDS into the run; repeatable. Must exit 0 to count as fired; "
                             "exit 0 is the only evidence the fault did anything unless --pid-file/--expected-restarts is used")
    parser.add_argument("--max-error-rate", type=float, default=0.0,
                        help="limit on the failed fraction of requests OUTSIDE fault windows")
    parser.add_argument("--fault-window", type=float, default=30.0, metavar="SECONDS",
                        help="requests that overlap [fault start, fault end + SECONDS] are expected errors: "
                             "listed in the summary, not counted against --max-error-rate")
    parser.add_argument("--recovery-budget-s", type=float, default=120.0,
                        help="after each fired fault a request must succeed (start and finish) within this many seconds")
    parser.add_argument("--warmup-s", type=float, default=600.0,
                        help="private-bytes growth is fitted only to samples taken after this many seconds")
    parser.add_argument("--expected-restarts", type=int,
                        help="exact number of pid changes in --pid-file the fault schedule must cause (default 0); "
                             "more or fewer fails the run, so a fault that hit nothing is caught")
    parser.add_argument("--min-available-mib", type=float,
                        help="fail if host available physical memory ever drops below this")
    parser.add_argument("--min-commit-headroom-mib", type=float,
                        help="fail if host commit limit minus commit used ever drops below this")
    parser.add_argument("--max-ttft-p95", type=float)
    parser.add_argument("--max-restarts", type=int)
    parser.add_argument("--max-private-growth-mib-per-hour", type=float)
    parser.add_argument("--allow-nondeterminism", action="store_true",
                        help="do not fail when temperature-0 output differs between repeats")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.base_url:
        print("--base-url is required", file=sys.stderr)
        return 64
    if args.duration <= 0 or args.sample_interval <= 0:
        print("--duration and --sample-interval must be positive", file=sys.stderr)
        return 64
    try:
        parse_headers(args.header, args.header_env, os.environ)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 64
    summary, code = run(args)
    print(json.dumps({k: summary[k] for k in ("success", "failed_criteria", "requests", "ttft_s", "decode_tps",
                                                 "faults", "out_dir")}, indent=2))
    if code != 0:
        print("RELIABILITY RUN FAILED: " + ", ".join(summary["failed_criteria"]), file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
