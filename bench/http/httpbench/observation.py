"""Small, dependency-free probes for server-side benchmark observations.

The probes are deliberately independent of the request client.  A metrics URL
may point at a different origin and therefore never inherits request headers.
Snapshots are JSON serialisable and retain enough state for callers to report
unknown data and file truncation honestly.
"""

from __future__ import annotations

import math
import os
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


_METRIC = re.compile(
    r"^\s*(?:llamacpp:)?spec_decode_accepted_tokens_per_pos_total\s*"
    r"\{(?P<labels>[^}]*)\}\s+(?P<value>[^\s#]+)(?:\s+[^\s]+)?\s*$"
)
# The value class excludes the backslash so an escape can only match one way
# (no catastrophic backtracking on backslash-heavy label values).
_LABEL = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\"(?P<value>(?:\\.|[^\"\\])*)\"")
_POSITION = re.compile(r"^-?[0-9]+$")
_LOG_PATTERNS = {
    "making_room": re.compile(r"making room", re.I),
    "exceeds_cache_size_limit": re.compile(r"exceeds cache size limit", re.I),
    "forcing_full_prompt_re_processing": re.compile(r"forcing full prompt re-processing", re.I),
    "selected_slot_by_id": re.compile(r"selected slot by id", re.I),
}
_PROGRESS = re.compile(r"progress\s*=\s*1\.00(?:\b|$)", re.I)


def _finite_number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _prometheus_positions(body: str) -> tuple[dict[str, float], int]:
    counters: dict[str, float] = {}
    malformed = 0
    for line in body.splitlines():
        match = _METRIC.match(line)
        if not match:
            if line.lstrip().startswith(("llamacpp:spec_decode_accepted_tokens_per_pos_total",
                                        "spec_decode_accepted_tokens_per_pos_total")):
                malformed += 1
            continue
        label_text = match.group("labels")
        matches = list(_LABEL.finditer(label_text))
        labels = {m.group("key"): m.group("value") for m in matches}
        remainder = _LABEL.sub("", label_text).replace(",", "").strip()
        if remainder or len(labels) != len(matches):
            malformed += 1
            continue
        position = labels.get("position")
        value = _finite_number(match.group("value"))
        if position is None or not _POSITION.fullmatch(position) or int(position) < 0 or value is None:
            malformed += 1
            continue
        if position in counters:
            malformed += 1
            continue
        counters[position] = value
    return counters, malformed


class MetricsProbe:
    """Fetch and parse one Prometheus exposition endpoint."""

    def __init__(self, url: str, timeout: float = 5.0):
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("metrics URL must be an HTTP(S) endpoint without embedded credentials")
        _ = parsed.port  # validate the port before starting the run
        self.url = url
        requested = float(timeout)
        self.timeout = max(requested, 0.05) if math.isfinite(requested) else 5.0

    def snapshot(self) -> dict[str, Any]:
        started = time.monotonic()
        try:
            # Deliberately no caller headers: metrics may be cross-origin.
            with urlopen(Request(self.url, method="GET"), timeout=self.timeout) as response:
                body = response.read()  # metrics is a complete exposition, not a preview
                status = int(response.status)
            text = body.decode("utf-8", errors="replace")
            positions, malformed = _prometheus_positions(text)
            result: dict[str, Any] = {
                "status": status,
                "error": None,
                "accepted_per_position": positions,
                "elapsed_s": time.monotonic() - started,
            }
            if malformed:
                result["parse_errors"] = malformed
            return result
        except HTTPError as exc:
            return {"status": int(exc.code), "error": str(exc), "accepted_per_position": {}}
        except (OSError, URLError, TimeoutError, ValueError) as exc:
            return {"status": None, "error": str(exc), "accepted_per_position": {}}


def metrics_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    """Return counter increases, preserving unknown/reset state explicitly."""
    if not before or not after:
        return {"accepted_per_position": {}, "errors": ["missing snapshot"], "resets": []}
    if before.get("error") or after.get("error"):
        errors = [x for x in (before.get("error"), after.get("error")) if x]
        return {"accepted_per_position": {}, "errors": errors, "resets": []}
    old = before.get("accepted_per_position")
    new = after.get("accepted_per_position")
    if not isinstance(old, dict) or not isinstance(new, dict):
        return {"accepted_per_position": {}, "errors": ["invalid counter map"], "resets": []}
    if before.get("parse_errors") or after.get("parse_errors"):
        return {"accepted_per_position": {}, "errors": ["malformed or duplicate accepted-position samples"],
                "resets": [], "unknown": []}
    if not old or not new:
        return {"accepted_per_position": {}, "errors": ["accepted-position counters absent"],
                "resets": [], "unknown": sorted(set(old) | set(new))}
    out: dict[str, float] = {}
    resets: list[str] = []
    unknown: list[str] = []
    for position in sorted(set(old) | set(new), key=lambda x: (not str(x).isdigit(), str(x))):
        if position not in old or position not in new:
            unknown.append(str(position))
            continue
        try:
            previous = float(old.get(position, 0))
            current = float(new.get(position, 0))
        except (TypeError, ValueError):
            unknown.append(str(position))
            continue
        if not (math.isfinite(previous) and math.isfinite(current)) or min(previous, current) < 0:
            unknown.append(str(position))
            continue
        if current < previous:
            resets.append(str(position))
        elif position in new:
            out[str(position)] = current - previous
    return {"accepted_per_position": out, "errors": [], "resets": resets, "unknown": unknown}


def _count_log_text(text: str) -> dict[str, int]:
    counts = {name: len(pattern.findall(text)) for name, pattern in _LOG_PATTERNS.items()}
    # A single completed progress line is normal.  Report only additional
    # entries in consecutive matching runs as repeated progress.
    repeated = 0
    progress_lines = 0
    previous_identity: tuple | None = None
    for line in text.splitlines():
        if _PROGRESS.search(line):
            progress_lines += 1
            slot = re.search(r"\bslot(?:\s+\w+:)?\s*(?:id\s*)?[=:]?\s*(\d+)\b", line, re.I)
            task = re.search(r"\btask\s*(?:id\s*)?[=:]?\s*(\d+)\b", line, re.I)
            current_identity = (slot.group(1) if slot else None, task.group(1) if task else None)
            if current_identity == (None, None):
                current_identity = None
            if current_identity is not None and current_identity == previous_identity:
                repeated += 1
            previous_identity = current_identity
        else:
            previous_identity = None
    counts["repeated_progress_1_00"] = repeated
    counts["progress_1_00"] = progress_lines
    return counts


class ChildLogProbe:
    """Count selected child-log events appended between snapshots."""

    def __init__(self, path: str, max_bytes: int = 8 * 1024 * 1024):
        self.path = os.fspath(path)
        self.max_bytes = max(1024, int(max_bytes))

    def snapshot(self) -> dict[str, Any]:
        try:
            stat = os.stat(self.path)
            size = int(stat.st_size)
            offset = size
            with open(self.path, "rb") as stream:
                head = stream.read(128)
            return {"status": "ok", "error": None, "path": self.path, "offset": offset,
                    "identity": [stat.st_dev, stat.st_ino],
                    "size": size, "fingerprint": head.hex(), "fingerprint_size": min(size, 128),
                    "counters": {name: 0 for name in (*_LOG_PATTERNS, "repeated_progress_1_00", "progress_1_00")}}
        except OSError as exc:
            return {"status": "missing", "error": str(exc), "path": self.path, "offset": 0, "size": 0,
                    "counters": {}}

    def delta(self, before: dict[str, Any] | None, after: dict[str, Any] | None = None) -> dict[str, Any]:
        if after is None:
            after = self.snapshot()
        if not before or before.get("status") != "ok" or after.get("status") != "ok":
            error = (before or {}).get("error") or after.get("error") or "missing snapshot"
            return {"counters": {}, "errors": [error], "reset": False}
        try:
            old_offset = int(before.get("offset", 0))
            old_size = int(before.get("size", old_offset))
            new_size = int(after.get("size", 0))
            with open(self.path, "rb") as check:
                current_head = check.read(int(before.get("fingerprint_size", 128))).hex()
            if (before.get("identity") != after.get("identity") or before.get("fingerprint") != current_head
                    or new_size < old_offset or new_size < old_size):
                return {"counters": {}, "errors": ["log truncated or replaced"], "reset": True}
            with open(self.path, "rb") as stream:
                stream.seek(old_offset)
                appended = max(new_size - old_offset, 0)
                data = stream.read(min(appended, self.max_bytes))
            counters = _count_log_text(data.decode("utf-8", errors="replace"))
            errors = ["appended log interval exceeds read cap"] if appended > len(data) else []
            return {"counters": counters, "errors": errors, "reset": False, "partial": bool(errors), "bytes": len(data)}
        except (OSError, ValueError) as exc:
            return {"counters": {}, "errors": [str(exc)], "reset": False}


def child_log_delta(probe: ChildLogProbe, before: dict[str, Any], after: dict[str, Any] | None = None) -> dict[str, Any]:
    """Functional alias useful to callers that keep probes in a registry."""
    return probe.delta(before, after)
