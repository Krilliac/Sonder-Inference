"""Small, dependency-free probes for server-side benchmark observations.

The probes are deliberately independent of the request client.  A metrics URL
may point at a different origin and therefore never inherits request headers.
Snapshots are JSON serialisable and retain enough state for callers to report
unknown data and file truncation honestly.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .metrics import classify_cache_state


PROVENANCE_SCHEMA = "sonder.bench.http.provenance/1"

REQUIRED_PROVENANCE_FIELDS = (
    "source_commit",
    "upstream_revision_or_digest",
    "model_sha256",
    "draft_sha256",
    "resolved_settings",
    "effective_settings",
    "corpus_revision",
    "sample_count",
    "timing_definition",
    "validation_outcome",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")

DEFAULT_TIMING_DEFINITION: dict[str, str] = {
    "client_ttft_ms": "monotonic elapsed ms from HTTP request write to first token-bearing SSE delta (content, reasoning_content, or tool_calls)",
    "client_itl_ms": "monotonic gap ms between consecutive token-bearing SSE chunks divided by tokens carried in the arriving chunk",
    "client_e2e_ms": "monotonic elapsed ms from HTTP request write to stream completion ([DONE] or final stop event)",
    "backend_prefill_ms": "server-reported timings.prompt_ms (backend prompt evaluation compute, excluding network/queue)",
    "backend_decode_ms": "server-reported timings.predicted_ms (backend token generation compute, excluding network/queue)",
    "backend_compute_ms": "backend_prefill_ms + backend_decode_ms when both are reported by the server (null/unknown when absent)",
    "committed_output_tokens": "target-verified output tokens emitted to the client (usage.completion_tokens / timings.predicted_n)",
    "proposed_draft_tokens": "speculative draft tokens proposed before verification (timings.draft_n; null/unknown when unreported)",
    "cold_vs_warm_rule": "explicit cache_state/cold flag, else turn/repeat index (1=cold, >=2=warm), else reported cached tokens (>0=warm, 0=cold, null=unclassified)",
    "percentiles": "linear interpolation (numpy default linear method) over ok requests; ITL pooled across token gaps of ok requests",
}


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
        except (OSError, URLError, TimeoutError, ValueError, http.client.HTTPException) as exc:
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


# ------------------------------------------------------------------ provenance

def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Compute a streaming lowercase SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def validate_sha256_hex(value: str, *, allow_special: tuple[str, ...] = ()) -> str:
    """Validate a 64-char SHA-256 hex digest (or an explicitly allowed sentinel)."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("SHA-256 digest must be a non-empty string")
    cleaned = value.strip().lower()
    if cleaned in allow_special:
        return cleaned
    if not _SHA256_RE.fullmatch(cleaned):
        allowed = f" or one of {allow_special}" if allow_special else ""
        raise ValueError(f"invalid SHA-256 digest {value!r}: expected 64 hex chars{allowed}")
    return cleaned


def corpus_revision_for(cfg: dict[str, Any]) -> str:
    """Deterministic SHA-256 revision tag over canonical scenario configuration JSON."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def detect_git_commit(cwd: str | None = None) -> str:
    """Best-effort current git HEAD commit SHA, or ``unknown`` when unavailable."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    sha = proc.stdout.strip().lower()
    return sha if _GIT_SHA_RE.fullmatch(sha) else "unknown"


def _extract_sonder_identity(server: dict[str, Any]) -> dict[str, Any]:
    sid = (server.get("sonder_identity") or {}).get("body")
    return sid if isinstance(sid, dict) else {}


def _extract_props_body(server: dict[str, Any]) -> dict[str, Any]:
    props = (server.get("props") or {}).get("body")
    return props if isinstance(props, dict) else {}


def _missing_provenance_fields(record: dict[str, Any]) -> list[str]:
    missing = []
    for key in ("source_commit", "upstream_revision_or_digest", "model_sha256", "draft_sha256"):
        val = record.get(key)
        if not isinstance(val, str) or not val.strip() or val.strip().lower() == "unknown":
            missing.append(key)
    return missing


def summarize_validation_outcome(
    checks: list[dict[str, Any]],
    verdict_value: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the structured ``validation_outcome`` sub-record from final checks."""
    passed = [c["check"] for c in checks if c.get("status") == "pass"]
    warned = [c["check"] for c in checks if c.get("status") == "warn"]
    failed = [c["check"] for c in checks if c.get("status") == "fail"]
    skipped = [c["check"] for c in checks if c.get("status") == "skip"]
    missing = _missing_provenance_fields(provenance)
    return {
        "verdict": verdict_value,
        "checks_total": len(checks),
        "checks_passed": len(passed),
        "checks_warned": len(warned),
        "checks_failed": len(failed),
        "checks_skipped": len(skipped),
        "failed_checks": failed,
        "warned_checks": warned,
        "provenance_complete": len(missing) == 0,
        "missing_provenance_fields": missing,
    }


def _sample_counts(results: dict[str, Any]) -> dict[str, int]:
    scenarios = results.get("scenarios") or []
    expected = sum(int(s.get("expected_requests") or 0) for s in scenarios)
    reqs = [r for s in scenarios for r in (s.get("requests") or [])]
    ok_reqs = [r for r in reqs if r.get("ok")]
    cold = 0
    warm = 0
    unclassified = 0
    for r in ok_reqs:
        state = classify_cache_state(r)
        if state == "cold":
            cold += 1
        elif state == "warm":
            warm += 1
        else:
            unclassified += 1
    return {
        "expected": expected,
        "completed": len(reqs),
        "ok": len(ok_reqs),
        "failed": len(reqs) - len(ok_reqs),
        "cold": cold,
        "warm": warm,
        "unclassified": unclassified,
    }


def build_provenance(
    results: dict[str, Any],
    cfg: dict[str, Any],
    *,
    checks: list[dict[str, Any]] | None = None,
    verdict_value: str = "pass",
    overrides: dict[str, Any] | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Assemble the strict reproducible benchmark provenance record."""
    ov = dict(overrides or {})
    server = results.get("server") or {}
    target = results.get("target") or {}
    props = _extract_props_body(server)
    sid = _extract_sonder_identity(server)
    backend_id = sid.get("backend_identity") if isinstance(sid.get("backend_identity"), dict) else {}

    source_commit = str(ov.get("source_commit") or "").strip() or detect_git_commit(cwd)

    upstream = str(ov.get("upstream_revision_or_digest") or "").strip()
    if not upstream and ov.get("upstream_exe"):
        upstream = "sha256:" + sha256_file(ov["upstream_exe"])
    if not upstream:
        for candidate in (
            server.get("build_info"),
            backend_id.get("upstream_revision"),
            backend_id.get("executable_sha256"),
            backend_id.get("version"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                upstream = candidate.strip()
                break
    if not upstream:
        upstream = "unknown"

    model_sha256 = "unknown"
    if ov.get("model_sha256"):
        model_sha256 = validate_sha256_hex(str(ov["model_sha256"]), allow_special=("unknown",))
    elif ov.get("model_file"):
        model_sha256 = sha256_file(ov["model_file"])
    else:
        for candidate in (backend_id.get("model_sha256"), sid.get("model_sha256")):
            if isinstance(candidate, str) and _SHA256_RE.fullmatch(candidate.strip()):
                model_sha256 = candidate.strip().lower()
                break

    draft_sha256 = "none"
    if ov.get("draft_sha256"):
        draft_sha256 = validate_sha256_hex(str(ov["draft_sha256"]), allow_special=("none", "unknown"))
    elif ov.get("draft_file"):
        draft_sha256 = sha256_file(ov["draft_file"])
    else:
        found_digest = None
        for candidate in (backend_id.get("draft_sha256"), sid.get("draft_sha256")):
            if isinstance(candidate, str) and _SHA256_RE.fullmatch(candidate.strip()):
                found_digest = candidate.strip().lower()
                break
        if found_digest is not None:
            draft_sha256 = found_digest
        elif backend_id.get("draft_model") or backend_id.get("draft_model_path") or props.get("draft_model_path"):
            draft_sha256 = "unknown"

    resolved_settings: dict[str, Any] = {
        "base_url": target.get("base_url"),
        "model": target.get("model"),
        "ctx_size": server.get("n_ctx"),
        "ctx_source": server.get("ctx_source"),
        "defaults": dict(target.get("defaults") or {}),
        "extra_body": dict(target.get("extra_body") or {}),
        "prompt_sizing": results.get("prompt_sizing"),
        "total_slots": props.get("total_slots"),
        "server_default_generation_settings": props.get("default_generation_settings"),
    }
    if isinstance(ov.get("resolved_settings"), dict):
        resolved_settings.update(ov["resolved_settings"])

    all_reqs = [r for s in (results.get("scenarios") or []) for r in (s.get("requests") or [])]
    ok_reqs = [r for r in all_reqs if r.get("ok")]
    observed_models = sorted({str(r["model"]) for r in ok_reqs if r.get("model")})
    observed_warnings: list[Any] = []
    for r in all_reqs:
        w = r.get("sonder_warnings")
        if isinstance(w, list):
            for item in w:
                if item not in observed_warnings:
                    observed_warnings.append(item)
        elif w and w not in observed_warnings:
            observed_warnings.append(w)

    proposed_vals = [r.get("proposed_draft_tokens") for r in ok_reqs if r.get("proposed_draft_tokens") is not None]
    if not proposed_vals:
        draft_active: Any = "unknown"
    else:
        draft_active = any(v > 0 for v in proposed_vals)

    cached_vals = [r.get("cached_tokens") for r in ok_reqs if r.get("cached_tokens") is not None]
    if not cached_vals:
        cache_active: Any = "unknown"
    else:
        cache_active = any(v > 0 for v in cached_vals)

    dgs = props.get("default_generation_settings") if isinstance(props.get("default_generation_settings"), dict) else {}
    effective_settings: dict[str, Any] = {
        "observed_models": observed_models,
        "server_n_ctx": server.get("n_ctx"),
        "server_total_slots": props.get("total_slots"),
        "server_speculative": dgs.get("speculative", "unknown"),
        "sonder_warnings": observed_warnings,
        "draft_observed_active": draft_active,
        "prefix_cache_observed_active": cache_active,
    }
    if isinstance(ov.get("effective_settings"), dict):
        effective_settings.update(ov["effective_settings"])

    corpus_rev = str(ov.get("corpus_revision") or cfg.get("corpus_revision") or "").strip()
    if not corpus_rev:
        corpus_rev = corpus_revision_for(cfg)

    record: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "source_commit": source_commit,
        "upstream_revision_or_digest": upstream,
        "model_sha256": model_sha256,
        "draft_sha256": draft_sha256,
        "resolved_settings": resolved_settings,
        "effective_settings": effective_settings,
        "corpus_revision": corpus_rev,
        "sample_count": _sample_counts(results),
        "timing_definition": dict(DEFAULT_TIMING_DEFINITION),
    }
    record["validation_outcome"] = summarize_validation_outcome(checks or [], verdict_value, record)
    return record


def validate_provenance(record: Any, *, strict: bool = False) -> list[str]:
    """Validate a benchmark provenance record against the strict schema contract."""
    if not isinstance(record, dict):
        return ["provenance must be a JSON object"]
    errs: list[str] = []
    if record.get("schema") != PROVENANCE_SCHEMA:
        errs.append(f"provenance.schema must be {PROVENANCE_SCHEMA!r}, got {record.get('schema')!r}")
    for key in REQUIRED_PROVENANCE_FIELDS:
        if key not in record:
            errs.append(f"provenance missing required field {key!r}")
    for key in ("source_commit", "upstream_revision_or_digest", "corpus_revision"):
        val = record.get(key)
        if not isinstance(val, str) or not val.strip():
            errs.append(f"provenance.{key} must be a non-empty string")
    model_sha = record.get("model_sha256")
    if not isinstance(model_sha, str):
        errs.append("provenance.model_sha256 must be a string")
    else:
        try:
            validate_sha256_hex(model_sha, allow_special=("unknown",))
        except ValueError as exc:
            errs.append(f"provenance.model_sha256 invalid: {exc}")
    draft_sha = record.get("draft_sha256")
    if not isinstance(draft_sha, str):
        errs.append("provenance.draft_sha256 must be a string")
    else:
        try:
            validate_sha256_hex(draft_sha, allow_special=("none", "unknown"))
        except ValueError as exc:
            errs.append(f"provenance.draft_sha256 invalid: {exc}")
    for key in ("resolved_settings", "effective_settings", "timing_definition"):
        if not isinstance(record.get(key), dict) or not record[key]:
            errs.append(f"provenance.{key} must be a non-empty object")
    sc = record.get("sample_count")
    if not isinstance(sc, dict):
        errs.append("provenance.sample_count must be an object")
    else:
        for k in ("expected", "completed", "ok", "failed", "cold", "warm", "unclassified"):
            v = sc.get(k)
            if type(v) is not int or v < 0:
                errs.append(f"provenance.sample_count.{k} must be a non-negative integer")
        if all(type(sc.get(k)) is int for k in ("completed", "ok", "failed")):
            if sc["ok"] + sc["failed"] != sc["completed"]:
                errs.append("provenance.sample_count ok + failed must equal completed")
        if all(type(sc.get(k)) is int for k in ("ok", "cold", "warm", "unclassified")):
            if sc["cold"] + sc["warm"] + sc["unclassified"] != sc["ok"]:
                errs.append("provenance.sample_count cold + warm + unclassified must equal ok")
    vo = record.get("validation_outcome")
    if not isinstance(vo, dict):
        errs.append("provenance.validation_outcome must be an object")
    else:
        if vo.get("verdict") not in ("pass", "warn", "fail"):
            errs.append(f"provenance.validation_outcome.verdict invalid: {vo.get('verdict')!r}")
        for k in ("checks_total", "checks_passed", "checks_warned", "checks_failed", "checks_skipped"):
            if type(vo.get(k)) is not int or vo[k] < 0:
                errs.append(f"provenance.validation_outcome.{k} must be a non-negative integer")
        if not isinstance(vo.get("provenance_complete"), bool):
            errs.append("provenance.validation_outcome.provenance_complete must be a boolean")
    if strict:
        missing = _missing_provenance_fields(record)
        if missing:
            errs.append(f"strict provenance requires known values for {missing}")
    return errs
