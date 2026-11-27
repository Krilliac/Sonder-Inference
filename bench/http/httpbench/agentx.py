"""InferenceX AgentX-style Weka trace loader, prompt synthesizer, and replay engine.

Targets the public Hugging Face corpus ``semianalysisai/cc-traces-weka-062126``
(a single ``traces.jsonl`` containing coding-agent traces with subagents) using
local JSON/JSONL trace files only — never downloads datasets implicitly and
requires only the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .textgen import Sizer, WORDS

DATASET_ID = "semianalysisai/cc-traces-weka-062126"
DATASET_FILE = "traces.jsonl"
DATASET_VARIANT = "with-subagents"
DEFAULT_BLOCK_SIZE = 64
DEFAULT_MAX_CONTEXT_LENGTH = 32768
DEFAULT_IDLE_GAP_CAP_S = 10.0
DEFAULT_CONCURRENCY_LEVELS = (1, 2, 4, 8)
TRACE_FIT_MODES = ("scale", "drop")
DEFAULT_TRACE_FIT = "scale"


def _reject_nonfinite(constant: str) -> Any:
    raise ValueError(f"non-finite JSON constant not allowed in trace: {constant!r}")


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _strict_float(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}, got {value!r}")
    return out


def _clean_ident(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    cleaned = value.strip()
    if any(ord(c) < 32 or ord(c) == 127 for c in cleaned):
        raise ValueError(f"{name} contains control characters")
    return cleaned


@dataclass(frozen=True)
class TraceRequest:
    t: float
    req_type: str
    input_tokens: int
    output_tokens: int
    hash_ids: tuple[int, ...]
    api_time: float | None = None
    think_time: float | None = None
    ttft: float | None = None
    model: str | None = None

    @property
    def effective_output_tokens(self) -> int:
        # SI /v1/chat/completions requires max_tokens >= 1 (src/server/src/openai.cpp).
        return max(1, self.output_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.effective_output_tokens


@dataclass(frozen=True)
class SubagentBlock:
    t: float
    agent_id: str
    subagent_type: str
    duration_ms: float | None
    requests: tuple[TraceRequest, ...]


@dataclass(frozen=True)
class WekaTrace:
    trace_id: str
    block_size: int
    hash_id_scope: str
    models: tuple[str, ...]
    items: tuple[TraceRequest | SubagentBlock, ...]
    original_peak_tokens: int | None = None
    scale_factor: float = 1.0

    @property
    def all_requests(self) -> list[TraceRequest]:
        out: list[TraceRequest] = []
        for item in self.items:
            if isinstance(item, SubagentBlock):
                out.extend(item.requests)
            else:
                out.append(item)
        return out

    @property
    def peak_context_tokens(self) -> int:
        reqs = self.all_requests
        return max((r.total_tokens for r in reqs), default=0)

    @property
    def total_requests(self) -> int:
        return len(self.all_requests)

    @property
    def subagent_count(self) -> int:
        return sum(1 for item in self.items if isinstance(item, SubagentBlock))


@dataclass
class LoadedTraceBatch:
    traces: list[WekaTrace]
    traces_loaded: int
    skipped_traces: list[dict[str, Any]]
    scaled_traces: list[dict[str, Any]]
    trace_fit: str
    max_context_length: int
    truncated_at_max_traces: bool = False


def _parse_request(obj: Any, where: str) -> TraceRequest:
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must be a JSON object")
    t = _strict_float(obj.get("t", 0.0), f"{where}.t", minimum=0.0)
    raw_in = obj.get("in") if "in" in obj else obj.get("input_length")
    raw_out = obj.get("out") if "out" in obj else obj.get("output_length")
    if raw_in is None:
        raise ValueError(f"{where} missing required 'in' (or 'input_length') token count")
    if raw_out is None:
        raise ValueError(f"{where} missing required 'out' (or 'output_length') token count")
    in_tok = _strict_int(raw_in, f"{where}.in", minimum=1)
    # Real traces in semianalysisai/cc-traces-weka-062126 contain out=0 turns.
    out_tok = _strict_int(raw_out, f"{where}.out", minimum=0)
    raw_hashes = obj.get("hash_ids", [])
    if not isinstance(raw_hashes, list):
        raise ValueError(f"{where}.hash_ids must be a list of integers")
    hashes = tuple(_strict_int(h, f"{where}.hash_ids[{i}]", minimum=0) for i, h in enumerate(raw_hashes))
    api_time = (
        _strict_float(obj["api_time"], f"{where}.api_time", minimum=0.0)
        if obj.get("api_time") is not None
        else None
    )
    think_time = (
        _strict_float(obj["think_time"], f"{where}.think_time", minimum=0.0)
        if obj.get("think_time") is not None
        else None
    )
    ttft = (
        _strict_float(obj["ttft"], f"{where}.ttft", minimum=0.0)
        if obj.get("ttft") is not None
        else None
    )
    req_type = str(obj.get("type") or "s")
    model = str(obj["model"]) if isinstance(obj.get("model"), str) and obj.get("model") else None
    return TraceRequest(
        t=t,
        req_type=req_type,
        input_tokens=in_tok,
        output_tokens=out_tok,
        hash_ids=hashes,
        api_time=api_time,
        think_time=think_time,
        ttft=ttft,
        model=model,
    )


def _parse_subagent(obj: dict[str, Any], where: str, idx: int) -> SubagentBlock:
    t = _strict_float(obj.get("t", 0.0), f"{where}.t", minimum=0.0)
    raw_aid = obj.get("agent_id")
    agent_id = _clean_ident(raw_aid, f"{where}.agent_id") if raw_aid is not None else f"subagent_{idx}"
    subagent_type = str(obj.get("subagent_type") or "Subagent")
    duration_ms = (
        _strict_float(obj["duration_ms"], f"{where}.duration_ms", minimum=0.0)
        if obj.get("duration_ms") is not None
        else None
    )
    raw_reqs = obj.get("requests")
    if not isinstance(raw_reqs, list) or not raw_reqs:
        raise ValueError(f"{where}.requests must be a non-empty list")
    reqs = tuple(_parse_request(r, f"{where}.requests[{j}]") for j, r in enumerate(raw_reqs))
    return SubagentBlock(
        t=t,
        agent_id=agent_id,
        subagent_type=subagent_type,
        duration_ms=duration_ms,
        requests=reqs,
    )


def parse_weka_trace(obj: Any, where: str = "trace") -> WekaTrace:
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must be a JSON object")
    trace_id = _clean_ident(obj.get("id") if "id" in obj else obj.get("session_id"), f"{where}.id")
    block_size = _strict_int(obj.get("block_size", DEFAULT_BLOCK_SIZE), f"{where}.block_size", minimum=1)
    hash_id_scope = str(obj.get("hash_id_scope") or "local")
    raw_models = obj.get("models") or []
    if not isinstance(raw_models, list) or not all(isinstance(m, str) for m in raw_models):
        raise ValueError(f"{where}.models must be a list of strings")
    raw_items = obj.get("requests")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"{where}.requests must be a non-empty list")
    items: list[TraceRequest | SubagentBlock] = []
    for i, entry in enumerate(raw_items):
        entry_where = f"{where}.requests[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{entry_where} must be a JSON object")
        if entry.get("type") == "subagent":
            items.append(_parse_subagent(entry, entry_where, i))
        else:
            items.append(_parse_request(entry, entry_where))
    return WekaTrace(
        trace_id=trace_id,
        block_size=block_size,
        hash_id_scope=hash_id_scope,
        models=tuple(raw_models),
        items=tuple(items),
    )


def _iter_one_file(path: str) -> Iterator[WekaTrace]:
    """Stream Weka traces one record at a time without buffering large JSONL files."""
    yielded = 0
    if path.lower().endswith(".jsonl"):
        with open(path, encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text, parse_constant=_reject_nonfinite)
                except ValueError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                yield parse_weka_trace(obj, f"{path}:{lineno}")
                yielded += 1
        if yielded == 0:
            raise ValueError(f"trace file contains no records: {path}")
        return

    # For .json (or other suffix): try streaming line-by-line if line 1 is a
    # standalone JSON object; otherwise parse as a single JSON object or array.
    with open(path, encoding="utf-8") as stream:
        first_lineno = 0
        first_text = ""
        for lineno, line in enumerate(stream, 1):
            stripped = line.strip()
            if stripped:
                first_lineno = lineno
                first_text = stripped
                break
        if not first_text:
            raise ValueError(f"trace file is empty: {path}")
        if first_text.startswith("{"):
            try:
                first_obj = json.loads(first_text, parse_constant=_reject_nonfinite)
            except ValueError:
                first_obj = None
            if isinstance(first_obj, dict):
                yield parse_weka_trace(first_obj, f"{path}:{first_lineno}")
                yielded += 1
                for lineno, line in enumerate(stream, first_lineno + 1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        obj = json.loads(text, parse_constant=_reject_nonfinite)
                    except ValueError as exc:
                        raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                    yield parse_weka_trace(obj, f"{path}:{lineno}")
                    yielded += 1
                return
        rest = stream.read()
    full_text = first_text + ("\n" + rest if rest else "")
    try:
        parsed = json.loads(full_text, parse_constant=_reject_nonfinite)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError(f"trace array is empty: {path}")
        for i, item in enumerate(parsed):
            yield parse_weka_trace(item, f"{path}[{i}]")
        return
    yield parse_weka_trace(parsed, path)


def iter_weka_traces(path: str | os.PathLike[str]) -> Iterator[WekaTrace]:
    """Yield Weka-format coding-agent traces lazily from a local JSON/JSONL file or directory."""
    fs_path = os.fspath(path)
    if not os.path.exists(fs_path):
        raise FileNotFoundError(f"trace path does not exist: {fs_path}")
    if os.path.isdir(fs_path):
        files = sorted(
            os.path.join(fs_path, name)
            for name in os.listdir(fs_path)
            if name.lower().endswith((".json", ".jsonl")) and os.path.isfile(os.path.join(fs_path, name))
        )
        if not files:
            raise ValueError(f"no .json or .jsonl trace files found in {fs_path}")
        for file_path in files:
            yield from _iter_one_file(file_path)
        return
    yield from _iter_one_file(fs_path)


def load_weka_traces(path: str | os.PathLike[str]) -> list[WekaTrace]:
    """Load all Weka traces from a local path (prefer ``load_and_fit_weka_traces`` for large corpora)."""
    return list(iter_weka_traces(path))


def _scale_request(req: TraceRequest, ratio: float, limit: int) -> TraceRequest:
    if req.output_tokens == 0:
        scaled_out = 0
    else:
        scaled_out = max(1, int(math.floor(req.output_tokens * ratio)))
    eff_out = max(1, scaled_out)
    if eff_out >= limit:
        eff_out = max(1, limit - 1)
        scaled_out = eff_out if req.output_tokens > 0 else 0
    max_in = max(1, limit - eff_out)
    scaled_in = max(1, min(max_in, int(math.floor(req.input_tokens * ratio))))

    # Position-indexed stride subsampling preserves shared leading hash blocks
    # across growing main turns and subagent branches because index j always
    # maps to original position floor(j / ratio), independent of request length.
    if req.hash_ids:
        n_blocks = max(1, min(len(req.hash_ids), int(math.floor(len(req.hash_ids) * ratio))))
        last_idx = len(req.hash_ids) - 1
        scaled_hashes = tuple(req.hash_ids[min(last_idx, int(j / ratio))] for j in range(n_blocks))
    else:
        scaled_hashes = ()

    return TraceRequest(
        t=req.t,
        req_type=req.req_type,
        input_tokens=scaled_in,
        output_tokens=scaled_out,
        hash_ids=scaled_hashes,
        api_time=req.api_time,
        think_time=req.think_time,
        ttft=req.ttft,
        model=req.model,
    )


def scale_trace_to_context(
    trace: WekaTrace,
    max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
) -> WekaTrace:
    """Proportionally shrink a trace's input/output lengths and hash blocks to fit ``max_context_length``."""
    limit = _strict_int(max_context_length, "max_context_length", minimum=1)
    peak = trace.peak_context_tokens
    if peak <= limit:
        return trace
    if limit < 2:
        raise ValueError("max_context_length must be >= 2 to scale a trace (requires >= 1 input + >= 1 output token)")
    ratio = limit / float(peak)
    scaled_items: list[TraceRequest | SubagentBlock] = []
    for item in trace.items:
        if isinstance(item, SubagentBlock):
            scaled_reqs = tuple(_scale_request(r, ratio, limit) for r in item.requests)
            scaled_items.append(
                SubagentBlock(
                    t=item.t,
                    agent_id=item.agent_id,
                    subagent_type=item.subagent_type,
                    duration_ms=item.duration_ms,
                    requests=scaled_reqs,
                )
            )
        else:
            scaled_items.append(_scale_request(item, ratio, limit))
    return WekaTrace(
        trace_id=trace.trace_id,
        block_size=trace.block_size,
        hash_id_scope=trace.hash_id_scope,
        models=trace.models,
        items=tuple(scaled_items),
        original_peak_tokens=peak,
        scale_factor=ratio,
    )


def load_and_fit_weka_traces(
    path: str | os.PathLike[str],
    *,
    max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
    trace_fit: str = DEFAULT_TRACE_FIT,
    max_traces: int | None = None,
) -> LoadedTraceBatch:
    """Stream traces line-by-line, fitting (scaling or dropping) each trace immediately.

    Rejected traces (and unscaled originals in ``scale`` mode) are discarded as
    soon as they are inspected so multi-gigabyte JSONL corpora never retain
    millions of unused ``hash_ids`` in RAM.
    """
    limit = _strict_int(max_context_length, "max_context_length", minimum=1)
    if trace_fit not in TRACE_FIT_MODES:
        raise ValueError(f"trace_fit must be one of {TRACE_FIT_MODES}, got {trace_fit!r}")
    if max_traces is not None:
        max_traces = _strict_int(max_traces, "max_traces", minimum=1)

    kept: list[WekaTrace] = []
    skipped: list[dict[str, Any]] = []
    scaled_meta: list[dict[str, Any]] = []
    loaded = 0
    truncated = False

    gen = iter_weka_traces(path)
    try:
        for trace in gen:
            loaded += 1
            peak = trace.peak_context_tokens
            if peak <= limit:
                kept.append(trace)
            elif trace_fit == "drop" or limit < 2:
                skipped.append({
                    "trace_id": trace.trace_id,
                    "peak_tokens": peak,
                    "max_context_length": limit,
                    "reason": f"peak request tokens ({peak}) exceed max_context_length ({limit})",
                })
                # Drop trace reference immediately so its hash tuples can be collected.
                del trace
            else:
                fitted = scale_trace_to_context(trace, limit)
                scaled_meta.append({
                    "trace_id": fitted.trace_id,
                    "original_peak_tokens": peak,
                    "scaled_peak_tokens": fitted.peak_context_tokens,
                    "scale_factor": fitted.scale_factor,
                })
                kept.append(fitted)
                del trace
            if max_traces is not None and len(kept) >= max_traces:
                truncated = True
                break
    finally:
        gen.close()

    return LoadedTraceBatch(
        traces=kept,
        traces_loaded=loaded,
        skipped_traces=skipped,
        scaled_traces=scaled_meta,
        trace_fit=trace_fit,
        max_context_length=limit,
        truncated_at_max_traces=truncated,
    )


def filter_traces_by_context(
    traces: list[WekaTrace],
    max_context_length: int = DEFAULT_MAX_CONTEXT_LENGTH,
) -> tuple[list[WekaTrace], list[dict[str, Any]]]:
    """Keep traces whose peak request (in + max(1, out)) fits ``max_context_length``."""
    limit = _strict_int(max_context_length, "max_context_length", minimum=1)
    kept: list[WekaTrace] = []
    skipped: list[dict[str, Any]] = []
    for trace in traces:
        peak = trace.peak_context_tokens
        if peak > limit:
            skipped.append({
                "trace_id": trace.trace_id,
                "peak_tokens": peak,
                "max_context_length": limit,
                "reason": f"peak request tokens ({peak}) exceed max_context_length ({limit})",
            })
        else:
            kept.append(trace)
    return kept, skipped


class PromptBlockSynthesizer:
    """Deterministically synthesizes synthetic-token prompts from Weka ``hash_ids``.

    Identical ``(session_salt, hash_id)`` pairs always yield byte-identical text
    blocks, preserving growing shared prefixes across main turns and shared
    ancestor blocks across subagent branches while isolating concurrent users
    via distinct ``session_salt`` values.
    """

    def __init__(self, sizer: Sizer, *, session_salt: str, block_size: int = DEFAULT_BLOCK_SIZE):
        self.sizer = sizer
        self.session_salt = session_salt
        self.block_size = _strict_int(block_size, "block_size", minimum=1)
        self.block_chars = max(1, round(self.block_size * self.sizer.chars_per_token))
        self._blocks: dict[int, str] = {}
        self._lock = threading.Lock()

    def _synthesize_chars(self, key: str, target_chars: int) -> str:
        digest = hashlib.sha256(f"{self.session_salt}|{key}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rnd = random.Random(seed)
        tag = f"[{digest[:4].hex()}] "
        chunks: list[str] = [tag]
        total = len(tag)
        while total < target_chars:
            word = rnd.choice(WORDS) + " "
            chunks.append(word)
            total += len(word)
        text = "".join(chunks)
        return text[:target_chars]

    def block(self, hash_id: int) -> str:
        with self._lock:
            existing = self._blocks.get(hash_id)
            if existing is not None:
                return existing
            created = self._synthesize_chars(f"blk:{hash_id}", self.block_chars)
            self._blocks[hash_id] = created
            return created

    def build_prompt(self, req: TraceRequest, *, tail_key: str = "tail") -> str:
        target_chars = max(1, round(req.input_tokens * self.sizer.chars_per_token))
        prefix = "".join(self.block(h) for h in req.hash_ids)
        if len(prefix) == target_chars:
            return prefix
        if len(prefix) > target_chars:
            return prefix[:target_chars]
        rem = target_chars - len(prefix)
        return prefix + self._synthesize_chars(f"tail:{tail_key}:{req.input_tokens}", rem)


def _capped_gap(raw_gap: float, cap_s: float) -> float:
    return min(max(0.0, raw_gap), cap_s)


def replay_trace_session(
    trace: WekaTrace,
    *,
    sizer: Sizer,
    session_id: str,
    session_salt: str,
    send_fn: Callable[[dict[str, Any], str], dict[str, Any]],
    idle_gap_cap_s: float = DEFAULT_IDLE_GAP_CAP_S,
    idle_time_scale: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Replay one Weka coding-agent session including concurrent subagent branches."""
    cap = _strict_float(idle_gap_cap_s, "idle_gap_cap_s", minimum=0.0)
    scale = _strict_float(idle_time_scale, "idle_time_scale", minimum=0.0)
    synth = PromptBlockSynthesizer(sizer, session_salt=session_salt, block_size=trace.block_size)
    seen_lock = threading.Lock()
    seen_hashes: set[int] = set()

    def _sleep_gap(capped_s: float) -> None:
        wait = capped_s * scale
        if wait > 0.0:
            sleep_fn(wait)

    def _classify_and_record_hashes(hash_ids: tuple[int, ...]) -> tuple[str, int]:
        with seen_lock:
            shared = 0
            for h in hash_ids:
                if h in seen_hashes:
                    shared += 1
                else:
                    break
            seen_hashes.update(hash_ids)
        return ("warm" if shared > 0 else "cold"), shared

    def _execute_request(
        req: TraceRequest,
        *,
        branch: str,
        is_subagent: bool,
        agent_id: str | None,
        turn: int,
        raw_gap_s: float,
        capped_gap_s: float,
    ) -> dict[str, Any]:
        cache_state, shared_blocks = _classify_and_record_hashes(req.hash_ids)
        prompt = synth.build_prompt(req, tail_key=f"{branch}:t{turn}")
        eff_out = req.effective_output_tokens
        meta = {
            "session_id": session_id,
            "trace_id": trace.trace_id,
            "trace_scale_factor": trace.scale_factor,
            "branch": branch,
            "is_subagent": is_subagent,
            "agent_id": agent_id,
            "turn": turn,
            "target_input_tokens": req.input_tokens,
            "target_output_tokens": req.output_tokens,
            "effective_output_tokens": eff_out,
            "output_tokens_clamped": req.output_tokens < 1,
            "shared_prefix_blocks": shared_blocks,
            "hash_block_count": len(req.hash_ids),
            "raw_idle_gap_s": raw_gap_s,
            "idle_gap_s": capped_gap_s,
            "cache_state": cache_state,
            "cold": cache_state == "cold",
        }
        row = dict(send_fn(meta, prompt))
        row.update(meta)
        raw_actual = row.get("committed_output_tokens")
        if raw_actual is None:
            raw_actual = row.get("completion_tokens")
        if isinstance(raw_actual, int) and not isinstance(raw_actual, bool):
            actual_out: int | None = raw_actual
        else:
            actual_out = None
        row["actual_output_tokens"] = actual_out
        row["output_tokens_delta"] = (actual_out - eff_out) if actual_out is not None else None
        row["output_tokens_ratio"] = (actual_out / float(eff_out)) if (actual_out is not None and eff_out > 0) else None
        return row

    results: list[dict[str, Any]] = []
    prev_end_t: float | None = None
    main_turn = 0
    idx = 0
    n_items = len(trace.items)

    while idx < n_items:
        item = trace.items[idx]
        if isinstance(item, TraceRequest):
            main_turn += 1
            if item.think_time is not None:
                raw_gap = item.think_time
            elif prev_end_t is not None:
                raw_gap = max(0.0, item.t - prev_end_t)
            else:
                raw_gap = 0.0
            capped = _capped_gap(raw_gap, cap)
            _sleep_gap(capped)
            row = _execute_request(
                item,
                branch="main",
                is_subagent=False,
                agent_id=None,
                turn=main_turn,
                raw_gap_s=raw_gap,
                capped_gap_s=capped,
            )
            results.append(row)
            prev_end_t = item.t + (item.api_time or 0.0)
            idx += 1
            continue

        # Group consecutive subagent blocks so sibling subagents execute concurrently
        # and join before the parent's next main-agent turn.
        batch: list[SubagentBlock] = []
        while idx < n_items and isinstance(trace.items[idx], SubagentBlock):
            batch.append(trace.items[idx])  # type: ignore[arg-type]
            idx += 1

        batch_start_t = min(sub.t for sub in batch)
        if prev_end_t is not None:
            batch_raw_gap = max(0.0, batch_start_t - prev_end_t)
            _sleep_gap(_capped_gap(batch_raw_gap, cap))

        def _run_subagent(sub: SubagentBlock) -> list[dict[str, Any]]:
            sub_rows: list[dict[str, Any]] = []
            start_offset = max(0.0, sub.t - batch_start_t)
            if start_offset > 0.0:
                _sleep_gap(_capped_gap(start_offset, cap))
            sub_prev_end: float | None = None
            for s_turn, s_req in enumerate(sub.requests, 1):
                if s_req.think_time is not None:
                    s_raw = s_req.think_time
                elif sub_prev_end is not None:
                    s_raw = max(0.0, s_req.t - sub_prev_end)
                else:
                    s_raw = start_offset if s_turn == 1 else 0.0
                s_capped = _capped_gap(s_raw, cap)
                if s_turn > 1:
                    _sleep_gap(s_capped)
                sub_rows.append(
                    _execute_request(
                        s_req,
                        branch=f"subagent:{sub.agent_id}",
                        is_subagent=True,
                        agent_id=sub.agent_id,
                        turn=s_turn,
                        raw_gap_s=s_raw,
                        capped_gap_s=s_capped,
                    )
                )
                sub_prev_end = s_req.t + (s_req.api_time or 0.0)
            return sub_rows

        if len(batch) == 1:
            results.extend(_run_subagent(batch[0]))
        else:
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                for sub_rows in ex.map(_run_subagent, batch):
                    results.extend(sub_rows)

        sub_ends: list[float] = []
        for sub in batch:
            if sub.duration_ms is not None:
                sub_ends.append(sub.t + sub.duration_ms / 1000.0)
            elif sub.requests:
                last_r = sub.requests[-1]
                sub_ends.append(last_r.t + (last_r.api_time or 0.0))
        if sub_ends:
            prev_end_t = max(prev_end_t or 0.0, max(sub_ends))

    return results
