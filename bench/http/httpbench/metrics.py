"""Metric math: per-request derivation from a stream trace, and summary stats.

Conventions (kept deliberately explicit so a result file can be audited):

- TTFT: request sent -> first chunk carrying generated text (content,
  reasoning_content or a tool-call delta).
- Inter-token latency (ITL): one sample per token-bearing chunk after the
  first, equal to the gap since the previous token-bearing chunk divided by
  the tokens that chunk carried. When a chunk does not say how many tokens it
  carries, the stream average (completion tokens / token chunks) is used.
- Prefix hit: cached prompt tokens / prompt tokens, where cached comes from
  ``timings.cache_n`` (llama-server) or
  ``usage.prompt_tokens_details.cached_tokens`` (OpenAI). When neither is
  reported the hit is ``None`` (unknown), never 0.
- Client decode tok/s: (completion tokens - 1) / (end - TTFT), matching
  bench/README.md. Client prefill tok/s: uncached prompt tokens / TTFT
  (includes network and queueing, so it is a lower bound).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable


def percentile(values: Iterable[float], p: float) -> float | None:
    """Linear-interpolated percentile (numpy's default 'linear' method)."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(values: Iterable[float | None]) -> dict[str, Any]:
    xs = [v for v in values if v is not None]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "p99": percentile(xs, 99),
        "min": min(xs),
        "max": max(xs),
    }


@dataclass
class Chunk:
    t: float                   # seconds since request start
    text: str = ""             # content delta
    reasoning: str = ""        # reasoning_content delta
    tool: bool = False         # carried a tool-call delta
    n_tokens: int | None = None  # explicit tokens in this chunk, when the server says

    @property
    def has_tokens(self) -> bool:
        return bool(self.text or self.reasoning or self.tool or (self.n_tokens or 0) > 0)


@dataclass
class StreamTrace:
    """Raw observations of one streamed request."""
    status: int | None = None
    chunks: list[Chunk] = field(default_factory=list)
    end_t: float | None = None
    usage: dict | None = None
    timings: dict | None = None
    extra_final: dict = field(default_factory=dict)  # /completion final fields
    finish_reason: str | None = None
    done: bool = False          # saw [DONE] or a stop:true final event
    error: str | None = None
    model: str | None = None


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def cached_tokens(usage: dict | None, timings: dict | None) -> tuple[int | None, str | None]:
    """Cached prompt tokens and where the number came from; (None, None) when unreported."""
    if timings and _num(timings.get("cache_n")) is not None:
        return int(timings["cache_n"]), "timings.cache_n"
    if usage:
        det = usage.get("prompt_tokens_details")
        if isinstance(det, dict) and _num(det.get("cached_tokens")) is not None:
            return int(det["cached_tokens"]), "usage.prompt_tokens_details.cached_tokens"
    return None, None


def prompt_tokens(usage: dict | None, timings: dict | None, extra: dict | None = None) -> int | None:
    """Total prompt tokens (cached + processed).

    llama-server's ``timings.prompt_n`` counts only the tokens it processed,
    so it is added to ``cache_n``. Servers differ on whether
    ``usage.prompt_tokens`` includes cached tokens, so the largest candidate
    wins.
    """
    cands: list[int] = []
    if usage and _num(usage.get("prompt_tokens")) is not None:
        cands.append(int(usage["prompt_tokens"]))
    if timings and _num(timings.get("prompt_n")) is not None:
        cands.append(int(timings["prompt_n"]) + int(_num(timings.get("cache_n")) or 0))
    if extra and _num(extra.get("tokens_evaluated")) is not None:
        cands.append(int(extra["tokens_evaluated"]))
    return max(cands) if cands else None


def completion_tokens(trace: StreamTrace) -> tuple[int, str]:
    if trace.usage and _num(trace.usage.get("completion_tokens")) is not None:
        return int(trace.usage["completion_tokens"]), "usage"
    if trace.timings and _num(trace.timings.get("predicted_n")) is not None:
        return int(trace.timings["predicted_n"]), "timings"
    if trace.extra_final and _num(trace.extra_final.get("tokens_predicted")) is not None:
        return int(trace.extra_final["tokens_predicted"]), "tokens_predicted"
    explicit = [c.n_tokens for c in trace.chunks if c.has_tokens and c.n_tokens]
    if explicit and len(explicit) == sum(1 for c in trace.chunks if c.has_tokens):
        return sum(explicit), "chunk_tokens"
    return sum(1 for c in trace.chunks if c.has_tokens), "chunks"


def inter_token_latencies(chunks: list[Chunk], total_tokens: int | None) -> list[float]:
    """Per-chunk ITL samples in seconds (gap / tokens in chunk)."""
    tok_chunks = [c for c in chunks if c.has_tokens]
    if len(tok_chunks) < 2:
        return []
    avg = 1.0
    if total_tokens and total_tokens > 0:
        avg = max(total_tokens / len(tok_chunks), 1e-9)
    out = []
    prev = tok_chunks[0].t
    for c in tok_chunks[1:]:
        per = c.n_tokens if c.n_tokens else avg
        out.append(max(c.t - prev, 0.0) / per)
        prev = c.t
    return out


def classify_cache_state(req: dict[str, Any]) -> str:
    """Classify one request as ``cold``, ``warm``, or ``unclassified``.

    Precedence:
      1. Explicit ``cache_state`` ("cold" or "warm").
      2. Explicit boolean ``cold`` (True -> cold, False -> warm).
      3. Multi-turn ``turn`` or ``repeat`` index (1 -> cold, >=2 -> warm).
      4. ``group == "warmup"`` -> cold.
      5. Reported cached token count (>0 -> warm, 0 -> cold, None -> unclassified).
    """
    explicit = req.get("cache_state")
    if explicit in ("cold", "warm"):
        return str(explicit)
    cold_flag = req.get("cold")
    if isinstance(cold_flag, bool):
        return "cold" if cold_flag else "warm"
    for key in ("turn", "repeat"):
        idx = req.get(key)
        if type(idx) is int and idx >= 1:
            return "cold" if idx == 1 else "warm"
    if req.get("group") == "warmup":
        return "cold"
    cached = req.get("cached_tokens")
    if cached is None:
        cached = req.get("cache_n")
    if isinstance(cached, (int, float)) and not isinstance(cached, bool):
        return "warm" if cached > 0 else "cold"
    return "unclassified"


def derive(trace: StreamTrace) -> dict[str, Any]:
    """Per-request metrics from a trace. Times in seconds unless named _ms."""
    tok_chunks = [c for c in trace.chunks if c.has_tokens]
    ttft = tok_chunks[0].t if tok_chunks else None
    e2e = trace.end_t
    ctoks, ctok_src = completion_tokens(trace)
    ptoks = prompt_tokens(trace.usage, trace.timings, trace.extra_final)
    cached, cached_src = cached_tokens(trace.usage, trace.timings)
    hit = None
    if cached is not None and ptoks:
        hit = min(cached / ptoks, 1.0)
    itl = inter_token_latencies(trace.chunks, ctoks)

    decode_client = None
    if ttft is not None and e2e is not None and ctoks > 1 and e2e > ttft:
        decode_client = (ctoks - 1) / (e2e - ttft)
    prefill_client = None
    if ttft and ptoks:
        processed = ptoks - cached if cached is not None else ptoks
        if processed > 0:
            prefill_client = processed / ttft

    t = trace.timings or {}
    cache_n = _num(t.get("cache_n"))
    prompt_n = _num(t.get("prompt_n"))
    draft_n = _num(t.get("draft_n"))
    draft_acc = _num(t.get("draft_n_accepted"))
    acceptance = draft_acc / draft_n if draft_n and draft_acc is not None else None

    # Keep committed target output tokens strictly separate from proposed/accepted
    # speculative draft tokens (never conflate rejected proposals into output).
    proposed_draft_tokens = int(draft_n) if draft_n is not None else None
    accepted_draft_tokens = int(draft_acc) if draft_acc is not None else None
    rejected_draft_tokens = (
        max(0, proposed_draft_tokens - accepted_draft_tokens)
        if proposed_draft_tokens is not None and accepted_draft_tokens is not None
        else None
    )

    # Keep backend compute timings (server timings.prompt_ms / predicted_ms)
    # strictly separate from client-observed wire latencies (TTFT / ITL / e2e).
    backend_prefill_ms = _num(t.get("prompt_ms"))
    backend_decode_ms = _num(t.get("predicted_ms"))
    backend_compute_ms = (
        backend_prefill_ms + backend_decode_ms
        if backend_prefill_ms is not None and backend_decode_ms is not None
        else None
    )
    client_ttft_ms = ttft * 1000.0 if ttft is not None else None
    client_e2e_ms = e2e * 1000.0 if e2e is not None else None
    client_itl_ms = [x * 1000.0 for x in itl]

    return {
        "ok": trace.error is None and trace.status == 200 and trace.done,
        "status": trace.status,
        "error": trace.error,
        "done": trace.done,
        "finish_reason": trace.finish_reason,
        "ttft_s": ttft,
        "e2e_s": e2e,
        "itl_s": itl,
        "client_observed_ms": {
            "ttft_ms": client_ttft_ms,
            "e2e_ms": client_e2e_ms,
            "itl_ms": summarize(client_itl_ms),
        },
        "backend_prefill_ms": backend_prefill_ms,
        "backend_decode_ms": backend_decode_ms,
        "backend_compute_ms": backend_compute_ms,
        "chunks": len(tok_chunks),
        "prompt_tokens": ptoks,
        "completion_tokens": ctoks,
        "completion_tokens_source": ctok_src,
        "committed_output_tokens": ctoks,
        "proposed_draft_tokens": proposed_draft_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "rejected_draft_tokens": rejected_draft_tokens,
        "cached_tokens": cached,
        "cached_tokens_source": cached_src,
        # Additive llama-server fields used by scenario reports.  Keep the
        # source-labelled cached_tokens/prompt_tokens fields above intact.
        "cache_n": int(cache_n) if cache_n is not None else None,
        "prompt_n": int(prompt_n) if prompt_n is not None else None,
        "prefix_hit": hit,
        "prefill_tps_client": prefill_client,
        "prefill_tps_server": _num(t.get("prompt_per_second")),
        "decode_tps_client": decode_client,
        "decode_tps_server": _num(t.get("predicted_per_second")),
        "draft_n": proposed_draft_tokens,
        "draft_accepted": accepted_draft_tokens,
        "draft_acceptance": acceptance,
        "timings": trace.timings,
        "usage": trace.usage,
        "sonder_warnings": trace.extra_final.get("sonder_warnings"),
    }


def _committed_tokens(r: dict[str, Any]) -> int | None:
    val = r.get("committed_output_tokens")
    if val is None:
        val = r.get("completion_tokens")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return int(val)
    return None


def _proposed_tokens(r: dict[str, Any]) -> int | None:
    val = r.get("proposed_draft_tokens")
    if val is None:
        val = r.get("draft_n")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return int(val)
    return None


def _accepted_tokens(r: dict[str, Any]) -> int | None:
    val = r.get("accepted_draft_tokens")
    if val is None:
        val = r.get("draft_accepted")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return int(val)
    return None


def group_summary(reqs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of derived request metrics."""
    ok = [r for r in reqs if r.get("ok")]
    itl: list[float] = []
    for r in ok:
        itl.extend(r.get("itl_s") or [])
    hits = [r["prefix_hit"] for r in ok if r.get("prefix_hit") is not None]
    recall = [r["recall_correct"] for r in ok if r.get("recall_correct") is not None]

    committed_vals = [v for v in (_committed_tokens(r) for r in ok) if v is not None]
    proposed_vals = [v for v in (_proposed_tokens(r) for r in ok) if v is not None]
    accepted_vals = [v for v in (_accepted_tokens(r) for r in ok) if v is not None]

    return {
        "requests": len(reqs),
        "ok": len(ok),
        "errors": [r.get("error") for r in reqs if not r.get("ok")][:5],
        "ttft_s": summarize(r.get("ttft_s") for r in ok),
        "itl_s": summarize(itl),
        "e2e_s": summarize(r.get("e2e_s") for r in ok),
        "client_observed_ms": {
            "ttft_ms": summarize(r["ttft_s"] * 1000.0 for r in ok if r.get("ttft_s") is not None),
            "itl_ms": summarize(x * 1000.0 for x in itl),
            "e2e_ms": summarize(r["e2e_s"] * 1000.0 for r in ok if r.get("e2e_s") is not None),
        },
        "backend_compute_ms": {
            "total_ms": summarize(r.get("backend_compute_ms") for r in ok),
            "prefill_ms": summarize(r.get("backend_prefill_ms") for r in ok),
            "decode_ms": summarize(r.get("backend_decode_ms") for r in ok),
        },
        "prompt_tokens": summarize(r.get("prompt_tokens") for r in ok),
        "completion_tokens_total": sum(r.get("completion_tokens") or 0 for r in ok),
        "committed_output_tokens_total": sum(committed_vals),
        "committed_output_tokens": summarize(committed_vals),
        # None when no request reported proposed/accepted draft counts: unknown, not zero.
        "proposed_draft_tokens_total": sum(proposed_vals) if proposed_vals else None,
        "proposed_draft_tokens": summarize(proposed_vals),
        "accepted_draft_tokens_total": sum(accepted_vals) if accepted_vals else None,
        "accepted_draft_tokens": summarize(accepted_vals),
        "prefill_tps_client": summarize(r.get("prefill_tps_client") for r in ok),
        "prefill_tps_server": summarize(r.get("prefill_tps_server") for r in ok),
        "decode_tps_client": summarize(r.get("decode_tps_client") for r in ok),
        "decode_tps_server": summarize(r.get("decode_tps_server") for r in ok),
        # None when no request reported cached tokens: unknown, not zero.
        "prefix_hit": summarize(hits) if hits else None,
        "prefix_hit_known": len(hits),
        "recall": (sum(1 for x in recall if x) / len(recall)) if recall else None,
        "recall_n": len(recall),
        "draft_acceptance": summarize(r.get("draft_acceptance") for r in ok if r.get("draft_acceptance") is not None),
    }


def split_cold_warm_summary(reqs: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently summarize cold, warm, and unclassified requests."""
    buckets: dict[str, list[dict[str, Any]]] = {"cold": [], "warm": [], "unclassified": []}
    for r in reqs:
        buckets[classify_cache_state(r)].append(r)
    return {name: group_summary(rows) for name, rows in buckets.items()}


def summarize_by_corpus_tag(reqs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize requests grouped by their ``corpus_tag`` (splitting cold vs warm)."""
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for r in reqs:
        tag = r.get("corpus_tag")
        if isinstance(tag, str) and tag:
            by_tag.setdefault(tag, []).append(r)
    return {
        tag: {
            "summary": group_summary(rows),
            "cold_vs_warm": split_cold_warm_summary(rows),
        }
        for tag, rows in sorted(by_tag.items())
    }
