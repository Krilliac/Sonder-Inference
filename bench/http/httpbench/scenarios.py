"""Pure helpers for the multi-agent HTTP benchmark scenarios."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .client import validate_headers


def scenario_headers(config: dict[str, Any], defaults: dict[str, str] | None = None) -> dict[str, str]:
    """Return validated per-request headers, with scenario values winning."""
    out = validate_headers(defaults)
    for key, value in validate_headers(config.get("headers")).items():
        out = {k: v for k, v in out.items() if k.lower() != key.lower()}
        out[key] = value
    return out


def pinned_thinking(warnings: Any) -> bool:
    """Recognise the additive warning shape emitted by Sonder health/requests."""
    if isinstance(warnings, dict):
        warnings = warnings.get("sonder", warnings)
        warnings = warnings.get("warnings", []) if isinstance(warnings, dict) else warnings
    if isinstance(warnings, str):
        warnings = [warnings]
    if not isinstance(warnings, (list, tuple)):
        return False
    text = " ".join(str(x).lower() for x in warnings)
    if not ("thinking" in text or "reasoning_effort" in text) or not ("pin" in text or "forced" in text):
        return False
    # A server pin to ``off`` must not inflate the output budget; ``on`` and
    # effort pins reserve tokens for hidden reasoning.
    if re.search(r"pin(?:ned)?\s*\(?\s*(?:off|none)\s*\)?", text):
        return False
    return True


def max_tokens_for(config: dict[str, Any], defaults: dict[str, Any], warnings: Any) -> int:
    """Raise recall/agent defaults when thinking is pinned, preserving overrides."""
    explicit = bool(config.get("_max_tokens_explicit") or config.get("max_tokens_explicit"))
    value = int(config.get("max_tokens", defaults.get("max_tokens", 64)))
    kind = config.get("kind")
    if not explicit and kind in {"agent_session", "agent_alternate", "long_context", "repeat_prompt"} \
            and pinned_thinking(warnings):
        return max(value, 1500)
    return value


def alternate_reuse(requests: Iterable[dict[str, Any]], minimum: float = 0.9) -> dict[str, Any]:
    """Assess A's turn 2+ cache_n against its measured prompt length."""
    a = [r for r in requests if r.get("agent") == "A" and int(r.get("turn", 0)) >= 2]
    ratios = []
    unknown = []
    for r in a:
        prompt = r.get("prompt_tokens")
        cached = r.get("cached_tokens")
        if r.get("ok", True) and isinstance(prompt, (int, float)) and prompt > 0 and isinstance(cached, (int, float)):
            ratios.append(float(cached) / float(prompt))
        else:
            unknown.append(r.get("turn"))
    return {"pass": bool(ratios) and not unknown and min(ratios) >= minimum,
            "ratios": ratios, "unknown_turns": unknown, "turns": len(a), "minimum": minimum}


def stall_failures(requests: Iterable[dict[str, Any]], wall_s: float, limit_s: float = 250.0) -> list[str]:
    failures = []
    rows = list(requests)
    if any(r.get("status") == 503 or "HTTP 503" in str(r.get("error")) for r in rows):
        failures.append("one or more requests returned HTTP 503")
    markers = {r.get("marker") for r in rows if r.get("marker")}
    for r in rows:
        if (r.get("e2e_s") or 0) > limit_s:
            failures.append(f"request wall {r['e2e_s']:.3f}s exceeds {limit_s:g}s")
        if r.get("ok") is False:
            failures.append(f"request {r.get('group')} failed: {r.get('error')}")
        content = str(r.get("_content", "")) + "\n" + str(r.get("_reasoning", ""))
        leaked = [m for m in markers if m != r.get("marker") and m in content]
        if leaked:
            failures.append(f"marker leak in {r.get('key') or r.get('group')}: {leaked[0]}")
    return failures
