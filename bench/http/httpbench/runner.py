"""Scenario execution, server identity and non-vacuity checks."""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from . import RESULTS_SCHEMA, SCENARIOS_SCHEMA, __version__
from .client import Target, trace_text, validate_headers
from .gpumem import GpuSampler, spill_check
from .metrics import (classify_cache_state, derive, group_summary, split_cold_warm_summary,
                      summarize, summarize_by_corpus_tag)
from .observation import (ChildLogProbe, MetricsProbe, build_provenance, metrics_delta,
                          summarize_validation_outcome, validate_provenance)
from .scenarios import (CORPUS_TAGS, alternate_reuse, max_tokens_for, overlap_count,
                        pinned_thinking, scenario_headers, stall_failures, trailing_summary,
                        validate_corpus_tag, warm_tool_return_reuse)
from .textgen import (Sizer, agent_system_prompt, corpus_prompt, plant, planted_facts,
                      record_list, tool_result_block, tool_schemas)

ACCEPTANCE_MATRIX_SCENARIOS: dict[str, Any] = {
    "schema": SCENARIOS_SCHEMA,
    "defaults": {"max_tokens": 48, "temperature": 0, "seed": 42},
    "scenarios": [
        {"name": "single-request", "kind": "single_request",
         "corpus_tags": ["prose", "code", "tool_json", "reasoning"],
         "prompt_tokens": 512, "warm_repeat": True, "max_tokens": 48},
        {"name": "simultaneous-siblings", "kind": "simultaneous_siblings", "siblings": 3,
         "shared_prefix_tokens": 2000, "corpus_tag": "code", "max_tokens": 48},
        {"name": "staggered-arrivals", "kind": "staggered_arrivals", "arrivals": 3,
         "stagger_ms": 25.0, "prompt_tokens": 1000, "corpus_tag": "prose", "max_tokens": 48},
        {"name": "long-trailing", "kind": "long_trailing", "short_peers": 2,
         "short_max_tokens": 16, "long_max_tokens": 96, "shared_prefix_tokens": 1000,
         "corpus_tag": "code"},
        {"name": "warm-tool-return", "kind": "warm_tool_return", "turns": 4,
         "system_tokens": 4000, "tools": 12, "turn_tokens": 300,
         "corpus_tag": "tool_json", "max_tokens": 48},
    ],
}

BUILTIN_SCENARIOS: dict[str, Any] = {
    "schema": SCENARIOS_SCHEMA,
    "defaults": {"max_tokens": 64, "temperature": 0, "seed": 42},
    "scenarios": [
        {"name": "agent-stable", "kind": "agent_session", "turns": 6, "system_tokens": 6000,
         "tools": 16, "turn_tokens": 400, "volatile_top": False, "max_tokens": 48},
        {"name": "agent-volatile-top", "kind": "agent_session", "turns": 6, "system_tokens": 6000,
         "tools": 16, "turn_tokens": 400, "volatile_top": True, "max_tokens": 48},
        {"name": "fanout-sweep", "kind": "concurrency_sweep", "levels": [1, 2, 4, 8], "waves": 2,
         "shared_prefix_tokens": 2000, "max_tokens": 64},
        {"name": "long-context", "kind": "long_context", "sizes": [32768, 65536, 100000, 130000],
         "depths": [0.1, 0.5, 0.9], "max_tokens": 64, "api": "chat"},
        {"name": "repeat-prompt", "kind": "repeat_prompt", "prompt_tokens": 8000, "repeats": 3, "max_tokens": 32},
        {"name": "agent-alternate", "kind": "agent_alternate", "turns": 3,
         "agent_tokens": {"A": 22000, "B": 5000, "C": 30000}, "max_tokens": 48},
        {"name": "priority-contention", "kind": "priority_contention", "background_tokens": 20000,
         "interactive_tokens": 500, "max_tokens": 48},
        {"name": "concurrency-stall", "kind": "concurrency_stall", "clients": 3, "max_tokens": 48},
        *copy.deepcopy(ACCEPTANCE_MATRIX_SCENARIOS["scenarios"]),
    ],
}

KINDS = ("agent_session", "concurrency_sweep", "long_context", "repeat_prompt", "agent_alternate",
         "priority_contention", "concurrency_stall", "single_request", "simultaneous_siblings",
         "staggered_arrivals", "long_trailing", "warm_tool_return")


def validate_config(cfg: dict) -> list[str]:
    errs = []
    if not isinstance(cfg, dict) or not isinstance(cfg.get("scenarios"), list):
        return ["config must be an object with a 'scenarios' list"]
    names = set()
    for i, s in enumerate(cfg["scenarios"]):
        if not isinstance(s, dict):
            errs.append(f"scenarios[{i}] is not an object")
            continue
        if s.get("kind") not in KINDS:
            errs.append(f"scenarios[{i}].kind must be one of {KINDS}, got {s.get('kind')!r}")
        n = s.get("name") or s.get("kind")
        if n in names:
            errs.append(f"duplicate scenario name {n!r}")
        names.add(n)
        if s.get("kind") == "concurrency_sweep" and not all(isinstance(x, int) and x > 0 for x in s.get("levels", [1])):
            errs.append(f"scenarios[{i}].levels must be positive integers")
        if s.get("kind") == "long_context":
            if not all(isinstance(x, int) and x > 0 for x in s.get("sizes", [])):
                errs.append(f"scenarios[{i}].sizes must be positive integers")
            if not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in s.get("depths", [0.5])):
                errs.append(f"scenarios[{i}].depths must be in [0, 1]")
        if s.get("headers") is not None and not isinstance(s.get("headers"), dict):
            errs.append(f"scenarios[{i}].headers must be an object")
        elif isinstance(s.get("headers"), dict):
            try:
                validate_headers(s["headers"])
            except (TypeError, ValueError) as e:
                errs.append(f"scenarios[{i}].headers invalid: {e}")
        if s.get("priority") is not None and s.get("priority") not in ("interactive", "subagent", "background"):
            errs.append(f"scenarios[{i}].priority must be interactive, subagent, or background")
        if s.get("prompt_cache_key") is not None and not isinstance(s.get("prompt_cache_key"), str):
            errs.append(f"scenarios[{i}].prompt_cache_key must be a string")
        if s.get("corpus_tag") is not None and not validate_corpus_tag(s.get("corpus_tag")):
            errs.append(f"scenarios[{i}].corpus_tag must be one of {CORPUS_TAGS}, got {s.get('corpus_tag')!r}")
        if s.get("corpus_tags") is not None:
            tags = s.get("corpus_tags")
            if not isinstance(tags, list) or not tags or not all(validate_corpus_tag(t) for t in tags):
                errs.append(f"scenarios[{i}].corpus_tags must be a non-empty list of {CORPUS_TAGS}")
        if s.get("kind") == "agent_alternate":
            tokens = s.get("agent_tokens", {"A": 22000, "B": 5000, "C": 30000})
            if (not isinstance(tokens, dict) or set(tokens) != {"A", "B", "C"}
                    or not all(type(v) is int and v > 0 for v in tokens.values())):
                errs.append(f"scenarios[{i}].agent_tokens must contain positive A, B, and C values")
            if type(s.get("turns", 3)) is not int or s.get("turns", 3) <= 0:
                errs.append(f"scenarios[{i}].turns must be a positive integer")
        if s.get("kind") == "concurrency_stall" and s.get("clients", 3) != 3:
            errs.append(f"scenarios[{i}].clients must be exactly 3")
        if s.get("kind") == "simultaneous_siblings":
            if type(s.get("siblings", 3)) is not int or s.get("siblings", 3) < 2:
                errs.append(f"scenarios[{i}].siblings must be an integer >= 2")
        if s.get("kind") == "staggered_arrivals":
            if type(s.get("arrivals", 3)) is not int or s.get("arrivals", 3) < 2:
                errs.append(f"scenarios[{i}].arrivals must be an integer >= 2")
            stagger = s.get("stagger_ms", 25.0)
            if type(stagger) not in (int, float) or not math.isfinite(stagger) or stagger < 0:
                errs.append(f"scenarios[{i}].stagger_ms must be finite and >= 0")
        if s.get("kind") == "long_trailing":
            if type(s.get("short_peers", 2)) is not int or s.get("short_peers", 2) < 1:
                errs.append(f"scenarios[{i}].short_peers must be a positive integer")
            for field in ("short_max_tokens", "long_max_tokens"):
                if field in s and (type(s[field]) is not int or s[field] <= 0):
                    errs.append(f"scenarios[{i}].{field} must be a positive integer")
        if s.get("kind") == "warm_tool_return":
            if type(s.get("turns", 4)) is not int or s.get("turns", 4) < 2:
                errs.append(f"scenarios[{i}].turns must be an integer >= 2")
        if "warm_repeat" in s and not isinstance(s["warm_repeat"], bool):
            errs.append(f"scenarios[{i}].warm_repeat must be a boolean")
        for field in ("prefix_tokens", "background_tokens", "interactive_tokens",
                      "prompt_tokens", "shared_prefix_tokens"):
            if field in s and (type(s[field]) is not int or s[field] <= 0):
                errs.append(f"scenarios[{i}].{field} must be a positive integer")
        for field in ("wall_limit_s", "interactive_start_timeout_s", "timeout", "wall_timeout"):
            if field in s and (type(s[field]) not in (int, float) or not math.isfinite(s[field]) or s[field] <= 0):
                errs.append(f"scenarios[{i}].{field} must be finite and positive")
    return errs


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Runner:
    def __init__(self, target: Target, *, model: str | None = None, defaults: dict | None = None,
                 extra_body: dict | None = None, sampler: GpuSampler | None = None,
                 ctx_size: int | None = None, shared_baseline_mib: float | None = None,
                 log: Callable[[str], None] | None = None,
                 metrics_url: str | None = None, child_log: str | None = None,
                 provenance: dict[str, Any] | None = None, require_provenance: bool = False):
        self.t = target
        self.model = model
        self.defaults = dict(defaults or {})
        self.extra_body = dict(extra_body or {})
        self.sampler = sampler or GpuSampler()
        self.ctx_size = ctx_size
        self.ctx_cli = ctx_size is not None
        self.shared_baseline_mib = shared_baseline_mib
        self.baseline_source = "cli" if shared_baseline_mib is not None else None
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.sizer: Sizer | None = None
        self._lock = threading.Lock()
        self.server_warnings: Any = None
        self.metrics_probe = MetricsProbe(metrics_url) if metrics_url else None
        self.child_log_probe = ChildLogProbe(child_log) if child_log else None
        self.provenance_overrides = dict(provenance or {})
        self.require_provenance = bool(require_provenance)
        self.run_id = uuid.uuid4().hex

    def _observe_before(self) -> dict:
        return {"metrics": self.metrics_probe.snapshot() if self.metrics_probe else None,
                "child_log": self.child_log_probe.snapshot() if self.child_log_probe else None}

    def _observe_after(self, before: dict, result: dict) -> None:
        if self.metrics_probe:
            after = self.metrics_probe.snapshot()
            result["metrics"] = metrics_delta(before["metrics"], after)
            result["metrics"].update({"before": before["metrics"], "after": after,
                                      "scope": "server-wide interval"})
        if self.child_log_probe:
            result["child_log"] = self.child_log_probe.delta(before["child_log"])

    # ------------------------------------------------------------------ identity

    def identify(self) -> dict[str, Any]:
        ident: dict[str, Any] = {}
        probes = [("models", self.t.oai("/models")), ("props", self.t.root("/props")),
                  ("health", self.t.root("/health")), ("sonder_health", self.t.root("/v1/sonder/health")),
                  ("sonder_identity", self.t.root("/v1/sonder/identity"))]
        for name, path in probes:
            status, body = self.t.request_json("GET", path)
            entry: dict[str, Any] = {"path": path, "status": status}
            if status == 200:
                if name == "props" and isinstance(body, dict):
                    body = _trim_props(body)
                entry["body"] = body
            elif status is None:
                entry["error"] = body
            ident[name] = entry
        models = (ident["models"].get("body") or {}).get("data") if ident["models"].get("status") == 200 else None
        if not self.model and isinstance(models, list) and models and isinstance(models[0], dict):
            self.model = models[0].get("id")
        ident["model_ids"] = [m.get("id") for m in models] if isinstance(models, list) else []
        for source in ("sonder_identity", "sonder_health"):
            sid = ident[source].get("body")
            if isinstance(sid, dict):
                warnings = (sid.get("sonder") or {}).get("warnings") if isinstance(sid.get("sonder"), dict) else None
                if warnings:
                    self.server_warnings = warnings
        if not self.ctx_cli:
            self.ctx_size = _ctx_from_identity(ident)
            ident["ctx_source"] = "server" if self.ctx_size else None
        else:
            ident["ctx_source"] = "cli"
        ident["n_ctx"] = self.ctx_size
        props = ident["props"].get("body") if ident["props"].get("status") == 200 else None
        ident["build_info"] = props.get("build_info") if isinstance(props, dict) else None
        return ident

    # ------------------------------------------------------------------ requests

    def _params(self, sc: dict) -> dict:
        p = dict(self.defaults)
        for k in ("max_tokens", "temperature", "seed", "top_p", "top_k"):
            if k in sc:
                p[k] = sc[k]
        warnings = self.server_warnings
        if "max_tokens" in p or pinned_thinking(warnings):
            p["max_tokens"] = max_tokens_for(sc, p, warnings)
        return p

    def body(self, api: str, sc: dict, messages: list[dict] | None = None, prompt: str | None = None,
             tools: list[dict] | None = None) -> tuple[str, dict]:
        p = self._params(sc)
        extra = copy.deepcopy(self.extra_body)
        extra.update(copy.deepcopy(sc.get("extra_body") or {}))
        for key in ("prompt_cache_key", "priority"):
            if key in sc:
                extra[key] = sc[key]
        if api == "completion":
            body = {"prompt": prompt if prompt is not None else _flatten(messages or []), "stream": True,
                    "cache_prompt": True, "n_predict": p.get("max_tokens", 64)}
            for k in ("temperature", "seed", "top_p", "top_k"):
                if k in p:
                    body[k] = p[k]
            body.update(extra)
            return self.t.root("/completion"), body
        body = {"messages": messages if messages is not None else [{"role": "user", "content": prompt or ""}],
                "stream": True, "stream_options": {"include_usage": True}}
        if self.model:
            body["model"] = self.model
        for k in ("max_tokens", "temperature", "seed", "top_p", "top_k"):
            if k in p:
                body[k] = p[k]
        if tools:
            body["tools"] = tools
        body.update(extra)
        return self.t.oai("/chat/completions"), body

    def one(self, scenario: str, group: str, api: str, sc: dict, t_run0: float, **kw) -> dict[str, Any]:
        on_sent = kw.pop("on_sent", None)
        path, body = self.body(api, sc, **kw)
        before_metrics = self.metrics_probe.snapshot() if self.metrics_probe else None
        t_start = time.perf_counter() - t_run0
        headers = scenario_headers(sc)
        trace = self.t.stream(path, body, api=api, headers=headers, timeout=sc.get("timeout"),
                              wall_timeout=sc.get("wall_timeout"), on_sent=on_sent)
        if trace.extra_final.get("sonder_warnings"):
            self.server_warnings = trace.extra_final["sonder_warnings"]
        m = derive(trace)
        if self.metrics_probe:
            after_metrics = self.metrics_probe.snapshot()
            m["metrics"] = metrics_delta(before_metrics, after_metrics)
            m["metrics"].update({"before": before_metrics, "after": after_metrics,
                                 "scope": "server-wide request window; concurrent windows overlap"})
        content, reasoning = trace_text(trace)
        m.update({"scenario": scenario, "group": group, "api": api, "t_start_s": t_start,
                  "content_tail": content[-240:], "content_chars": len(content),
                  "reasoning_chars": len(reasoning), "model": trace.model,
                  "not_streamed": bool(trace.extra_final.get("not_streamed"))})
        m["_content"] = content
        m["_reasoning"] = reasoning
        m["header_names"] = sorted(headers)
        m["prompt_cache_key"] = body.get("prompt_cache_key")
        m["priority"] = body.get("priority")
        cap_name = "n_predict" if api == "completion" else "max_tokens"
        cap = body.get(cap_name, 64)
        m["max_tokens_sent"] = body.get(cap_name)
        m["http_attempts"] = 1
        explicit_body = {**self.extra_body, **(sc.get("extra_body") or {})}
        larger = max_tokens_for(sc, {"max_tokens": cap}, trace.extra_final.get("sonder_warnings"))
        if m["ok"] and not sc.get("_thinking_retry") and larger > cap and cap_name not in explicit_body:
            prompt_n = m.get("prompt_tokens")
            if self.ctx_size and (prompt_n is None or prompt_n + larger + 256 > self.ctx_size):
                m["thinking_retry_skipped"] = "increased output budget cannot be proved to fit server n_ctx"
            else:
                retry_sc = {**sc, "max_tokens": larger, "_thinking_retry": True}
                retried = self.one(scenario, group, api, retry_sc, t_run0, **kw)
                initial = {k: v for k, v in m.items() if k not in ("_content", "_reasoning")}
                retried["thinking_retry"] = {"reason": "server warning reports pinned thinking", "initial": initial}
                retried["http_attempts"] = 2
                retried["total_attempt_wall_s"] = (m.get("e2e_s") or 0) + (retried.get("e2e_s") or 0)
                return retried
        return m

    # ------------------------------------------------------------------ scenarios

    def run_scenario(self, sc: dict, t_run0: float) -> dict[str, Any]:
        kind = sc["kind"]
        name = sc.get("name") or kind
        public = dict(sc)
        if isinstance(public.get("headers"), dict):
            public["headers"] = {str(k): "<redacted>" for k in public["headers"]}
        res: dict[str, Any] = {"name": name, "kind": kind, "config": public, "started_at": _now_iso()}
        res["gpu_before"] = self.sampler.sample(f"{name}:before")
        self._maybe_set_baseline(res["gpu_before"])
        observation_before = self._observe_before()
        t0 = time.perf_counter()
        fn = {"agent_session": self._agent, "concurrency_sweep": self._sweep,
              "long_context": self._long, "repeat_prompt": self._repeat,
              "agent_alternate": self._alternate, "priority_contention": self._priority,
              "concurrency_stall": self._stall, "single_request": self._single_request,
              "simultaneous_siblings": self._simultaneous_siblings,
              "staggered_arrivals": self._staggered_arrivals,
              "long_trailing": self._long_trailing,
              "warm_tool_return": self._warm_tool_return}[kind]
        try:
            fn(sc, name, res, t_run0)
        except Exception as e:  # keep the partial scenario; the checks will fail it
            res["exception"] = f"{type(e).__name__}: {e}"
            self.log(f"[{name}] exception: {res['exception']}")
        res["wall_s"] = time.perf_counter() - t0
        self._observe_after(observation_before, res)
        res["gpu_after"] = self.sampler.sample(f"{name}:after")
        res["spill"] = spill_check(res["gpu_after"], self.shared_baseline_mib)
        reqs = res.setdefault("requests", [])
        scenario_tag = sc.get("corpus_tag")
        for r in reqs:
            r.pop("_content", None)
            r.pop("_reasoning", None)
            if scenario_tag and not r.get("corpus_tag"):
                r["corpus_tag"] = scenario_tag
            r["cache_state"] = classify_cache_state(r)
        res["summary"] = group_summary(reqs)
        res["cold_vs_warm"] = split_cold_warm_summary(reqs)
        by_tag = summarize_by_corpus_tag(reqs)
        if by_tag:
            res["by_corpus_tag"] = by_tag
        groups: dict[str, list] = {}
        for r in reqs:
            groups.setdefault(r["group"], []).append(r)
        res["groups"] = {g: group_summary(rs) for g, rs in groups.items()}
        for group, rows in groups.items():
            res["groups"][group].update({key: summarize(r.get(key) for r in rows if r.get("ok"))
                                         for key in ("cache_n", "prompt_n")})
            if len(rows) == 1 and "metrics" in rows[0]:
                res["groups"][group]["metrics"] = rows[0]["metrics"]
        for g, extra in (res.pop("_group_extra", {}) or {}).items():
            res["groups"].setdefault(g, {}).update(extra)
        return res

    def _maybe_set_baseline(self, sample: dict | None) -> None:
        if self.shared_baseline_mib is None and sample and not sample.get("error") and "shared_mib" in sample:
            self.shared_baseline_mib = sample["shared_mib"]
            self.baseline_source = f"first sample ({sample['label']})"

    def _agent(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        seed = int(sc.get("seed_text", 1234))
        turns = int(sc.get("turns", 6))
        tools_mode = sc.get("tools_mode", "system")
        system = agent_system_prompt(self.sizer, int(sc.get("system_tokens", 6000)),
                                     int(sc.get("tools", 16)) if tools_mode == "system" else 0, seed)
        tools = tool_schemas(int(sc.get("tools", 16))) if tools_mode == "native" else None
        rnd = random.Random(seed + 1)
        history: list[dict] = []
        api = sc.get("api", "chat")
        res["expected_requests"] = turns
        res["system_prompt_tokens"] = self.sizer.measure(system)
        for turn in range(1, turns + 1):
            sys_text = system
            if sc.get("volatile_top"):
                sys_text = f"[session clock {time.time_ns()} nonce {uuid.uuid4().hex}]\n" + system
            block = tool_result_block(rnd, turn, int(sc.get("turn_tokens", 400)), self.sizer.chars_per_token)
            history.append({"role": "user", "content": block + "\n\nWhat is the next step? Answer in one sentence."})
            msgs = [{"role": "system", "content": sys_text}] + history
            r = self.one(name, f"turn{turn}", api, sc, t_run0, messages=msgs, tools=tools)
            r["turn"] = turn
            r["cache_state"] = "cold" if (turn == 1 or sc.get("volatile_top")) else "warm"
            res.setdefault("requests", []).append(r)
            self.log(f"[{name}] turn {turn}: ok={r['ok']} ttft={_fmt(r['ttft_s'])} hit={_fmt(r['prefix_hit'])} "
                     f"prompt={r['prompt_tokens']} gen={r['completion_tokens']}" + (f" err={r['error']}" if r["error"] else ""))
            history.append(_assistant_history(r))

    def _sweep(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        levels = [int(x) for x in sc.get("levels", [1, 2, 4, 8])]
        waves = int(sc.get("waves", 2))
        api = sc.get("api", "chat")
        rnd = random.Random(int(sc.get("seed_text", 99)))
        prefix_lines = record_list(rnd, int(sc.get("shared_prefix_tokens", 2000)), self.sizer.chars_per_token)
        prefix = "Shared parent context for all sub-agents:\n" + "\n".join(
            self.sizer.fit(prefix_lines, int(sc.get("shared_prefix_tokens", 2000))))

        def msgs(i: int) -> list[dict]:
            return [{"role": "system", "content": prefix},
                    {"role": "user", "content": f"Sub-task {i}: summarize Record {i + 1} in one sentence."}]

        reqs = res.setdefault("requests", [])
        # Warm the shared prefix once, as the parent agent would have.
        w = self.one(name, "warmup", api, sc, t_run0, messages=msgs(0))
        w["cache_state"] = "cold"
        reqs.append(w)
        res["expected_requests"] = 1 + sum(level * waves for level in levels)
        extra: dict[str, dict] = {}
        counter = 1
        for level in levels:
            n = level * waves
            idx = list(range(counter, counter + n))
            counter += n
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=level) as ex:
                out = list(ex.map(lambda i: self.one(name, f"c{level}", api, sc, t_run0, messages=msgs(i)), idx))
            makespan = time.perf_counter() - t0
            for r in out:
                r["concurrency"] = level
                r["cache_state"] = "warm"
            reqs.extend(out)
            toks = sum(r["completion_tokens"] or 0 for r in out if r["ok"])
            extra[f"c{level}"] = {"concurrency": level, "makespan_s": makespan,
                                  "aggregate_decode_tps": toks / makespan if makespan > 0 else None,
                                  "requests_per_s": n / makespan if makespan > 0 else None}
            hits = [r["prefix_hit"] for r in out if r["prefix_hit"] is not None]
            self.log(f"[{name}] c={level}: {sum(r['ok'] for r in out)}/{n} ok, makespan {makespan:.2f}s, "
                     f"{extra[f'c{level}']['aggregate_decode_tps'] or 0:.1f} tok/s aggregate, "
                     f"hit {_fmt(sum(hits) / len(hits)) if hits else 'unknown'}")
        res["_group_extra"] = extra

    def _long(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        sizes = [int(x) for x in sc.get("sizes", [32768])]
        depths = [float(x) for x in sc.get("depths", [0.1, 0.5, 0.9])]
        api = sc.get("api", "chat")
        margin = int(sc.get("margin_tokens", 256))
        res["skipped"] = []
        expected = 0
        reqs = res.setdefault("requests", [])
        for size in sizes:
            cap_field = "n_predict" if api == "completion" else "max_tokens"
            max_tokens = int(self.body(api, sc, prompt="")[1].get(cap_field, 64))
            if self.ctx_size and size + max_tokens + margin > self.ctx_size:
                res["skipped"].append({"size": size, "reason": f"size + max_tokens + {margin} > server n_ctx {self.ctx_size}"})
                self.log(f"[{name}] skip {size}: exceeds server n_ctx {self.ctx_size}")
                continue
            rnd = random.Random(size)
            facts = planted_facts(rnd, len(depths))
            question_budget = 80
            lines = record_list(rnd, size, self.sizer.chars_per_token)
            header = f"Document set {size} ({uuid.UUID(int=rnd.getrandbits(128)).hex}). Read all records and notes.\n"
            body_target = size - question_budget - len(depths) * 30 - self.sizer.measure(header)
            lines = self.sizer.fit(lines, max(1, body_target))
            lines, actual = plant(lines, facts, depths)
            body = header + "\n".join(lines)
            expected += len(depths)
            for k, f in enumerate(facts):
                q = "\n\nQuestion: " + f["question"]
                group = f"{size}"
                if api == "completion":
                    r = self.one(name, group, api, sc, t_run0, prompt=body + q + "\nAnswer:")
                else:
                    r = self.one(name, group, api, sc, t_run0, messages=[{"role": "user", "content": body + q}])
                content = r.get("_content") or ""
                reasoning = r.get("_reasoning") or ""
                r.update({"size": size, "depth": depths[k], "depth_actual": actual[k], "expected": f["phrase"],
                          "recall_correct": f["phrase"].lower() in (content + "\n" + reasoning).lower() if r["ok"] else None,
                          "found_in_reasoning": f["phrase"].lower() in reasoning.lower(),
                          "cold": k == 0})
                reqs.append(r)
                self.log(f"[{name}] size {size} depth {depths[k]}: ok={r['ok']} prompt={r['prompt_tokens']} "
                         f"ttft={_fmt(r['ttft_s'])} recall={r['recall_correct']}" + (f" err={r['error']}" if r["error"] else ""))
        res["expected_requests"] = expected

    def _repeat(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        n = int(sc.get("repeats", 3))
        api = sc.get("api", "chat")
        rnd = random.Random(int(sc.get("seed_text", 4242)))
        target = int(sc.get("prompt_tokens", 8000))
        lines = self.sizer.fit(record_list(rnd, target, self.sizer.chars_per_token), target)
        prompt = f"Repeat-prompt probe {rnd.getrandbits(64):x}.\n" + "\n".join(lines) + \
            "\n\nQuestion: What is the code of Record 7? Answer briefly."
        res["expected_requests"] = n
        reqs = res.setdefault("requests", [])
        for i in range(1, n + 1):
            if api == "completion":
                r = self.one(name, f"rep{i}", api, sc, t_run0, prompt=prompt + "\nAnswer:")
            else:
                r = self.one(name, f"rep{i}", api, sc, t_run0, messages=[{"role": "user", "content": prompt}])
            r["repeat"] = i
            reqs.append(r)
            self.log(f"[{name}] repeat {i}: ok={r['ok']} ttft={_fmt(r['ttft_s'])} hit={_fmt(r['prefix_hit'])}")

    def _alternate(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        turns = int(sc.get("turns", 3))
        sizes = {k: int(v) for k, v in (sc.get("agent_tokens") or {"A": 22000, "B": 5000, "C": 30000}).items()}
        reqs = res.setdefault("requests", [])
        res["expected_requests"] = turns * len(sizes)
        prefixes = {}
        histories = {agent: [] for agent in ("A", "B", "C")}
        for agent in ("A", "B", "C"):
            rnd = random.Random(int(sc.get("seed_text", 7000)) + ord(agent))
            prefixes[agent] = "Agent " + agent + " stable context.\n" + "\n".join(
                self.sizer.fit(record_list(rnd, sizes[agent], self.sizer.chars_per_token), sizes[agent]))
        for turn in range(1, turns + 1):
            for agent in ("A", "B", "C"):
                histories[agent].append({"role": "user", "content": f"Agent {agent}, turn {turn}: continue."})
                base_key = sc.get("prompt_cache_key", f"{self.run_id}:{name}")
                sub = dict(sc, prompt_cache_key=f"{base_key}:agent-{agent}")
                r = self.one(name, f"{agent}-turn{turn}", sc.get("api", "chat"), sub, t_run0,
                             messages=[{"role": "system", "content": prefixes[agent]}] + histories[agent])
                r.update({"agent": agent, "turn": turn, "key": f"agent-{agent}"})
                reqs.append(r)
                histories[agent].append(_assistant_history(r))
        res["alternate_reuse"] = alternate_reuse(reqs)

    def _priority(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        target = int(sc.get("background_tokens", 20000))
        rnd = random.Random(int(sc.get("seed_text", 8800)))
        nonce = uuid.uuid4().hex
        res["cold_nonce"] = nonce
        bg_text = f"Cold background prefill {nonce}.\n" + "\n".join(
            self.sizer.fit(record_list(rnd, target, self.sizer.chars_per_token), target))
        base_key = str(sc.get("prompt_cache_key", "priority-contention"))
        bg = dict(sc, prompt_cache_key=f"{base_key}:background:{nonce}", priority="background")
        inter = dict(sc, prompt_cache_key=f"{base_key}:interactive:{nonce}", priority="interactive")
        for request_sc in (bg, inter):
            request_sc["headers"] = {k: v for k, v in (sc.get("headers") or {}).items()
                                     if k.lower() != "x-sonder-priority"}
            request_sc["headers"]["X-Sonder-Priority"] = request_sc["priority"]
        interactive_text = "Interactive canary.\n" + "\n".join(
            self.sizer.fit(record_list(random.Random(8811), int(sc.get("interactive_tokens", 500)),
                                       self.sizer.chars_per_token), int(sc.get("interactive_tokens", 500))))
        out: list[dict] = []
        sent = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as ex:
            future = ex.submit(self.one, name, "background", sc.get("api", "chat"), bg, t_run0,
                               messages=[{"role": "user", "content": bg_text}], on_sent=sent.set)
            if not sent.wait(float(sc.get("interactive_start_timeout_s", 5.0))):
                res["contention_failure"] = "background request was not sent before interactive request"
            ir = self.one(name, "interactive", sc.get("api", "chat"), inter, t_run0,
                          messages=[{"role": "user", "content": interactive_text}])
            out.extend([future.result(), ir])
        for r, group in zip(out, ("background", "interactive")):
            r["priority"] = group
            r["cache_state"] = "cold"
            r["cold"] = True
        res["requests"] = out
        res["expected_requests"] = 2
        res["interactive_ttft_s"] = out[1].get("ttft_s")
        bg_start = out[0].get("t_start_s")
        bg_end = bg_start + (out[0].get("e2e_s") or 0) if bg_start is not None else None
        res["overlapped"] = bool(sent.is_set() and bg_end is not None and
                                  bg_start <= out[1].get("t_start_s", 0) < bg_end)

    def _stall(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        clients = int(sc.get("clients", 3))
        if clients != 3:
            raise ValueError("concurrency_stall requires exactly 3 clients")
        barrier = threading.Barrier(clients)
        base_key = sc.get("prompt_cache_key", f"{self.run_id}:{name}")
        prefixes = []
        for i in range(clients):
            target = int(sc.get("prefix_tokens", 20000))
            rnd = random.Random(int(sc.get("seed_text", 9100)) + i)
            prefixes.append("Client prefix.\n" + "\n".join(
                self.sizer.fit(record_list(rnd, target, self.sizer.chars_per_token), target)))
        def run(i: int) -> dict:
            key = f"{base_key}:client-{i}"
            sub = dict(sc, prompt_cache_key=key, wall_timeout=min(float(sc.get("wall_limit_s", 250)), 250.0))
            barrier.wait(timeout=10)
            return self.one(name, f"client{i}", sc.get("api", "chat"), sub, t_run0,
                            messages=[{"role": "system", "content": prefixes[i]},
                                      {"role": "user", "content": f"marker=STALL_CANARY_{i}; return only your marker."}])
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=clients) as ex:
            out = list(ex.map(run, range(clients)))
        wall = time.perf_counter() - t0
        for i, r in enumerate(out):
            r.update({"key": f"stall-client-{i}", "marker": f"STALL_CANARY_{i}"})
            r["marker_present"] = r["marker"] in ((r.get("_content") or "") + "\n" + (r.get("_reasoning") or ""))
        # Keep content private until checks have inspected it.
        for r in out:
            r["cache_state"] = "cold"
        res["requests"] = out
        res["expected_requests"] = clients
        res["stall_failures"] = stall_failures(out, wall, min(float(sc.get("wall_limit_s", 250)), 250.0))
        res["stall_wall_s"] = wall

    def _single_request(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        tags = list(sc.get("corpus_tags") or CORPUS_TAGS)
        warm_repeat = bool(sc.get("warm_repeat", True))
        target_tokens = int(sc.get("prompt_tokens", 512))
        api = sc.get("api", "chat")
        seed = int(sc.get("seed_text", 3100))
        base_key = sc.get("prompt_cache_key", f"{self.run_id}:{name}")
        reqs = res.setdefault("requests", [])
        res["expected_requests"] = len(tags) * (2 if warm_repeat else 1)
        for idx, tag in enumerate(tags):
            prompt = corpus_prompt(tag, self.sizer, target_tokens, seed, variant=idx)
            sub = dict(sc, prompt_cache_key=f"{base_key}:{tag}")
            kw = {"prompt": prompt + "\nAnswer:"} if api == "completion" else {
                "messages": [{"role": "user", "content": prompt}]
            }
            r_cold = self.one(name, f"{tag}-cold", api, sub, t_run0, **kw)
            r_cold.update({"corpus_tag": tag, "cache_state": "cold", "cold": True, "repeat": 1})
            reqs.append(r_cold)
            if warm_repeat:
                r_warm = self.one(name, f"{tag}-warm", api, sub, t_run0, **kw)
                r_warm.update({"corpus_tag": tag, "cache_state": "warm", "cold": False, "repeat": 2})
                reqs.append(r_warm)

    def _simultaneous_siblings(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        siblings = int(sc.get("siblings", 3))
        api = sc.get("api", "chat")
        tag = str(sc.get("corpus_tag") or "code")
        shared_tokens = int(sc.get("shared_prefix_tokens", 2000))
        seed = int(sc.get("seed_text", 3200))
        rnd = random.Random(seed)
        prefix_lines = self.sizer.fit(record_list(rnd, shared_tokens, self.sizer.chars_per_token), shared_tokens)
        prefix = f"Shared sibling parent context ({self.run_id}:{name}).\n" + "\n".join(prefix_lines)
        base_key = sc.get("prompt_cache_key", f"{self.run_id}:{name}")
        reqs = res.setdefault("requests", [])
        res["expected_requests"] = 1 + siblings

        warm_sc = dict(sc, prompt_cache_key=f"{base_key}:warmup")
        w = self.one(name, "warmup", api, warm_sc, t_run0,
                     messages=[{"role": "system", "content": prefix},
                               {"role": "user", "content": "Warm shared prefix. Reply ready."}])
        w.update({"cache_state": "cold", "cold": True, "corpus_tag": tag})
        reqs.append(w)

        barrier = threading.Barrier(siblings)
        wall_limit = min(float(sc.get("wall_limit_s", 250)), 250.0)

        def run_sibling(i: int) -> dict:
            marker = f"SIBLING_CANARY_{i}"
            sub = dict(sc, prompt_cache_key=f"{base_key}:sibling-{i}", wall_timeout=wall_limit)
            task = f"marker={marker}; " + corpus_prompt(tag, self.sizer, 64, seed + 50, variant=i)
            barrier.wait(timeout=10)
            row = self.one(name, "siblings", api, sub, t_run0,
                           messages=[{"role": "system", "content": prefix}, {"role": "user", "content": task}])
            row.update({"sibling": i, "key": f"sibling-{i}", "marker": marker,
                        "cache_state": "warm", "cold": False, "corpus_tag": tag})
            return row

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=siblings) as ex:
            out = list(ex.map(run_sibling, range(siblings)))
        makespan = time.perf_counter() - t0
        reqs.extend(out)
        committed = sum(r.get("committed_output_tokens") or 0 for r in out if r.get("ok"))
        res["overlapped_siblings"] = overlap_count(out)
        res["sibling_failures"] = stall_failures(out, makespan, wall_limit)
        res["_group_extra"] = {
            "siblings": {
                "concurrency": siblings,
                "makespan_s": makespan,
                "aggregate_decode_tps": committed / makespan if makespan > 0 else None,
                "overlapped_pairs": res["overlapped_siblings"],
            }
        }

    def _staggered_arrivals(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        arrivals = int(sc.get("arrivals", 3))
        stagger_ms = float(sc.get("stagger_ms", 25.0))
        stagger_s = stagger_ms / 1000.0
        api = sc.get("api", "chat")
        tag = str(sc.get("corpus_tag") or "prose")
        target_tokens = int(sc.get("prompt_tokens", 1000))
        seed = int(sc.get("seed_text", 3300))
        base_key = sc.get("prompt_cache_key", f"{self.run_id}:{name}")
        sent_events = [threading.Event() for _ in range(arrivals)]

        def run_arrival(i: int) -> dict:
            if i > 0:
                sent_events[i - 1].wait(timeout=5.0)
                if stagger_s > 0:
                    time.sleep(stagger_s)
            prompt = corpus_prompt(tag, self.sizer, target_tokens, seed, variant=i)
            sub = dict(sc, prompt_cache_key=f"{base_key}:arrival-{i}")
            row = self.one(name, f"arrival{i}", api, sub, t_run0,
                           messages=[{"role": "user", "content": prompt}],
                           on_sent=sent_events[i].set)
            row.update({"arrival_index": i, "intended_stagger_ms": i * stagger_ms,
                        "cache_state": "cold", "cold": True, "corpus_tag": tag})
            return row

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=arrivals) as ex:
            out = list(ex.map(run_arrival, range(arrivals)))
        makespan = time.perf_counter() - t0
        res["requests"] = out
        res["expected_requests"] = arrivals
        res["stagger_ms"] = stagger_ms
        res["overlapped_pairs"] = overlap_count(out)
        res["makespan_s"] = makespan

    def _long_trailing(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        short_peers = int(sc.get("short_peers", 2))
        short_max = int(sc.get("short_max_tokens", 16))
        long_max = int(sc.get("long_max_tokens", 96))
        api = sc.get("api", "chat")
        tag = str(sc.get("corpus_tag") or "code")
        shared_tokens = int(sc.get("shared_prefix_tokens", 1000))
        seed = int(sc.get("seed_text", 3400))
        rnd = random.Random(seed)
        prefix_lines = self.sizer.fit(record_list(rnd, shared_tokens, self.sizer.chars_per_token), shared_tokens)
        prefix = f"Trailing batch context ({self.run_id}:{name}).\n" + "\n".join(prefix_lines)
        base_key = sc.get("prompt_cache_key", f"{self.run_id}:{name}")
        total = short_peers + 1
        barrier = threading.Barrier(total)

        def run_worker(idx: int) -> dict:
            is_trailing = idx == short_peers
            group = "trailing" if is_trailing else "short-peer"
            cap = long_max if is_trailing else short_max
            sub = dict(sc, prompt_cache_key=f"{base_key}:{group}-{idx}",
                       max_tokens=cap, _max_tokens_explicit=True)
            task = corpus_prompt(tag, self.sizer, 96 if is_trailing else 48, seed + 20, variant=idx)
            barrier.wait(timeout=10)
            row = self.one(name, group, api, sub, t_run0,
                           messages=[{"role": "system", "content": prefix}, {"role": "user", "content": task}])
            row.update({"role": "trailing" if is_trailing else "short_peer",
                        "cache_state": "cold", "cold": True, "corpus_tag": tag})
            return row

        with ThreadPoolExecutor(max_workers=total) as ex:
            out = list(ex.map(run_worker, range(total)))
        peers = [r for r in out if r.get("role") == "short_peer"]
        trailing_row = next((r for r in out if r.get("role") == "trailing"), None)
        res["requests"] = out
        res["expected_requests"] = total
        res["trailing"] = trailing_summary(peers, trailing_row)
        res["overlapped_pairs"] = overlap_count(out)

    def _warm_tool_return(self, sc: dict, name: str, res: dict, t_run0: float) -> None:
        turns = int(sc.get("turns", 4))
        api = sc.get("api", "chat")
        tag = str(sc.get("corpus_tag") or "tool_json")
        seed = int(sc.get("seed_text", 3500))
        tools_mode = sc.get("tools_mode", "system")
        system = agent_system_prompt(self.sizer, int(sc.get("system_tokens", 4000)),
                                     int(sc.get("tools", 12)) if tools_mode == "system" else 0, seed)
        tools = tool_schemas(int(sc.get("tools", 12))) if tools_mode == "native" else None
        rnd = random.Random(seed + 1)
        history: list[dict] = []
        base_key = sc.get("prompt_cache_key", f"{self.run_id}:{name}")
        sub = dict(sc, prompt_cache_key=base_key)
        res["expected_requests"] = turns
        res["system_prompt_tokens"] = self.sizer.measure(system)
        reqs = res.setdefault("requests", [])
        for turn in range(1, turns + 1):
            if turn == 1:
                user_msg = (
                    "Inspect the workspace ledger and emit the first tool call as strict JSON "
                    '{"tool": "<name>", "arguments": {"path": "<path>"}}.'
                )
                group = "cold-turn1"
                state = "cold"
            else:
                block = tool_result_block(rnd, turn, int(sc.get("turn_tokens", 300)), self.sizer.chars_per_token)
                user_msg = (
                    block
                    + '\n\nTool return received. Emit the next JSON tool call or final status object.'
                )
                group = f"warm-return{turn}"
                state = "warm"
            history.append({"role": "user", "content": user_msg})
            msgs = [{"role": "system", "content": system}] + history
            r = self.one(name, group, api, sub, t_run0, messages=msgs, tools=tools)
            r.update({"turn": turn, "cache_state": state, "cold": turn == 1, "corpus_tag": tag})
            reqs.append(r)
            history.append(_assistant_history(r))
        res["warm_tool_reuse"] = warm_tool_return_reuse(reqs)

    # ------------------------------------------------------------------ run

    def run(self, cfg: dict, label: str = "", argv: list[str] | None = None) -> dict[str, Any]:
        started = _now_iso()
        self.run_id = uuid.uuid4().hex
        ident = self.identify()
        self.log(f"target {self.t.display} model={self.model!r} n_ctx={self.ctx_size} build={ident.get('build_info')!r}")
        self.sizer = Sizer(self.t.tokenize)
        self.log(f"prompt sizing: {self.sizer.method}")
        self.defaults = {**(cfg.get("defaults") or {}), **self.defaults}
        base = self.sampler.sample("baseline")
        self._maybe_set_baseline(base)
        t_run0 = time.perf_counter()
        observation_before = self._observe_before()
        scenarios = []
        for sc in cfg["scenarios"]:
            self.log(f"== scenario {sc.get('name') or sc['kind']} ({sc['kind']})")
            scenarios.append(self.run_scenario(sc, t_run0))
        all_reqs = [r for s in scenarios for r in (s.get("requests") or [])]
        results: dict[str, Any] = {
            "schema": RESULTS_SCHEMA,
            "label": label,
            "started_at": started,
            "finished_at": _now_iso(),
            "wall_s": time.perf_counter() - t_run0,
            "harness": {"name": "bench_http", "version": __version__, "python": platform.python_version(),
                        "platform": platform.platform(), "argv": argv or []},
            "target": {"base_url": self.t.display, "model": self.model, "extra_body": self.extra_body,
                       "defaults": self.defaults},
            "server": ident,
            "prompt_sizing": self.sizer.method,
            "gpu": {"available": self.sampler.available, "reason": self.sampler.reason,
                    "pid": self.sampler.pid, "process_name": self.sampler.process_name,
                    "shared_baseline_mib": self.shared_baseline_mib, "baseline_source": self.baseline_source,
                    "samples": self.sampler.samples},
            "cold_vs_warm": split_cold_warm_summary(all_reqs),
            "scenarios": scenarios,
        }
        by_corpus = summarize_by_corpus_tag(all_reqs)
        if by_corpus:
            results["by_corpus_tag"] = by_corpus
        self._observe_after(observation_before, results)
        results["provenance"] = build_provenance(results, cfg, overrides=self.provenance_overrides)
        checks = run_checks(results, require_provenance=self.require_provenance)
        results["checks"] = checks
        results["verdict"] = verdict(checks)
        results["provenance"]["validation_outcome"] = summarize_validation_outcome(
            checks, results["verdict"], results["provenance"])
        return results


def _assistant_history(row: dict) -> dict:
    content = row.get("_content") or ""
    reasoning = row.get("_reasoning") or ""
    history = {"role": "assistant", "content": content.strip() if reasoning else content.strip() or "(no answer)"}
    if reasoning:
        history["reasoning_content"] = reasoning
    return history


def _trim_props(p: dict) -> dict:
    out = {}
    for k in ("build_info", "model_path", "total_slots", "n_ctx", "modalities", "is_sleeping"):
        if k in p:
            out[k] = p[k]
    dgs = p.get("default_generation_settings")
    if isinstance(dgs, dict):
        out["default_generation_settings"] = {k: dgs[k] for k in ("n_ctx", "id_slot", "speculative") if k in dgs}
        params = dgs.get("params")
        if isinstance(params, dict):
            out["default_generation_settings"]["params"] = {
                k: params[k] for k in ("speculative.n_max", "speculative.n_min", "cache_prompt", "n_keep") if k in params}
    if isinstance(p.get("chat_template"), str):
        out["chat_template_chars"] = len(p["chat_template"])
    return out


def _ctx_from_identity(ident: dict) -> int | None:
    props = ident.get("props", {}).get("body")
    if isinstance(props, dict):
        dgs = props.get("default_generation_settings") or {}
        for v in (dgs.get("n_ctx"), props.get("n_ctx")):
            if isinstance(v, int) and v > 0:
                return v
    sid = ident.get("sonder_identity", {}).get("body")
    if isinstance(sid, dict) and isinstance(sid.get("backend_identity"), dict):
        v = sid["backend_identity"].get("context_tokens")
        if isinstance(v, int) and v > 0:
            return v
    return None


def _flatten(messages: list[dict]) -> str:
    parts = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in messages]
    return "\n\n".join(parts) + "\n\nAssistant:"


def _fmt(x: Any) -> str:
    if x is None:
        return "?"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


# ---------------------------------------------------------------------- checks

def _check(name: str, status: str, detail: str, scenario: str | None = None) -> dict:
    return {"check": name, "status": status, "detail": detail, "scenario": scenario}


def run_checks(results: dict, *, require_provenance: bool = False) -> list[dict]:
    """Non-vacuity assertions. A check that cannot observe its signal says so; it never passes silently."""
    checks: list[dict] = []
    scen = results.get("scenarios") or []
    # No observation is mandatory unless the operator enabled it. Inspect
    # every request too: a failed intermediate scrape must not disappear in
    # an otherwise successful run-boundary delta.
    sources = [results]
    for scenario in scen:
        sources.append(scenario)
        for row in scenario.get("requests") or []:
            sources.append(row)
            initial = (row.get("thinking_retry") or {}).get("initial")
            if initial:
                sources.append(initial)
    for key, check_name in (("metrics", "metrics_available"), ("child_log", "child_log_available")):
        observations = [source[key] for source in sources if key in source]
        if observations:
            problems = []
            for observation in observations:
                problems.extend(str(e) for e in observation.get("errors") or [])
                for flag in ("resets", "unknown", "reset", "partial"):
                    if observation.get(flag):
                        problems.append(f"{flag}: {observation[flag]}")
            detail = "; ".join(dict.fromkeys(problems)) if problems else "all enabled observation windows were readable"
            checks.append(_check(check_name, "warn" if problems else "pass", detail[:1000]))
    if any(source.get("thinking_retry_skipped") for source in sources):
        checks.append(_check("thinking_headroom", "warn", "pinned thinking needs 1500 output tokens but context fit is unproven"))
    if not scen:
        checks.append(_check("scenarios_ran", "fail", "no scenarios ran"))
    total_ok = 0
    for s in scen:
        name = s["name"]
        reqs = s.get("requests") or []
        ok = sum(1 for r in reqs if r.get("ok"))
        total_ok += ok
        exp = s.get("expected_requests")
        if s.get("exception"):
            checks.append(_check("scenario_exception", "fail", s["exception"], name))
        if exp is None:
            checks.append(_check("request_count", "fail", "expected request count unknown", name))
        elif ok != exp:
            errs = "; ".join(str(e) for e in (s["summary"].get("errors") or [])[:2])
            checks.append(_check("request_count", "fail", f"{ok}/{exp} requests completed. {errs}".strip(), name))
        elif exp == 0:
            detail = "0 requests expected"
            if s.get("skipped"):
                detail += f" (all {len(s['skipped'])} sizes skipped: {s['skipped'][0]['reason']})"
            checks.append(_check("request_count", "warn", detail, name))
        else:
            checks.append(_check("request_count", "pass", f"{ok}/{exp} requests completed", name))
        gen = sum(r.get("completion_tokens") or 0 for r in reqs if r.get("ok"))
        if exp:
            checks.append(_check("tokens_generated", "pass" if gen > 0 else "fail",
                                 f"{gen} completion tokens", name))
        if any(r.get("not_streamed") for r in reqs):
            checks.append(_check("streamed", "warn", "server answered without SSE; TTFT/ITL are not token timings", name))
        if s["kind"] == "agent_session" and not s["config"].get("volatile_top"):
            later = [r for r in reqs if r.get("ok") and (r.get("turn") or 0) >= 2]
            hits = [r.get("prefix_hit") for r in later]
            if not later:
                checks.append(_check("prefix_reuse", "fail", "no successful turn 2+", name))
            elif all(h is None for h in hits):
                checks.append(_check("prefix_reuse", "warn",
                                     "server reports no cached-token count (timings.cache_n / "
                                     "usage.prompt_tokens_details.cached_tokens): prefix hit unknown", name))
            elif any(h == 0 for h in hits):
                zero = [r["turn"] for r in later if r.get("prefix_hit") == 0]
                checks.append(_check("prefix_reuse", "warn",
                                     f"prefix hit is 0 on turn(s) {zero}: the prompt cache is not reusing the stable "
                                     "prefix (a cache that stopped working looks like a slow model)", name))
            else:
                known = [h for h in hits if h is not None]
                checks.append(_check("prefix_reuse", "pass",
                                     f"prefix hit > 0 on turns 2+ (min {min(known):.3f}, mean {sum(known) / len(known):.3f})", name))
        if s["kind"] == "repeat_prompt":
            later = [r for r in reqs if r.get("ok") and (r.get("repeat") or 0) >= 2]
            hits = [r.get("prefix_hit") for r in later if r.get("prefix_hit") is not None]
            if later and not hits:
                checks.append(_check("repeat_reuse", "warn", "prefix hit unknown (server reports no cached tokens)", name))
            elif hits and min(hits) < 0.9:
                checks.append(_check("repeat_reuse", "warn", f"repeat prefix hit min {min(hits):.3f} < 0.9", name))
            elif hits:
                checks.append(_check("repeat_reuse", "pass", f"repeat prefix hit min {min(hits):.3f}", name))
        if s["kind"] == "long_context":
            rec = [r.get("recall_correct") for r in reqs if r.get("recall_correct") is not None]
            if rec:
                n_ok = sum(1 for x in rec if x)
                checks.append(_check("recall", "pass" if n_ok == len(rec) else "warn",
                                     f"{n_ok}/{len(rec)} planted facts recalled exactly", name))
        if s["kind"] == "agent_alternate":
            reuse = s.get("alternate_reuse") or alternate_reuse(reqs)
            checks.append(_check("alternate_prefix_reuse", "pass" if reuse.get("pass") else "fail",
                                 f"A turn 2+ measured cache ratios {reuse.get('ratios') or 'unknown'}", name))
        if s["kind"] == "concurrency_stall":
            failures = s.get("stall_failures") or []
            if s.get("stall_failures") is None or len(reqs) != 3:
                failures = [*failures, "three-client stall observations missing"]
            checks.append(_check("concurrency_stall", "fail" if failures else "pass",
                                 "; ".join(failures) if failures else "three keyed clients completed without stalls, 503s, or marker leaks", name))
            if any(r.get("ok") and not r.get("marker_present") for r in reqs):
                checks.append(_check("concurrency_canary", "warn", "one or more clients did not echo their marker", name))
        if s["kind"] == "priority_contention":
            interactive = s.get("interactive_ttft_s")
            okay = interactive is not None and s.get("overlapped") and not s.get("contention_failure")
            detail = (f"interactive TTFT {interactive}s; overlapped={s.get('overlapped')}" if interactive is not None
                      else "interactive request missing")
            checks.append(_check("priority_contention", "pass" if okay else "fail", detail, name))
        if s["kind"] == "simultaneous_siblings":
            failures = s.get("sibling_failures") or []
            checks.append(_check("simultaneous_siblings", "fail" if failures else "pass",
                                 "; ".join(failures) if failures else
                                 f"all siblings completed cleanly (overlapped_pairs={s.get('overlapped_siblings', 0)})",
                                 name))
        if s["kind"] == "long_trailing":
            tr = s.get("trailing") or {}
            okay = bool(tr.get("trailing_ok")) and tr.get("peers_ok") == tr.get("peers") and tr.get("peers", 0) > 0
            detail = (f"peers_ok={tr.get('peers_ok')}/{tr.get('peers')}, trailing_ok={tr.get('trailing_ok')}, "
                      f"peers_before_trailing={tr.get('peers_finished_before_trailing')}, "
                      f"solo_tail_s={_fmt(tr.get('solo_tail_s'))}")
            checks.append(_check("long_trailing", "pass" if okay else "fail", detail, name))
        if s["kind"] == "warm_tool_return":
            reuse = s.get("warm_tool_reuse") or warm_tool_return_reuse(reqs)
            if reuse.get("failed_turns") or reuse.get("turns", 0) == 0:
                checks.append(_check("warm_tool_return_reuse", "fail",
                                     f"warm tool-return turns failed or missing: {reuse.get('failed_turns')}", name))
            elif not reuse.get("known"):
                checks.append(_check("warm_tool_return_reuse", "warn",
                                     f"prefix reuse unknown on warm tool-return turns {reuse.get('unknown_turns')}", name))
            elif not reuse.get("pass"):
                checks.append(_check("warm_tool_return_reuse", "warn",
                                     f"warm tool-return prefix hit ratios {reuse.get('ratios')} below {reuse.get('minimum')}",
                                     name))
            else:
                checks.append(_check("warm_tool_return_reuse", "pass",
                                     f"warm tool-return prefix hit ratios {reuse.get('ratios')}", name))
        sp = s.get("spill")
        if sp and sp.get("spill"):
            checks.append(_check("vram_spill", "warn",
                                 f"shared GPU memory {sp['excess_mib']:+.0f} MiB over the clean baseline "
                                 f"(> {sp['margin_mib']:.0f} MiB): VRAM spilled to system memory", name))
    # Stable vs volatile comparison, when both ran.
    st = [s for s in scen if s["kind"] == "agent_session" and not s["config"].get("volatile_top")]
    vo = [s for s in scen if s["kind"] == "agent_session" and s["config"].get("volatile_top")]
    if st and vo:
        def later_mean(s):
            h = [r["prefix_hit"] for r in s["requests"] if r.get("ok") and (r.get("turn") or 0) >= 2
                 and r.get("prefix_hit") is not None]
            return sum(h) / len(h) if h else None
        a, b = later_mean(st[0]), later_mean(vo[0])
        if a is not None and b is not None:
            checks.append(_check("volatile_cache_loss", "pass" if b < a else "warn",
                                 f"turn 2+ mean prefix hit: stable {a:.3f}, volatile-top {b:.3f}"))
    # GPU
    gpu = results.get("gpu") or {}
    if gpu.get("available"):
        good = [x for x in gpu.get("samples") or [] if not x.get("error") and "dedicated_mib" in x]
        if not good:
            errs = [x.get("error") for x in gpu.get("samples") or []]
            checks.append(_check("gpu_used", "warn", f"PDH sampling produced no data: {errs[:1]}"))
        elif max(x["dedicated_mib"] for x in good) <= 0:
            checks.append(_check("gpu_used", "fail", "server process holds 0 MiB dedicated GPU memory: running on CPU?"))
        else:
            checks.append(_check("gpu_used", "pass",
                                 f"dedicated {max(x['dedicated_mib'] for x in good):.0f} MiB, shared baseline "
                                 f"{gpu.get('shared_baseline_mib')} MiB ({gpu.get('baseline_source')})"))
    else:
        checks.append(_check("gpu_used", "skip", f"GPU sampling off: {gpu.get('reason')}"))
    # Identity
    srv = results.get("server") or {}
    answered = [k for k in ("models", "props", "health", "sonder_health", "sonder_identity")
                if (srv.get(k) or {}).get("status") == 200]
    checks.append(_check("server_identity", "pass" if answered else "warn",
                         ("recorded " + ", ".join(answered)) if answered else "no identity endpoint answered 200"))
    if "provenance" in results or require_provenance:
        prov_errs = validate_provenance(results.get("provenance"), strict=require_provenance)
        checks.append(_check(
            "provenance_schema" if not require_provenance else "provenance_complete",
            "fail" if prov_errs else "pass",
            "; ".join(prov_errs) if prov_errs else "reproducible provenance schema valid",
        ))
    if scen and total_ok == 0:
        checks.append(_check("any_request", "fail", "no request succeeded"))
    return checks


def verdict(checks: list[dict]) -> str:
    st = {c["status"] for c in checks}
    return "fail" if "fail" in st else ("warn" if "warn" in st else "pass")


def load_config(path: str | None) -> dict:
    if not path:
        return copy.deepcopy(BUILTIN_SCENARIOS)
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    errs = validate_config(cfg)
    if errs:
        raise ValueError("invalid scenario config: " + "; ".join(errs))
    return cfg


def default_out_prefix(label: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (label or "run"))
    return os.path.join("bench", "http", "out", f"{stamp}-{safe}")
