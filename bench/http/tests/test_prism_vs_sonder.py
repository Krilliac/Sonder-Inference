"""Comprehensive unit & integration tests for scripts/bench_prism_vs_sonder.py.

Covers:
  - Corpus structure & all 12 reference quality answers
  - SSE streaming TTFT, wall decode tok/s, total wall tok/s, and server speculation timings
  - Warm multi-turn KV prefix-cache speedup and cache_n tracking across repeats
  - HTTP/1.1 persistent keep-alive socket reuse vs fresh-socket connections
  - Concurrency aggregation, makespan overlap, and linear-interpolated p50/p95 math
  - All 4 deterministic quality graders (math, 3s-timeout Python subprocess runner,
    strict JSON schema validator including bool!=int guard, and logic/regex checker)
  - Endpoint parity matrix (Prism 6/8 vs Sonder 8/8)
  - Cross-target output parity (exact match %, SequenceMatcher token similarity %, pass-rate delta)
  - Managed sequential server lifecycle (1 server at a time on a shared port, RSS sampling, tree teardown)
  - JSON and side-by-side Markdown report generation
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bench_prism_vs_sonder as bps  # noqa: E402


# ---------------------------------------------------------------------------
# Threaded Fake OpenAI + Sonder HTTP/1.1 Server
# ---------------------------------------------------------------------------


def _approx_tok(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


class FakeH2HState:
    def __init__(
        self,
        *,
        model_id: str = "Bonsai-2-9B-MTP",
        keep_alive: bool = True,
        sonder_routes: bool = True,
        prefill_base_s: float = 0.004,
        prefill_per_uncached_tok_s: float = 0.0,
        chunk_delay_s: float = 0.003,
        default_chunks: int = 5,
        draft_n: int | None = 12,
        draft_n_accepted: int | None = 9,
        server_decode_tps: float = 140.0,
        server_prefill_tps: float = 2800.0,
        reasoning_first_chunk: str = "",
        quality_overrides: dict[str, str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.keep_alive = keep_alive
        self.sonder_routes = sonder_routes
        self.prefill_base_s = prefill_base_s
        self.prefill_per_uncached_tok_s = prefill_per_uncached_tok_s
        self.chunk_delay_s = chunk_delay_s
        self.default_chunks = default_chunks
        self.draft_n = draft_n
        self.draft_n_accepted = draft_n_accepted
        self.server_decode_tps = server_decode_tps
        self.server_prefill_tps = server_prefill_tps
        self.reasoning_first_chunk = reasoning_first_chunk
        self.quality_overrides = dict(quality_overrides or {})

        corpus, _ = bps.load_corpus_file(bps.DEFAULT_CORPUS_PATH)
        self.probe_answers_by_prompt: dict[str, str] = {}
        for p in corpus["quality_probes"]:
            ans = self.quality_overrides.get(p["id"], p["reference_answer"])
            self.probe_answers_by_prompt[p["prompt"]] = ans

        self.lock = threading.Lock()
        self.slot_cache: dict[str, str] = {}
        self.last_rendered = ""
        self.chat_requests: list[dict[str, Any]] = []
        self.accepted_connections = 0

    def match_reply(self, messages: list[dict[str, Any]], response_format: dict[str, Any] | None) -> list[str]:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content") or "")
                break
        if last_user in self.probe_answers_by_prompt:
            full = self.probe_answers_by_prompt[last_user]
            # Split into deterministic multi-chunk pieces while preserving exact concatenation.
            step = max(1, (len(full) + 3) // 4)
            return [full[i : i + step] for i in range(0, len(full), step)] or [full]
        if response_format and response_format.get("type") == "json_object":
            js = '{"ok":true,"suite":"bonsai","slots":4}'
            return [js[:12], js[12:25], js[25:]]
        return [f"tok{i} " for i in range(self.default_chunks)]


class CountingThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler_cls: Any, state: FakeH2HState) -> None:
        self.h2h_state = state
        super().__init__(server_address, handler_cls)

    def get_request(self) -> tuple[Any, Any]:
        sock, addr = super().get_request()
        sock.settimeout(5.0)
        with self.h2h_state.lock:
            self.h2h_state.accepted_connections += 1
        return sock, addr


def _make_handler(state: FakeH2HState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1" if state.keep_alive else "HTTP/1.0"

        def log_message(self, *args: Any) -> None:
            pass

        def _send_json(self, status: int, payload: Any) -> None:
            raw = json.dumps(payload).encode("utf-8")
            client_wants_close = "close" in (self.headers.get("Connection") or "").lower()
            should_close = (not state.keep_alive) or client_wants_close
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close" if should_close else "keep-alive")
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()
            if should_close:
                self.close_connection = True

        def do_GET(self) -> None:
            if self.path == "/health":
                return self._send_json(200, {"status": "ok"})
            if self.path == "/v1/sonder/health":
                if not state.sonder_routes:
                    return self._send_json(404, {"error": {"message": "not found"}})
                return self._send_json(200, {"status": "ok", "sonder": {"api_version": 1, "backend": "llamacpp"}})
            if self.path == "/v1/models":
                return self._send_json(
                    200,
                    {"object": "list", "data": [{"id": state.model_id, "object": "model"}]},
                )
            return self._send_json(404, {"error": {"message": f"unknown GET {self.path}"}})

        def do_POST(self) -> None:
            clen = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(clen) if clen > 0 else b"{}"
            body = json.loads(raw_body.decode("utf-8") or "{}")

            if self.path == "/v1/tokenize":
                if not state.sonder_routes:
                    return self._send_json(404, {"error": {"message": "not found"}})
                txt = str(body.get("content") or body.get("text") or "")
                toks = list(range(1, _approx_tok(txt) + 1))
                return self._send_json(200, {"tokens": toks, "count": len(toks), "model": state.model_id})

            if self.path == "/v1/embeddings":
                inp = body.get("input")
                items = [inp] if isinstance(inp, str) else (inp if isinstance(inp, list) else [])
                if not items:
                    return self._send_json(400, {"error": {"message": "empty input"}})
                data = [
                    {"object": "embedding", "index": idx, "embedding": [0.125, -0.25, 0.5, 0.75]}
                    for idx in range(len(items))
                ]
                return self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": data,
                        "model": state.model_id,
                        "usage": {"prompt_tokens": 6 * len(items), "total_tokens": 6 * len(items)},
                    },
                )

            if self.path == "/v1/completions":
                prompt = str(body.get("prompt") or "")
                if not prompt:
                    return self._send_json(400, {"error": {"message": "missing prompt"}})
                return self._send_json(
                    200,
                    {
                        "id": "cmpl-fake-1",
                        "object": "text_completion",
                        "model": state.model_id,
                        "choices": [{"text": " 4", "index": 0, "logprobs": None, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": _approx_tok(prompt), "completion_tokens": 1, "total_tokens": _approx_tok(prompt) + 1},
                    },
                )

            if self.path != "/v1/chat/completions":
                return self._send_json(404, {"error": {"message": f"unknown POST {self.path}"}})

            messages = body.get("messages") or []
            resp_fmt = body.get("response_format")
            stream = bool(body.get("stream", False))
            max_tokens = max(1, int(body.get("max_tokens", 32)))
            rendered = "".join(f"<|{m.get('role')}|>{m.get('content')}" for m in messages) + "<|assistant|>"

            with state.lock:
                state.chat_requests.append(body)
                cache_key = str(body.get("prompt_cache_key") or "__default__")
                prev = state.slot_cache.get(cache_key, state.last_rendered)
                lcp_chars = _common_prefix_len(prev, rendered)
                p_tok = _approx_tok(rendered)
                cache_n = min(lcp_chars // 4, max(p_tok - 1, 0))
                uncached = max(p_tok - cache_n, 1)

            pieces = state.match_reply(messages, resp_fmt)[:max_tokens]
            full_reply = "".join(pieces)
            with state.lock:
                state.slot_cache[cache_key] = rendered + full_reply
                state.last_rendered = rendered + full_reply

            if not stream:
                return self._send_json(
                    200,
                    {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion",
                        "model": state.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": full_reply},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": p_tok,
                            "completion_tokens": len(pieces),
                            "total_tokens": p_tok + len(pieces),
                        },
                    },
                )

            prefill_sleep = state.prefill_base_s + state.prefill_per_uncached_tok_s * uncached
            if prefill_sleep > 0:
                time.sleep(prefill_sleep)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def send_sse(event_payload: Any) -> None:
                txt = event_payload if isinstance(event_payload, str) else json.dumps(event_payload)
                self.wfile.write(f"data: {txt}\n\n".encode("utf-8"))
                self.wfile.flush()

            send_sse({"choices": [{"index": 0, "delta": {"role": "assistant"}}], "model": state.model_id})

            emitted_tokens = 0
            if state.reasoning_first_chunk:
                send_sse(
                    {
                        "choices": [{"index": 0, "delta": {"reasoning_content": state.reasoning_first_chunk}}],
                        "model": state.model_id,
                    }
                )
                emitted_tokens += 1
                if state.chunk_delay_s > 0:
                    time.sleep(state.chunk_delay_s)

            for idx, piece in enumerate(pieces):
                send_sse({"choices": [{"index": 0, "delta": {"content": piece}}], "model": state.model_id})
                emitted_tokens += 1
                if idx + 1 < len(pieces) and state.chunk_delay_s > 0:
                    time.sleep(state.chunk_delay_s)

            timings: dict[str, Any] = {
                "prompt_n": uncached,
                "cache_n": cache_n,
                "prompt_ms": round(prefill_sleep * 1000.0, 3),
                "prompt_per_second": state.server_prefill_tps,
                "predicted_n": emitted_tokens,
                "predicted_ms": round(max(emitted_tokens - 1, 1) * state.chunk_delay_s * 1000.0, 3),
                "predicted_per_second": state.server_decode_tps,
            }
            if state.draft_n is not None:
                timings["draft_n"] = state.draft_n
                timings["draft_n_accepted"] = state.draft_n_accepted

            send_sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "timings": timings})
            if (body.get("stream_options") or {}).get("include_usage"):
                send_sse(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": p_tok,
                            "completion_tokens": emitted_tokens,
                            "total_tokens": p_tok + emitted_tokens,
                            "prompt_tokens_details": {"cached_tokens": cache_n},
                        },
                    }
                )
            send_sse("[DONE]")

    return Handler


class FakeH2HServer:
    def __init__(self, port: int = 0, **kwargs: Any) -> None:
        self.state = FakeH2HState(**kwargs)
        self.httpd = CountingThreadingHTTPServer(("127.0.0.1", port), _make_handler(self.state), self.state)
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> FakeH2HServer:
        self.thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Unit & Integration Tests
# ---------------------------------------------------------------------------


def test_corpus_structure_and_all_12_reference_graders() -> None:
    corpus, fnv_hash = bps.load_corpus_file(bps.DEFAULT_CORPUS_PATH)
    assert corpus["schema"] == "sonder.inference.corpus/1"
    assert corpus["name"] == "bonsai_head2head"
    assert len(fnv_hash) == 16

    # Cold prefill: short (~128), medium (~1024), long (~3072)
    prefill = corpus["prefill"]
    assert [p["id"] for p in prefill] == ["short", "medium", "long"]
    assert 96 <= _approx_tok(prefill[0]["prompt"]) <= 180
    assert 850 <= _approx_tok(prefill[1]["prompt"]) <= 1250
    assert 2700 <= _approx_tok(prefill[2]["prompt"]) <= 3500

    # Warm multi-turn prefix cache: 3 turns sharing ~1500-token prefix
    pc = corpus["prefix_cache"]
    assert len(pc["turns"]) == 3
    assert 1300 <= _approx_tok(pc["system_prefix"]) <= 1750

    # 4 Decode regimes
    regimes = {r["id"]: r for r in corpus["decode_regimes"]}
    assert set(regimes) == {"code_edit", "structured_json", "reasoning_math", "open_prose"}
    assert regimes["structured_json"]["response_format"] == {"type": "json_object"}

    # 12 Quality probes across 4 categories (3 each), all passing their own deterministic graders
    # when unsafe python execution is explicitly opted in (and safely skipping python by default).
    probes = corpus["quality_probes"]
    assert len(probes) == 12
    counts: dict[str, int] = {}
    for probe in probes:
        assert probe["temperature"] == 0
        assert probe["seed"] == 42
        counts[probe["category"]] = counts.get(probe["category"], 0) + 1
        default_grade = bps.grade_probe(probe["reference_answer"], probe)
        if probe["category"] == "python_code":
            assert default_grade["skipped"] is True
            assert default_grade["passed"] is False
            grade = bps.grade_probe(probe["reference_answer"], probe, allow_unsafe_python_exec=True)
            assert grade["passed"] is True, f"Reference answer failed for {probe['id']}: {grade}"
        else:
            assert default_grade["passed"] is True, f"Reference answer failed for {probe['id']}: {default_grade}"
    assert counts == {"math": 3, "python_code": 3, "json_schema": 3, "logic": 3}


def test_sse_ttft_decode_tps_and_speculation_timings() -> None:
    with FakeH2HServer(
        prefill_base_s=0.035,
        chunk_delay_s=0.015,
        default_chunks=5,
        draft_n=16,
        draft_n_accepted=12,
        reasoning_first_chunk="thinking...",
    ) as srv:
        client = bps.BenchmarkClient(srv.url, timeout=10.0)
        res = client.stream_chat(
            [{"role": "user", "content": "Measure SSE timings."}],
            model="Bonsai-2-9B-MTP",
            max_tokens=16,
            disable_thinking=True,
        )

    assert res["ok"] is True
    assert res["http_status"] == 200
    # 1 reasoning_content chunk + 5 content chunks = 6 emitted tokens
    assert res["tokens"] == 6
    assert res["reasoning_content"] == "thinking..."
    assert res["ttft_ms"] is not None and res["ttft_ms"] >= 25.0
    assert res["total_s"] > res["ttft_s"] > 0.0

    expected_decode_tps = (6 - 1) / (res["total_s"] - res["ttft_s"])
    expected_total_tps = 6 / res["total_s"]
    assert abs(res["wall_decode_tps"] - expected_decode_tps) < 1e-6
    assert abs(res["total_wall_tps"] - expected_total_tps) < 1e-6

    assert res["timings"]["draft_n"] == 16
    assert res["timings"]["draft_n_accepted"] == 12
    assert res["acceptance_ratio"] == 0.75
    assert srv.state.chat_requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_warm_prefix_cache_speedup_and_cache_n() -> None:
    corpus, _ = bps.load_corpus_file(bps.DEFAULT_CORPUS_PATH)
    # 60 us per uncached token -> ~1550 cold tokens sleep ~93 ms; ~45 warm uncached tokens sleep ~2.7 ms!
    with FakeH2HServer(
        prefill_base_s=0.002,
        prefill_per_uncached_tok_s=0.00006,
        chunk_delay_s=0.001,
    ) as srv:
        client = bps.BenchmarkClient(srv.url, timeout=30.0)
        prefix_res = bps.run_prefix_suite(
            client,
            corpus,
            model="Bonsai-2-9B-MTP",
            repeats=2,
            max_tokens=8,
            max_tokens_explicit=True,
            disable_thinking=True,
        )

    assert prefix_res["cold"]["ok_count"] == 2
    assert prefix_res["warm"]["ok_count"] == 4
    # Both repeats must start cold on Turn 1 (at most the ~2-token `<|system|>[` ChatML header matches).
    assert prefix_res["cold_cache_n"]["max"] <= 4
    # Warm turns 2 & 3 must reuse the ~1500-token prefix.
    assert prefix_res["warm_cache_n"]["median"] >= 1300
    assert prefix_res["cold_ttft_ms"]["median"] > prefix_res["warm_ttft_ms"]["median"]
    assert prefix_res["speedup_median_ratio"] is not None and prefix_res["speedup_median_ratio"] > 2.0
    assert prefix_res["speedup"]["mean"] is not None and prefix_res["speedup"]["mean"] > 2.0


def test_keepalive_connection_reuse_vs_fresh_sockets() -> None:
    # 1. HTTP/1.1 Keep-Alive enabled server: 10 requests reuse 1 single TCP connection!
    with FakeH2HServer(keep_alive=True) as srv_ka:
        client = bps.BenchmarkClient(srv_ka.url, timeout=10.0)
        ka_res = bps.run_keepalive_suite(client, num_requests=10)
        total_accepted = srv_ka.state.accepted_connections

    assert ka_res["keepalive"]["ok_count"] == 10
    assert ka_res["keepalive"]["connections_opened"] == 1
    assert ka_res["keepalive"]["reused_requests"] == 9
    assert ka_res["fresh_socket"]["ok_count"] == 10
    assert ka_res["fresh_socket"]["connections_opened"] == 10
    assert ka_res["fresh_socket"]["reused_requests"] == 0
    # Server ground-truth: 1 socket for persistent phase + 10 sockets for fresh phase = 11 total!
    assert total_accepted == 11
    assert ka_res["speedup_mean"] is not None and ka_res["speedup_mean"] > 0.0

    # 2. HTTP/1.0 Connection: close server: persistent client cleanly reconnects without errors!
    with FakeH2HServer(keep_alive=False) as srv_close:
        client_close = bps.BenchmarkClient(srv_close.url, timeout=10.0)
        close_res = bps.run_keepalive_suite(client_close, num_requests=6)
        accepted_close = srv_close.state.accepted_connections

    assert close_res["keepalive"]["ok_count"] == 6
    assert close_res["keepalive"]["connections_opened"] == 6
    assert close_res["keepalive"]["reused_requests"] == 0
    assert accepted_close == 12


def test_percentile_and_concurrency_aggregation() -> None:
    assert bps.percentile([], 95) is None
    assert bps.percentile([42.0], 95) == 42.0
    # [10, 20, 30, 40, 100]: p50 -> index 2 = 30.0; p95 -> rank 3.8 = 40 + 0.8 * (100 - 40) = 88.0
    sample = [40.0, 10.0, 100.0, 20.0, 30.0]
    assert bps.percentile(sample, 50) == 30.0
    assert abs((bps.percentile(sample, 95) or 0.0) - 88.0) < 1e-9
    # Booleans must never be treated as numbers
    stats = bps.summarize_metric([True, 10.0, 20.0, None])  # type: ignore[list-item]
    assert stats["n"] == 2
    assert stats["median"] == 15.0
    assert stats["mean"] == 15.0

    corpus, _ = bps.load_corpus_file(bps.DEFAULT_CORPUS_PATH)
    with FakeH2HServer(prefill_base_s=0.015, chunk_delay_s=0.010, default_chunks=6) as srv:
        client = bps.BenchmarkClient(srv.url, timeout=20.0)
        conc = bps.run_concurrency_suite(
            client,
            corpus,
            model="Bonsai-2-9B-MTP",
            levels=[1, 2, 4],
            repeats=2,
            max_tokens=6,
            max_tokens_explicit=True,
            disable_thinking=True,
        )

    c1 = conc["levels"]["1"]
    c4 = conc["levels"]["4"]
    assert c1["success_count"] == 2 and c1["total_count"] == 2
    assert c4["success_count"] == 8 and c4["total_count"] == 8
    assert c4["ttft_p50_ms"] <= c4["ttft_p95_ms"]
    assert c4["aggregate_tps"]["median"] > c1["aggregate_tps"]["median"] * 1.5


def test_quality_graders_positive_and_adversarial() -> None:
    # 1. Math grader (including <think> stripping & comma formatting)
    m_spec = {"expected": 2900, "tolerance": 1e-6}
    assert bps.grade_math("<think>99999</think>\nWork...\nFINAL_ANSWER: 2,900", m_spec)["passed"] is True
    assert bps.grade_math(r"Result is \boxed{2900}", m_spec)["passed"] is True
    assert bps.grade_math("FINAL_ANSWER: 2901", m_spec)["passed"] is False
    assert bps.grade_math("No digits at all.", m_spec)["passed"] is False

    # 2. Python grader: disabled by default (never executes untrusted code!),
    #    and when opted in, validates scrubbed env, early-exit sentinel, syntax, and timeout.
    py_spec = {
        "entry_point": "add_pair",
        "timeout_s": 0.4,
        "test_code": "assert add_pair(2, 3) == 5\nassert add_pair(-1, 1) == 0\n",
    }
    # Default mode does NOT execute candidate code!
    default_py = bps.grade_python_code(
        "```python\nimport sys\nsys.exit(99)\ndef add_pair(a, b):\n    return a + b\n```",
        py_spec,
    )
    assert default_py["skipped"] is True and default_py["passed"] is False
    # Opted-in execution checks sentinel (sys.exit(0) cannot spoof pass!) and strips env secrets.
    early_exit = bps.grade_python_code(
        "```python\nimport sys\ndef add_pair(a, b):\n    sys.exit(0)\n```",
        py_spec,
        allow_unsafe_exec=True,
    )
    assert early_exit["passed"] is False and "exited early" in early_exit["detail"].lower()

    assert bps.grade_python_code(
        "```python\ndef add_pair(a, b):\n    return a + b\n```",
        py_spec,
        allow_unsafe_exec=True,
    )["passed"] is True
    bad_ans = bps.grade_python_code(
        "```python\ndef add_pair(a, b):\n    return a * b\n```",
        py_spec,
        allow_unsafe_exec=True,
    )
    assert bad_ans["passed"] is False and "AssertionError" in bad_ans["detail"]
    syn_err = bps.grade_python_code(
        "```python\ndef add_pair(a, b)\n    return a + b\n```",
        py_spec,
        allow_unsafe_exec=True,
    )
    assert syn_err["passed"] is False and "SyntaxError" in syn_err["detail"]

    t0 = time.perf_counter()
    hang_res = bps.grade_python_code(
        "```python\ndef add_pair(a, b):\n    while True:\n        pass\n```",
        py_spec,
        timeout_s=0.4,
        allow_unsafe_exec=True,
    )
    elapsed = time.perf_counter() - t0
    assert hang_res["passed"] is False
    assert "timeout" in hang_res["detail"].lower()
    assert elapsed < 4.0

    # 3. JSON schema validator (including Python bool-is-subclass-of-int trap & exact_keys!)
    j_spec = {
        "exact_keys": True,
        "required_keys": {"name": "str", "slots": "int", "active": "bool", "tags": "list[str]"},
        "expected_values": {"slots": 4, "active": True},
    }
    assert bps.grade_json_schema('{"name":"sonder","slots":4,"active":true,"tags":["cuda"]}', j_spec)["passed"] is True
    assert bps.grade_json_schema('```json\n{"name":"sonder","slots":4,"active":true,"tags":[]}\n```', j_spec)["passed"] is True
    extra_k = bps.grade_json_schema('{"name":"sonder","slots":4,"active":true,"tags":[],"extra":1}', j_spec)
    assert extra_k["passed"] is False and "unexpected extra key" in extra_k["detail"]
    # Bool passed for int key MUST fail!
    bool_as_int = bps.grade_json_schema('{"name":"sonder","slots":true,"active":true,"tags":[]}', j_spec)
    assert bool_as_int["passed"] is False and "slots" in bool_as_int["detail"]
    missing_k = bps.grade_json_schema('{"name":"sonder","slots":4,"active":true}', j_spec)
    assert missing_k["passed"] is False and "missing required key" in missing_k["detail"]
    assert bps.grade_json_schema("[1, 2, 3]", j_spec)["passed"] is False

    # 4. Logic / instruction-following checker
    l_spec = {
        "non_empty_line_count": 2,
        "no_blank_lines": True,
        "line_regex": r"^-\s+.+\.$",
        "regex_all": [r"(?i)alpha", r"(?i)omega"],
        "regex_none": [r"(?i)\bhowever\b"],
    }
    assert bps.grade_logic("- Alpha starts first.\n- Omega finishes second.", l_spec)["passed"] is True
    assert bps.grade_logic("- Alpha starts first.\n\n- Omega finishes second.", l_spec)["passed"] is False
    assert bps.grade_logic("- Alpha starts, however.\n- Omega finishes second.", l_spec)["passed"] is False
    assert bps.grade_logic("- Alpha without period\n- Omega finishes second.", l_spec)["passed"] is False


def test_endpoint_parity_matrix_prism_vs_sonder() -> None:
    with FakeH2HServer(sonder_routes=False) as prism_srv, FakeH2HServer(sonder_routes=True) as sonder_srv:
        p_ep = bps.run_endpoints_suite(bps.BenchmarkClient(prism_srv.url), model="Bonsai-2-9B-MTP")
        s_ep = bps.run_endpoints_suite(bps.BenchmarkClient(sonder_srv.url), model="Bonsai-2-9B-MTP")

    assert p_ep["supported_count"] == 6
    assert p_ep["total_count"] == 8
    assert p_ep["by_id"]["tokenize"]["supported"] is False
    assert p_ep["by_id"]["tokenize"]["http_status"] == 404
    assert p_ep["by_id"]["sonder_health"]["supported"] is False

    assert s_ep["supported_count"] == 8
    assert s_ep["total_count"] == 8
    assert all(item["supported"] for item in s_ep["probes"])


def test_attach_mode_cross_target_parity_and_reports(tmp_path: Path) -> None:
    # Prism misses 1 math probe; Sonder passes all 12/12 and rewords 1 logic explanation slightly.
    prism_overrides = {"math_linear_comb": "FINAL_ANSWER: 9999"}
    sonder_overrides = {
        "logic_flag_deduction": (
            "Rule 1 fixes C=true, so Rule 2 forces A=false and Rule 3 gives B=true.\n"
            "FLAGS: A=false, B=true, C=true"
        )
    }

    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    with FakeH2HServer(
        prefill_base_s=0.012,
        chunk_delay_s=0.006,
        sonder_routes=False,
        quality_overrides=prism_overrides,
    ) as prism_srv, FakeH2HServer(
        prefill_base_s=0.003,
        chunk_delay_s=0.001,
        sonder_routes=True,
        quality_overrides=sonder_overrides,
    ) as sonder_srv:
        rc = bps.main([
            "--target", f"prism={prism_srv.url}",
            "--target", f"sonder={sonder_srv.url}",
            "--model", "Bonsai-2-9B-MTP",
            "--suites", "all",
            "--repeats", "2",
            "--max-tokens", "32",
            "--concurrency", "1,2",
            "--keepalive-requests", "5",
            "--disable-thinking",
            "--allow-unsafe-python-exec",
            "--quiet",
            "--output-json", str(out_json),
            "--output-md", str(out_md),
        ])

    assert rc == 0
    assert out_json.is_file() and out_md.is_file()
    assert b"\r\n" not in out_json.read_bytes()
    assert b"\r\n" not in out_md.read_bytes()

    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["schema"] == bps.SCHEMA_VERSION
    assert len(rep["targets"]) == 2
    p_qual = rep["targets"][0]["suites"]["quality"]
    s_qual = rep["targets"][1]["suites"]["quality"]
    assert p_qual["passed"] == 11 and p_qual["total"] == 12
    assert s_qual["passed"] == 12 and s_qual["total"] == 12

    parity = rep["cross_target_parity"]
    assert parity is not None
    comp = parity["comparisons"][0]
    assert comp["baseline"] == "prism"
    assert comp["target"] == "sonder"
    # 10 of 12 probes match identically; 2 differ (math_linear_comb and logic_flag_deduction)
    assert comp["exact_match_count"] == 10
    assert abs(comp["exact_match_pct"] - (100.0 * 10 / 12)) < 1e-6
    assert 80.0 < comp["token_similarity_pct"] < 100.0
    assert abs(comp["pass_rate_delta_pct"] - (100.0 / 12.0)) < 1e-6

    md_text = out_md.read_text(encoding="utf-8")
    assert "## Comparison: `sonder` vs `prism` (Baseline)" in md_text
    assert "**WIN**" in md_text
    assert "### Cross-Target Output Parity Summary" in md_text


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_managed_sequential_mode_tears_down_before_next_target(tmp_path: Path) -> None:
    """Launch two managed targets sequentially on the SAME port to prove clean process teardown."""
    shared_port = _find_free_port()
    events_log = tmp_path / "lifecycle.jsonl"
    cfg_path = tmp_path / "managed_config.json"
    out_json = tmp_path / "managed_out.json"
    self_script = str(Path(__file__).resolve())

    cfg = {
        "targets": [
            {
                "name": "prism",
                "url": f"http://127.0.0.1:{shared_port}",
                "health_path": "/health",
                "model": "Bonsai-2-27B-PQ2",
                "startup_timeout_s": 15.0,
                "cmd": [
                    sys.executable,
                    self_script,
                    "--serve-fake",
                    "--name",
                    "prism",
                    "--port",
                    str(shared_port),
                    "--events-file",
                    str(events_log),
                ],
            },
            {
                "name": "sonder",
                "url": f"http://127.0.0.1:{shared_port}",
                "health_path": "/v1/sonder/health",
                "model": "Bonsai-2-27B-PQ2",
                "startup_timeout_s": 15.0,
                "env": {"SONDER_FAKE_TAG": "sequential_ok"},
                "cmd": [
                    sys.executable,
                    self_script,
                    "--serve-fake",
                    "--name",
                    "sonder",
                    "--port",
                    str(shared_port),
                    "--sonder-routes",
                    "--events-file",
                    str(events_log),
                ],
            },
        ]
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    rc = bps.main([
        "--config", str(cfg_path),
        "--suites", "keepalive,endpoints",
        "--repeats", "1",
        "--keepalive-requests", "4",
        "--quiet",
        "--output-json", str(out_json),
    ])
    assert rc == 0

    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["config"]["mode"] == "managed"
    assert [t["name"] for t in rep["targets"]] == ["prism", "sonder"]
    assert rep["targets"][0]["suites"]["endpoints"]["supported_count"] == 6
    assert rep["targets"][1]["suites"]["endpoints"]["supported_count"] == 8
    # Both managed servers must have positive RSS measurements recorded while alive.
    for t in rep["targets"]:
        assert t["resources"]["peak_rss_mib"] is not None and t["resources"]["peak_rss_mib"] > 0.0
        # And neither PID should remain alive after completion!
        assert bps.sample_process_rss_mib(t["resources"]["pid"]) is None

    events = [json.loads(ln) for ln in events_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    starts = [e for e in events if e["event"] == "started"]
    assert [e["name"] for e in starts] == ["prism", "sonder"]


def test_sse_parser_byte_splits_crlf_and_nested_sonder_timings() -> None:
    parser = bps.SSEParser()
    raw_stream = (
        b": keepalive comment\r\n\r\n"
        b'data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}\r\n\r\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"lo"}}],"usage":{"prompt_tokens":12,"completion_tokens":2,'
        b'"sonder":{"timings":{"cache_n":8,"draft_n":4,"draft_n_accepted":3}}}}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )
    events: list[bps.SSEEvent] = []
    # Feed 3 bytes at a time to exercise arbitrary TCP/CRLF boundary splits.
    for i in range(0, len(raw_stream), 3):
        events.extend(parser.feed(raw_stream[i : i + 3]))
    events.extend(parser.flush())

    assert len(events) == 3
    assert events[-1].is_done is True

    obj2 = json.loads(events[1].data)
    fin = bps._finalize_stream_metrics(
        ok=True,
        http_status=200,
        error=None,
        ttft_s=0.010,
        total_s=0.030,
        token_chunks=2,
        usage=obj2["usage"],
        raw_timings=obj2["usage"]["sonder"]["timings"],
        content="Hello",
        reasoning="",
        finish_reason="stop",
    )
    assert fin["timings"]["cache_n"] == 8
    assert fin["timings"]["draft_n"] == 4
    assert fin["timings"]["draft_n_accepted"] == 3
    assert fin["acceptance_ratio"] == 0.75


def test_target_model_override_autodiscovery_and_single_target_report(tmp_path: Path) -> None:
    out_md = tmp_path / "single.md"
    with FakeH2HServer(model_id="auto-discovered-bonsai-4b") as srv:
        # Omit --model so BenchmarkClient discovers it from GET /v1/models
        rc = bps.main([
            "--target", f"sonder={srv.url}",
            "--suites", "prefill,decode,keepalive",
            "--repeats", "1",
            "--max-tokens", "8",
            "--keepalive-requests", "3",
            "--quiet",
            "--output-md", str(out_md),
        ])
    assert rc == 0
    md = out_md.read_text(encoding="utf-8")
    assert "`auto-discovered-bonsai-4b`" in md
    assert "### 1. Cold Prefill Latency" in md
    assert "### 3. Decode Speed Regimes" in md


def test_managed_server_early_crash_fails_fast(tmp_path: Path) -> None:
    bad_cfg = tmp_path / "bad_managed.json"
    bad_cfg.write_text(
        json.dumps({
            "targets": [
                {
                    "name": "broken",
                    "url": f"http://127.0.0.1:{_find_free_port()}",
                    "startup_timeout_s": 30.0,
                    "cmd": [sys.executable, "-c", "import sys; sys.stderr.write('BOOM_BAD_MODEL\\n'); sys.exit(3)"],
                }
            ]
        }),
        encoding="utf-8",
    )
    t0 = time.perf_counter()
    rc = bps.main(["--config", str(bad_cfg), "--suites", "endpoints", "--quiet"])
    elapsed = time.perf_counter() - t0
    assert rc == 2
    assert elapsed < 5.0


def test_cli_validation_errors(tmp_path: Path) -> None:
    # Neither --target nor --config
    assert bps.main(["--suites", "endpoints", "--quiet"]) == 2
    # Both --target and --config
    dummy_cfg = tmp_path / "dummy.json"
    dummy_cfg.write_text('{"targets":[]}', encoding="utf-8")
    assert bps.main(["--target", "a=http://127.0.0.1:9", "--config", str(dummy_cfg), "--quiet"]) == 2
    # Unknown suite name
    assert bps.main(["--target", "a=http://127.0.0.1:9", "--suites", "nonexistent", "--quiet"]) == 2


def test_parity_excludes_failed_probes_and_nonstreaming_suppresses_wall_decode_tps() -> None:
    # Non-streaming fallback must NOT manufacture a wall_decode_tps from read() cleanup interval.
    m = bps._finalize_stream_metrics(
        ok=True,
        http_status=200,
        error=None,
        ttft_s=None,
        total_s=0.5,
        token_chunks=1,
        usage={"completion_tokens": 50, "prompt_tokens": 10},
        raw_timings={},
        content="hello",
        reasoning="",
        finish_reason="stop",
        streamed=False,
    )
    assert m["wall_decode_tps"] is None
    assert m["total_wall_tps"] == 100.0

    # Two failed/empty probes must NOT count as a 100% exact match in cross-target parity.
    t1 = {
        "name": "prism",
        "suites": {
            "quality": {
                "pass_rate_pct": 50.0,
                "probes": [
                    {"id": "p1", "category": "math", "request_ok": True, "passed": True, "clean_content": "42"},
                    {"id": "p2", "category": "math", "request_ok": False, "passed": False, "clean_content": ""},
                ],
            }
        },
    }
    t2 = {
        "name": "sonder",
        "suites": {
            "quality": {
                "pass_rate_pct": 50.0,
                "probes": [
                    {"id": "p1", "category": "math", "request_ok": True, "passed": True, "clean_content": "42"},
                    {"id": "p2", "category": "math", "request_ok": False, "passed": False, "clean_content": ""},
                ],
            }
        },
    }
    parity = bps.compute_cross_target_parity([t1, t2])
    assert parity is not None
    comp = parity["comparisons"][0]
    assert comp["probes_compared"] == 1
    assert comp["excluded_failed_count"] == 1
    assert comp["exact_match_count"] == 1
    assert comp["exact_match_pct"] == 100.0


# ---------------------------------------------------------------------------
# Subprocess Fake Server Entrypoint for Managed Mode Integration Test
# ---------------------------------------------------------------------------


def _serve_fake_cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-fake", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--sonder-routes", action="store_true")
    ap.add_argument("--events-file", required=True)
    ns = ap.parse_args(argv)

    stop_ev = threading.Event()

    def _on_sig(*_a: Any) -> None:
        stop_ev.set()

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    with FakeH2HServer(
        port=ns.port,
        sonder_routes=bool(ns.sonder_routes),
        prefill_base_s=0.001,
        chunk_delay_s=0.001,
    ):
        with open(ns.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "started", "name": ns.name, "pid": os.getpid(), "t": time.time()}) + "\n")
            f.flush()
        while not stop_ev.is_set():
            stop_ev.wait(0.1)
    return 0


if __name__ == "__main__":
    if "--serve-fake" in sys.argv:
        sys.exit(_serve_fake_cli(sys.argv[1:]))
