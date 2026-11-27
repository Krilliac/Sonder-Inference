"""Tests for AGX-01: P90, per-user output tok/s, realized prefix-hit rate,
and InferenceX AgentX Weka trace-replay workload against a checked-in fixture.
"""

from __future__ import annotations

import gc
import json
import math
import threading
import time
import tracemalloc
from pathlib import Path

import pytest

import bench_http
from fake_server import FakeServer
from httpbench.agentx import (
    DEFAULT_CONCURRENCY_LEVELS,
    DEFAULT_IDLE_GAP_CAP_S,
    DEFAULT_MAX_CONTEXT_LENGTH,
    DEFAULT_TRACE_FIT,
    PromptBlockSynthesizer,
    filter_traces_by_context,
    iter_weka_traces,
    load_and_fit_weka_traces,
    load_weka_traces,
    replay_trace_session,
    scale_trace_to_context,
)
from httpbench.metrics import (
    Chunk,
    StreamTrace,
    cached_tokens,
    derive,
    group_summary,
    percentile,
    prompt_tokens,
    summarize,
)
from httpbench.report import compare
from httpbench.runner import validate_config
from httpbench.textgen import Sizer

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "agentx_synthetic_traces.jsonl"


# ---------------------------------------------------------------------- metrics: P90, per-user tok/s, realized prefix hit

def test_summarize_includes_p90():
    xs = list(range(1, 101))
    s = summarize(xs)
    assert "p90" in s
    assert s["p90"] == pytest.approx(percentile(xs, 90))
    assert math.isclose(s["p90"], 90.1)
    assert s["p50"] <= s["p90"] <= s["p95"] <= s["p99"]
    assert summarize([]) == {"n": 0}


def test_cached_tokens_and_realized_prefix_hit_support_usage_sonder_timings():
    # Direct usage.prompt_tokens_details.cached_tokens
    u_details = {"prompt_tokens": 200, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 150}}
    assert cached_tokens(u_details, None) == (150, "usage.prompt_tokens_details.cached_tokens")

    # Direct usage.sonder.timings.cache_n (even when trace.timings is None)
    u_sonder = {
        "prompt_tokens": 200,
        "completion_tokens": 10,
        "sonder": {"timings": {"prompt_n": 40, "cache_n": 160}},
    }
    assert cached_tokens(u_sonder, None) == (160, "usage.sonder.timings.cache_n")
    assert prompt_tokens(u_sonder, None) == 200

    # Derive uses usage.sonder.timings.cache_n directly when trace.timings is None
    tr = StreamTrace(
        status=200,
        done=True,
        end_t=1.2,
        chunks=[Chunk(0.2, "a"), Chunk(0.7, "b"), Chunk(1.2, "c")],
        usage=u_sonder,
        timings=None,
    )
    m = derive(tr)
    assert m["cached_tokens"] == 160
    assert m["cached_tokens_source"] == "usage.sonder.timings.cache_n"
    assert m["cache_n"] == 160
    assert m["prompt_n"] == 40
    assert m["prefix_hit"] == pytest.approx(0.8)
    assert m["output_tps_per_user"] == pytest.approx(9 / 1.0)  # (10 - 1) / (1.2 - 0.2)

    # Reported zero stays zero; unknown stays None
    tr_zero = StreamTrace(
        status=200,
        done=True,
        end_t=0.5,
        chunks=[Chunk(0.1, "a"), Chunk(0.5, "b")],
        usage={"prompt_tokens": 100, "completion_tokens": 2, "sonder": {"timings": {"cache_n": 0, "prompt_n": 100}}},
    )
    assert derive(tr_zero)["prefix_hit"] == 0.0

    tr_unk = StreamTrace(
        status=200,
        done=True,
        end_t=0.5,
        chunks=[Chunk(0.1, "a"), Chunk(0.5, "b")],
        usage={"prompt_tokens": 100, "completion_tokens": 2},
    )
    assert derive(tr_unk)["prefix_hit"] is None
    assert derive(tr_unk)["cached_tokens"] is None


def test_group_summary_reports_per_user_output_tps_p90_and_token_weighted_prefix_hit():
    # Turn 1: 100 prompt, 0 cached, 20 tok/s
    # Turn 2: 900 prompt, 810 cached, 50 tok/s
    # Unweighted mean prefix hit = (0.0 + 0.9) / 2 = 0.45
    # Realized (token-weighted) prefix hit = (0 + 810) / (100 + 900) = 0.81
    reqs = [
        {"ok": True, "session_id": "u1", "ttft_s": 0.1, "e2e_s": 0.6, "itl_s": [0.05],
         "prompt_tokens": 100, "cached_tokens": 0, "prefix_hit": 0.0,
         "completion_tokens": 11, "decode_tps_client": 20.0, "output_tps_per_user": 20.0},
        {"ok": True, "session_id": "u1", "ttft_s": 0.05, "e2e_s": 0.25, "itl_s": [0.02],
         "prompt_tokens": 900, "cached_tokens": 810, "prefix_hit": 0.9,
         "completion_tokens": 11, "decode_tps_client": 50.0, "output_tps_per_user": 50.0},
        {"ok": True, "session_id": "u2", "ttft_s": 0.05, "e2e_s": 0.15, "itl_s": [0.01],
         "prompt_tokens": 1000, "cached_tokens": 900, "prefix_hit": 0.9,
         "completion_tokens": 11, "decode_tps_client": 100.0, "output_tps_per_user": 100.0},
    ]
    g = group_summary(reqs)
    assert g["output_tps_per_user"]["n"] == 3
    assert g["output_tps_per_user"]["p50"] == pytest.approx(50.0)
    assert g["output_tps_per_user"]["p90"] == pytest.approx(percentile([20.0, 50.0, 100.0], 90))
    assert g["output_tps_per_user_p90"] == pytest.approx(g["output_tps_per_user"]["p90"])
    assert g["output_tps_by_session"]["n"] == 2  # u1 mean=35.0, u2 mean=100.0
    assert g["output_tps_by_session"]["p50"] == pytest.approx(67.5)
    assert g["cached_tokens_total"] == 1710
    assert g["prefix_hit_prompt_tokens_total"] == 2000
    assert g["realized_prefix_hit_rate"] == pytest.approx(1710 / 2000)

    # Unknown cache stays None (never 0)
    unk = [{"ok": True, "ttft_s": 0.1, "e2e_s": 0.2, "itl_s": [], "prompt_tokens": 100,
            "cached_tokens": None, "prefix_hit": None, "completion_tokens": 4, "output_tps_per_user": 30.0}]
    gu = group_summary(unk)
    assert gu["realized_prefix_hit_rate"] is None
    assert gu["cached_tokens_total"] is None
    assert gu["prefix_hit_prompt_tokens_total"] is None


# ---------------------------------------------------------------------- trace loader, filtering, prompt synthesis

def test_load_fixture_and_filter_by_max_context_length():
    assert FIXTURE_PATH.is_file()
    traces = load_weka_traces(FIXTURE_PATH)
    assert [t.trace_id for t in traces] == ["synth_trace_01", "synth_trace_02", "synth_trace_oversized_64k"]
    assert traces[0].block_size == 64
    assert traces[0].subagent_count == 2
    assert traces[0].total_requests == 5  # 2 main + 2 sub_explore_1 + 1 sub_grep_2
    assert traces[1].total_requests == 3

    assert DEFAULT_MAX_CONTEXT_LENGTH == 32768
    assert DEFAULT_IDLE_GAP_CAP_S == 10.0
    assert DEFAULT_CONCURRENCY_LEVELS == (1, 2, 4, 8)

    kept, skipped = filter_traces_by_context(traces, DEFAULT_MAX_CONTEXT_LENGTH)
    assert [t.trace_id for t in kept] == ["synth_trace_01", "synth_trace_02"]
    assert len(skipped) == 1
    assert skipped[0]["trace_id"] == "synth_trace_oversized_64k"
    assert skipped[0]["peak_tokens"] == 65536 + 256

    # Tight cap filters everything explicitly
    kept_tight, skipped_tight = filter_traces_by_context(traces, 128)
    assert kept_tight == []
    assert len(skipped_tight) == 3


def test_load_weka_traces_rejects_hostile_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_weka_traces(tmp_path / "missing.jsonl")

    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_weka_traces(bad_json)

    neg_in = tmp_path / "neg.jsonl"
    neg_in.write_text(json.dumps({"id": "x", "requests": [{"t": 0.0, "in": -64, "out": 4, "hash_ids": []}]}),
                      encoding="utf-8")
    with pytest.raises(ValueError):
        load_weka_traces(neg_in)

    bool_hash = tmp_path / "bool_hash.jsonl"
    bool_hash.write_text(json.dumps({"id": "x", "requests": [{"t": 0.0, "in": 64, "out": 4, "hash_ids": [True]}]}),
                         encoding="utf-8")
    with pytest.raises(ValueError):
        load_weka_traces(bool_hash)

    nan_t = tmp_path / "nan_t.jsonl"
    nan_t.write_text('{"id":"x","requests":[{"t":NaN,"in":64,"out":4,"hash_ids":[0]}]}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_weka_traces(nan_t)

    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_weka_traces(empty_file)

    with pytest.raises(ValueError):
        filter_traces_by_context([], 0)
    with pytest.raises(ValueError):
        filter_traces_by_context([], -100)


def test_prompt_synthesizer_preserves_length_growing_prefix_subagents_and_cache_bust():
    sizer = Sizer(tokenize=None)  # 4 chars/token
    traces = load_weka_traces(FIXTURE_PATH)
    t0 = traces[0]
    synth_u0 = PromptBlockSynthesizer(sizer, session_salt="run1:c2:u0:synth_trace_01", block_size=t0.block_size)
    synth_u1 = PromptBlockSynthesizer(sizer, session_salt="run1:c2:u1:synth_trace_01", block_size=t0.block_size)

    main_turn1 = t0.items[0]  # hash_ids=[0,1,2,3], in=256
    sub_a_turn1 = t0.items[1].requests[0]  # hash_ids=[0,1,10,11], in=256
    sub_a_turn2 = t0.items[1].requests[1]  # hash_ids=[0,1,10,11,12], in=320
    main_turn2 = t0.items[3]  # hash_ids=[0,1,2,3,4,5,6,7], in=512

    p_m1 = synth_u0.build_prompt(main_turn1)
    p_m2 = synth_u0.build_prompt(main_turn2)
    p_sa1 = synth_u0.build_prompt(sub_a_turn1)
    p_sa2 = synth_u0.build_prompt(sub_a_turn2)
    p_u1_m1 = synth_u1.build_prompt(main_turn1)

    # Exact input length preservation (within 1 block tolerance on 4 chars/token)
    assert sizer.measure(p_m1) == 256
    assert sizer.measure(p_m2) == 512
    assert sizer.measure(p_sa1) == 256
    assert sizer.measure(p_sa2) == 320

    # Growing shared prefix: main turn 2 starts byte-for-byte with main turn 1
    assert p_m2.startswith(p_m1)
    assert p_sa2.startswith(p_sa1)

    # Subagent shares parent's first 2 blocks ([0, 1] = 128 tokens) byte-for-byte, then diverges
    prefix_2_blocks = synth_u0.block(0) + synth_u0.block(1)
    assert p_m1.startswith(prefix_2_blocks)
    assert p_sa1.startswith(prefix_2_blocks)
    assert p_sa1 != p_m1

    # Different concurrent users get distinct prefixes (cache-bust across users)
    assert not p_u1_m1.startswith(synth_u0.block(0))


def test_replay_caps_idle_gaps_at_10s_and_runs_subagents_concurrently():
    traces = load_weka_traces(FIXTURE_PATH)
    t0 = traces[0]
    sizer = Sizer(tokenize=None)

    sleeps: list[float] = []
    subagent_barrier = threading.Barrier(2, timeout=2.0)
    concurrent_subagents_observed = threading.Event()
    call_order: list[str] = []
    lock = threading.Lock()

    def fake_send(req_meta: dict, prompt: str) -> dict:
        branch = req_meta["branch"]
        with lock:
            call_order.append(f"{branch}:t{req_meta['turn']}")
        # Both subagent branches in synth_trace_01 must enter concurrently!
        if req_meta["is_subagent"] and req_meta["turn"] == 1:
            subagent_barrier.wait()
            concurrent_subagents_observed.set()
        return {
            "ok": True,
            "prompt_tokens": sizer.measure(prompt),
            "completion_tokens": req_meta["target_output_tokens"],
            "cached_tokens": 128 if req_meta["turn"] > 1 or req_meta["is_subagent"] else 0,
            "prefix_hit": 0.5 if req_meta["turn"] > 1 or req_meta["is_subagent"] else 0.0,
            "output_tps_per_user": 50.0,
            "decode_tps_client": 50.0,
            "ttft_s": 0.02,
            "e2e_s": 0.1,
            "itl_s": [0.02],
        }

    rows = replay_trace_session(
        t0,
        sizer=sizer,
        session_id="c1-u0-synth_trace_01",
        session_salt="salt0",
        send_fn=fake_send,
        idle_gap_cap_s=10.0,
        sleep_fn=sleeps.append,
    )
    assert len(rows) == 5
    assert concurrent_subagents_observed.is_set(), "subagent branches did not execute concurrently"
    # Main turn 1 runs before subagents; subagents finish before main turn 2
    assert call_order[0] == "main:t1"
    assert call_order[-1] == "main:t2"
    # Main turn 2 has raw think_time=20.0 -> capped at 10.0
    main_t2 = next(r for r in rows if r["branch"] == "main" and r["turn"] == 2)
    assert main_t2["raw_idle_gap_s"] == pytest.approx(20.0)
    assert main_t2["idle_gap_s"] == pytest.approx(10.0)
    assert max(sleeps) == pytest.approx(10.0)
    assert all(0.0 <= s <= 10.0 for s in sleeps)

    # Also verify synth_trace_02: gap 2.5s preserved (<10s), gap 15.0s capped to 10.0s
    sleeps2: list[float] = []
    rows2 = replay_trace_session(
        traces[1],
        sizer=sizer,
        session_id="c1-u0-synth_trace_02",
        session_salt="salt1",
        send_fn=lambda meta, prompt: {"ok": True, "prompt_tokens": sizer.measure(prompt),
                                      "completion_tokens": 5, "output_tps_per_user": 40.0},
        idle_gap_cap_s=10.0,
        sleep_fn=sleeps2.append,
    )
    assert [r["idle_gap_s"] for r in rows2] == pytest.approx([0.0, 2.5, 10.0])
    assert [r["raw_idle_gap_s"] for r in rows2] == pytest.approx([0.0, 2.5, 15.0])
    assert sleeps2 == pytest.approx([2.5, 10.0])


# ---------------------------------------------------------------------- CLI + end-to-end against FakeServer

def test_cli_max_context_length_default_and_agentx_preset(capsys):
    args = bench_http.build_parser().parse_args(["run", "--base-url", "http://127.0.0.1:1"])
    assert args.max_context_length == 32768
    assert args.trace_fit == "scale"
    assert DEFAULT_TRACE_FIT == "scale"

    assert bench_http.main(["run", "--base-url", "http://127.0.0.1:1", "--max-context-length", "0"]) == 2
    assert bench_http.main(["run", "--base-url", "http://127.0.0.1:1", "--max-context-length", "-5"]) == 2
    with pytest.raises(SystemExit) as exc:
        bench_http.build_parser().parse_args(["run", "--base-url", "http://127.0.0.1:1", "--trace-fit", "bogus"])
    assert exc.value.code == 2

    capsys.readouterr()
    assert bench_http.main(["scenarios", "--preset", "agentx"]) == 0
    preset = json.loads(capsys.readouterr().out)
    assert validate_config(preset) == []
    sc = preset["scenarios"][0]
    assert sc["kind"] == "agentx_trace"
    assert sc["levels"] == [1, 2, 4, 8]
    assert sc["max_context_length"] == 32768
    assert sc["trace_fit"] == "scale"
    assert sc["idle_gap_cap_s"] == 10.0


def test_end_to_end_agentx_trace_concurrency_1_2_4_8_against_fake_server(tmp_path):
    cfg = {
        "schema": "sonder.bench.http.scenarios/1",
        "defaults": {"temperature": 0, "seed": 42},
        "scenarios": [{
            "name": "agentx-trace",
            "kind": "agentx_trace",
            "trace_path": str(FIXTURE_PATH),
            "levels": [1, 2, 4, 8],
            "trace_fit": "drop",
            "idle_gap_cap_s": 10.0,
            "idle_time_scale": 0.0,  # don't sleep 10s in unit tests
        }],
    }
    cfg_path = tmp_path / "agentx.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    prefix = tmp_path / "out" / "agentx_run"

    # Use cache_report="usage" (OpenAI usage.prompt_tokens_details.cached_tokens) with keyed_cache=True
    with FakeServer(cache_report="usage", keyed_cache=True, delay=0.002, n_ctx=65536) as srv:
        rc = bench_http.main([
            "run", "--base-url", srv.url, "--scenarios", str(cfg_path),
            "--out", str(prefix), "--label", "agentx-test", "--quiet",
        ])

    assert rc == 0
    res = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    md = prefix.with_suffix(".md").read_text(encoding="utf-8")
    assert res["verdict"] == "pass", [c for c in res["checks"] if c["status"] not in ("pass", "skip")]

    sc = res["scenarios"][0]
    assert sc["kind"] == "agentx_trace"
    assert sc["max_context_length"] == 32768
    assert len(sc["skipped_traces"]) == 1
    assert sc["skipped_traces"][0]["trace_id"] == "synth_trace_oversized_64k"

    # Concurrency levels 1, 2, 4, 8 all ran with real metrics
    for level in (1, 2, 4, 8):
        g = sc["groups"][f"c{level}"]
        assert g["concurrency"] == level
        assert g["ok"] == g["requests"] and g["ok"] > 0
        assert g["output_tps_per_user"]["n"] > 0
        assert g["output_tps_per_user"]["p90"] > 0
        assert g["output_tps_per_user_p90"] == pytest.approx(g["output_tps_per_user"]["p90"])
        assert g["ttft_s"]["p90"] is not None
        assert g["itl_s"]["p90"] is not None
        assert g["realized_prefix_hit_rate"] is not None and g["realized_prefix_hit_rate"] > 0.0

    # AgentX headline summary and markdown output
    assert "agentx_summary" in sc
    assert set(sc["agentx_summary"]["levels"]) == {"c1", "c2", "c4", "c8"}
    assert "AgentX" in md
    assert "P90" in md or "p90" in md
    assert "realized prefix hit" in md.lower()

    # Compare includes P90 and per-user output tok/s metrics
    _diff_md, rows = compare(res, res)
    metrics_seen = {r["metric"] for r in rows}
    assert "per-user output tok/s p90" in metrics_seen
    assert "TTFT p90 ms" in metrics_seen
    assert "realized prefix hit" in metrics_seen


def test_agentx_cli_trace_file_shorthand_and_unknown_cache_warns(tmp_path):
    prefix = tmp_path / "shorthand"
    with FakeServer(cache_report="none", delay=0.001, n_ctx=65536) as srv:
        rc = bench_http.main([
            "run", "--base-url", srv.url,
            "--trace-file", str(FIXTURE_PATH),
            "--concurrency-levels", "1,2",
            "--max-context-length", "32768",
            "--idle-time-scale", "0",
            "--out", str(prefix), "--quiet",
        ])
    res = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert rc == 0 and res["verdict"] == "warn"
    sc = res["scenarios"][0]
    assert set(sc["groups"]) == {"c1", "c2"}
    assert sc["groups"]["c1"]["realized_prefix_hit_rate"] is None
    checks = {c["check"]: c for c in res["checks"]}
    assert checks["agentx_prefix_reuse"]["status"] == "warn"
    assert "unknown" in checks["agentx_prefix_reuse"]["detail"]


def test_agentx_all_traces_filtered_by_max_context_length_is_non_vacuous(tmp_path):
    prefix = tmp_path / "filtered_out"
    with FakeServer(delay=0.001) as srv:
        rc = bench_http.main([
            "run", "--base-url", srv.url,
            "--trace-file", str(FIXTURE_PATH),
            "--concurrency-levels", "1",
            "--max-context-length", "64",
            "--trace-fit", "drop",
            "--idle-time-scale", "0",
            "--out", str(prefix), "--quiet", "--strict",
        ])
    assert rc == 1
    res = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert res["verdict"] in ("warn", "fail")
    assert res["scenarios"][0]["expected_requests"] == 0
    assert len(res["scenarios"][0]["skipped_traces"]) == 3


def test_out_zero_accepted_and_clamped_to_one_with_metadata(tmp_path):
    traces = load_weka_traces(FIXTURE_PATH)
    t2 = traces[1]
    assert t2.trace_id == "synth_trace_02"
    # First request has type="n" and ttft=0.15; second request has out=0
    assert t2.items[0].req_type == "n"
    assert t2.items[0].ttft == pytest.approx(0.15)
    assert t2.items[1].output_tokens == 0

    # Negative out is still rejected
    neg_out = tmp_path / "neg_out.jsonl"
    neg_out.write_text(json.dumps({"id": "x", "requests": [{"t": 0.0, "in": 64, "out": -1, "hash_ids": [0]}]}),
                       encoding="utf-8")
    with pytest.raises(ValueError):
        load_weka_traces(neg_out)

    # End-to-end replay of synth_trace_02 against FakeServer sends max_tokens=1 for out=0 and records clamp
    only_t2 = tmp_path / "only_t2.jsonl"
    lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
    only_t2.write_text(lines[1] + "\n", encoding="utf-8")
    prefix = tmp_path / "out_zero"
    with FakeServer(cache_report="usage", keyed_cache=True, delay=0.001, n_ctx=65536) as srv:
        rc = bench_http.main([
            "run", "--base-url", srv.url,
            "--trace-file", str(only_t2),
            "--concurrency-levels", "1",
            "--idle-time-scale", "0",
            "--out", str(prefix), "--quiet",
        ])
        chat_posts = [r["body"] for r in srv.state.requests if r["path"] == "/v1/chat/completions"]
    assert rc == 0
    assert [p["max_tokens"] for p in chat_posts] == [10, 1, 10]
    res = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    reqs = res["scenarios"][0]["requests"]
    assert reqs[1]["target_output_tokens"] == 0
    assert reqs[1]["effective_output_tokens"] == 1
    assert reqs[1]["output_tokens_clamped"] is True
    assert reqs[0]["output_tokens_clamped"] is False


def test_trace_fit_scale_preserves_topology_prefix_and_reports_transparency(tmp_path):
    traces = load_weka_traces(FIXTURE_PATH)
    big = traces[2]
    assert big.trace_id == "synth_trace_oversized_64k"
    assert big.peak_context_tokens == 65536 + 256

    scaled = scale_trace_to_context(big, 2048)
    assert scaled.original_peak_tokens == 65536 + 256
    assert 0.0 < scaled.scale_factor < 1.0
    assert scaled.peak_context_tokens <= 2048
    assert scaled.subagent_count == 1
    assert scaled.total_requests == 3

    # Verify prefix-sharing preservation after scaling
    m1 = scaled.items[0]
    sub1 = scaled.items[1].requests[0]
    m2 = scaled.items[2]
    assert m1.input_tokens + max(1, m1.output_tokens) <= 2048
    assert m2.input_tokens + max(1, m2.output_tokens) <= 2048
    assert len(m2.hash_ids) >= len(m1.hash_ids) >= 1
    assert m2.hash_ids[: len(m1.hash_ids)] == m1.hash_ids
    # Subagent still shares leading blocks with parent
    assert sub1.hash_ids[0] == m1.hash_ids[0]

    # End-to-end CLI run with default trace_fit="scale" at max-context-length=2048 replays the 64K trace!
    only_big = tmp_path / "only_big.jsonl"
    lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
    only_big.write_text(lines[2] + "\n", encoding="utf-8")
    prefix = tmp_path / "scaled_run"
    with FakeServer(cache_report="usage", keyed_cache=True, delay=0.001, n_ctx=4096) as srv:
        rc = bench_http.main([
            "run", "--base-url", srv.url,
            "--trace-file", str(only_big),
            "--max-context-length", "2048",
            "--concurrency-levels", "1,2",
            "--idle-time-scale", "0",
            "--out", str(prefix), "--quiet",
        ])
    assert rc == 0
    res = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    md = prefix.with_suffix(".md").read_text(encoding="utf-8")
    assert res["verdict"] == "pass", [c for c in res["checks"] if c["status"] not in ("pass", "skip")]
    sc = res["scenarios"][0]
    assert sc["trace_fit"] == "scale"
    assert sc["traces_kept"] == 1
    assert sc["skipped_traces"] == []
    assert len(sc["scaled_traces"]) == 1
    st = sc["scaled_traces"][0]
    assert st["trace_id"] == "synth_trace_oversized_64k"
    assert st["original_peak_tokens"] == 65792
    assert st["scaled_peak_tokens"] <= 2048
    assert 0.0 < st["scale_factor"] < 0.05
    assert sc["agentx_summary"]["trace_fit"] == "scale"
    assert sc["agentx_summary"]["scaled_traces"] == sc["scaled_traces"]
    assert "scale" in md.lower() and "65792" in md
    assert sc["groups"]["c1"]["realized_prefix_hit_rate"] > 0.0


def test_streaming_loader_drops_rejected_traces_without_retaining_file_memory(tmp_path):
    # Write ~4 MB JSONL file of oversized traces (20 traces x 10 requests x 2000 hash_ids)
    # plus 2 tiny fitting traces at the end. Streaming drop mode must discard rejected
    # traces immediately instead of retaining all hash tuples in memory.
    big_jsonl = tmp_path / "large_traces.jsonl"
    hashes = list(range(2000))
    n_oversized = 20
    with big_jsonl.open("w", encoding="utf-8") as fh:
        for i in range(n_oversized):
            row = {
                "id": f"oversized_{i}",
                "block_size": 64,
                "requests": [
                    {"t": float(j), "type": "s", "in": 128000, "out": 64, "hash_ids": hashes}
                    for j in range(10)
                ],
            }
            fh.write(json.dumps(row) + "\n")
        for i in range(2):
            row = {
                "id": f"small_{i}",
                "block_size": 64,
                "requests": [{"t": 0.0, "type": "s", "in": 256, "out": 8, "hash_ids": [0, 1, 2, 3]}],
            }
            fh.write(json.dumps(row) + "\n")

    file_bytes = big_jsonl.stat().st_size
    assert file_bytes > 2_000_000

    # Verify generator streams incrementally
    gen = iter_weka_traces(big_jsonl)
    first = next(gen)
    assert first.trace_id == "oversized_0"
    gen.close()

    gc.collect()
    tracemalloc.start()
    try:
        before_cur, _ = tracemalloc.get_traced_memory()
        fit = load_and_fit_weka_traces(big_jsonl, max_context_length=1024, trace_fit="drop")
        gc.collect()
        after_cur, peak_mem = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    retained = max(0, after_cur - before_cur)
    assert [t.trace_id for t in fit.traces] == ["small_0", "small_1"]
    assert len(fit.skipped_traces) == n_oversized
    assert fit.traces_loaded == n_oversized + 2
    # Retained memory must be tiny (only 2 small traces + 20 skip dicts), nowhere near 4+ MB / 30+ MB!
    assert retained < 512_000, f"retained {retained} bytes; rejected hash_ids were not dropped"
    # Peak memory while streaming one trace at a time must stay far below retaining all 20 traces
    assert peak_mem < 8_000_000, f"peak {peak_mem} bytes indicates whole-file buffering"

    # Also verify max_traces early-stops scaling without parsing the entire file
    fit_early = load_and_fit_weka_traces(big_jsonl, max_context_length=1024, trace_fit="scale", max_traces=2)
    assert len(fit_early.traces) == 2
    assert fit_early.traces_loaded == 2
    assert fit_early.truncated_at_max_traces is True
    assert all(t.peak_context_tokens <= 1024 for t in fit_early.traces)



def test_hostile_metrics_and_cli_trace_inputs(tmp_path):
    # Non-finite values and booleans are ignored in summarize/percentile; negative cached_tokens ignored.
    s = summarize([10.0, float("nan"), float("inf"), True, None, 20.0])  # type: ignore[list-item]
    assert s["n"] == 2 and s["p50"] == 15.0 and s["p90"] == pytest.approx(19.0)
    assert cached_tokens({"prompt_tokens_details": {"cached_tokens": -5}}, None) == (None, None)
    assert cached_tokens({"sonder": {"timings": {"cache_n": float("nan")}}}, None) == (None, None)
    assert cached_tokens(None, {"cache_n": -1}) == (None, None)

    # Control chars in trace_id or agent_id rejected
    ctrl_trace = tmp_path / "ctrl.jsonl"
    ctrl_trace.write_text(json.dumps({
        "id": "bad\nid",
        "requests": [{"t": 0.0, "in": 64, "out": 4, "hash_ids": [0]}],
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        load_weka_traces(ctrl_trace)

    # Nonexistent --trace-file and invalid idle/concurrency flags exit 2 before contacting server
    assert bench_http.main(["run", "--base-url", "http://127.0.0.1:1", "--trace-file", str(tmp_path / "nope.jsonl")]) == 2
    assert bench_http.main(["run", "--base-url", "http://127.0.0.1:1", "--concurrency-levels", "1,0"]) == 2
    assert bench_http.main(["run", "--base-url", "http://127.0.0.1:1", "--idle-gap-cap", "-1"]) == 2
    assert bench_http.main(["run", "--base-url", "http://127.0.0.1:1", "--idle-time-scale", "nan"]) == 2

    # End-to-end HTTP stream returning ONLY usage.sonder.timings.cache_n (SI openai.cpp shape)
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _SonderUsageOnly(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        def log_message(self, *a):
            pass
        def do_GET(self):
            if self.path == "/v1/models":
                raw = json.dumps({"object": "list", "data": [{"id": "si-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self.send_response(404)
            self.end_headers()
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            ptok = max(1, len(body["messages"][0]["content"]) // 4)
            cache_n = ptok // 2
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for tok in ("one ", "two ", "three ", "four "):
                self.wfile.write(f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': tok}}]})}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.002)
            final_usage = {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": ptok,
                    "completion_tokens": 4,
                    "total_tokens": ptok + 4,
                    "sonder": {"timings": {"prompt_n": ptok - cache_n, "cache_n": cache_n}},
                },
            }
            self.wfile.write(f"data: {json.dumps(final_usage)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SonderUsageOnly)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        out_prefix = tmp_path / "sonder_usage"
        rc = bench_http.main([
            "run", "--base-url", f"http://127.0.0.1:{srv.server_port}",
            "--trace-file", str(FIXTURE_PATH),
            "--concurrency-levels", "1",
            "--idle-time-scale", "0",
            "--out", str(out_prefix), "--quiet",
        ])
        assert rc == 0
        res = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
        assert res["verdict"] == "pass"
        g = res["scenarios"][0]["groups"]["c1"]
        assert g["realized_prefix_hit_rate"] == pytest.approx(0.5, abs=0.02)
        assert g["output_tps_per_user_p90"] > 0
    finally:
        srv.shutdown()
        srv.server_close()

