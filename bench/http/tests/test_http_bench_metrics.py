import math

import pytest

from httpbench.gpumem import (expected_clean_shared, parse_counter_output, parse_tasklist_csv,
                              spill_check, counter_command)
from httpbench.metrics import (Chunk, StreamTrace, cached_tokens, derive, group_summary,
                               inter_token_latencies, percentile, prompt_tokens, summarize)
from httpbench.report import compare
from httpbench.runner import run_checks, validate_config, verdict, BUILTIN_SCENARIOS


def test_percentile_linear_interpolation_matches_numpy_default():
    xs = [1, 2, 3, 4]
    assert percentile(xs, 50) == 2.5
    assert percentile(xs, 0) == 1
    assert percentile(xs, 100) == 4
    assert math.isclose(percentile(xs, 95), 3.85)
    assert math.isclose(percentile(list(range(1, 101)), 99), 99.01)
    assert percentile([], 50) is None
    assert percentile([7], 99) == 7


def test_summarize_ignores_none_and_reports_empty():
    s = summarize([None, 3.0, 1.0, None, 2.0])
    assert s["n"] == 3 and s["mean"] == 2.0 and s["min"] == 1.0 and s["max"] == 3.0 and s["p50"] == 2.0
    assert summarize([None]) == {"n": 0}


def test_prefix_hit_sources_and_unknown_is_never_zero():
    assert cached_tokens(None, {"cache_n": 90}) == (90, "timings.cache_n")
    assert cached_tokens({"prompt_tokens_details": {"cached_tokens": 5}}, None) == \
        (5, "usage.prompt_tokens_details.cached_tokens")
    # timings wins over usage when both are present
    assert cached_tokens({"prompt_tokens_details": {"cached_tokens": 5}}, {"cache_n": 7})[0] == 7
    assert cached_tokens({"prompt_tokens": 10}, {"prompt_n": 10}) == (None, None)

    tr = StreamTrace(status=200, done=True, chunks=[Chunk(0.1, "a")], end_t=0.2,
                     usage={"prompt_tokens": 100, "completion_tokens": 1})
    m = derive(tr)
    assert m["prefix_hit"] is None and m["cached_tokens"] is None

    tr.timings = {"prompt_n": 10, "cache_n": 90}
    m = derive(tr)
    assert m["prompt_tokens"] == 100
    assert m["prefix_hit"] == pytest.approx(0.9)

    tr.timings = {"prompt_n": 100, "cache_n": 0}
    assert derive(tr)["prefix_hit"] == 0.0  # reported zero stays zero


def test_prompt_tokens_adds_llama_cache_n_to_processed():
    # llama-server: prompt_n counts only processed tokens
    assert prompt_tokens(None, {"prompt_n": 12, "cache_n": 988}) == 1000
    # usage that excludes cached tokens loses to prompt_n + cache_n
    assert prompt_tokens({"prompt_tokens": 12}, {"prompt_n": 12, "cache_n": 988}) == 1000
    assert prompt_tokens({"prompt_tokens": 1000}, {"prompt_n": 12, "cache_n": 988}) == 1000
    assert prompt_tokens(None, None, {"tokens_evaluated": 55}) == 55
    assert prompt_tokens(None, None) is None


def test_itl_divides_chunk_gap_by_tokens_per_chunk():
    chunks = [Chunk(1.0, "ab"), Chunk(1.2, "cd"), Chunk(1.6, "ef")]
    # 6 tokens over 3 chunks -> 2 tokens per chunk
    assert inter_token_latencies(chunks, 6) == pytest.approx([0.1, 0.2])
    # unknown token count -> 1 token per chunk
    assert inter_token_latencies(chunks, None) == pytest.approx([0.2, 0.4])
    # explicit per-chunk counts win
    chunks[2].n_tokens = 4
    assert inter_token_latencies(chunks, 6) == pytest.approx([0.1, 0.1])
    assert inter_token_latencies(chunks[:1], 1) == []
    # role-only / empty chunks carry no tokens and are skipped
    assert inter_token_latencies([Chunk(0.5), Chunk(1.0, "x"), Chunk(1.5, "y")], 2) == pytest.approx([0.5])


def test_derive_client_rates_and_draft_acceptance():
    tr = StreamTrace(status=200, done=True, end_t=2.0,
                     chunks=[Chunk(0.5, "", reasoning="think")] + [Chunk(0.5 + 0.1 * i, "t") for i in range(1, 11)],
                     usage={"prompt_tokens": 1000, "completion_tokens": 16},
                     timings={"prompt_n": 200, "cache_n": 800, "predicted_per_second": 42.0,
                              "prompt_per_second": 900.0, "draft_n": 20, "draft_n_accepted": 15})
    m = derive(tr)
    assert m["ok"]
    assert m["ttft_s"] == 0.5  # reasoning tokens count as first token
    assert m["completion_tokens"] == 16 and m["completion_tokens_source"] == "usage"
    assert m["decode_tps_client"] == pytest.approx(15 / 1.5)
    assert m["prefill_tps_client"] == pytest.approx(200 / 0.5)  # uncached tokens only
    assert m["decode_tps_server"] == 42.0 and m["prefill_tps_server"] == 900.0
    assert m["draft_acceptance"] == pytest.approx(0.75)
    assert len(m["itl_s"]) == 10


def test_derive_not_ok_without_done_or_on_error():
    tr = StreamTrace(status=200, done=False, chunks=[Chunk(0.1, "a")], end_t=0.2)
    assert derive(tr)["ok"] is False
    tr = StreamTrace(status=200, done=True, error="stream error", chunks=[], end_t=0.2)
    assert derive(tr)["ok"] is False
    # completion tokens fall back to counting chunks
    tr = StreamTrace(status=200, done=True, chunks=[Chunk(0.1, "a"), Chunk(0.2, "b")], end_t=0.3)
    m = derive(tr)
    assert m["completion_tokens"] == 2 and m["completion_tokens_source"] == "chunks"


def test_group_summary_prefix_hit_unknown_vs_known():
    r1 = {"ok": True, "ttft_s": 0.1, "e2e_s": 1.0, "itl_s": [0.01, 0.02], "prefix_hit": None, "completion_tokens": 5}
    g = group_summary([r1])
    assert g["prefix_hit"] is None and g["prefix_hit_known"] == 0
    r2 = dict(r1, prefix_hit=0.5)
    g = group_summary([r1, r2, {"ok": False, "error": "boom"}])
    assert g["prefix_hit"]["mean"] == 0.5 and g["ok"] == 2 and g["requests"] == 3
    assert g["itl_s"]["n"] == 4 and g["errors"] == ["boom"]


def test_pdh_parsing_and_spill_flag():
    out = ("pid_1234_luid_0x0_0x1_phys_0|shared usage|314572800\n"
           "pid_1234_luid_0x0_0x1_phys_0|dedicated usage|15032385536\n"
           "pid_1234_luid_0x0_0x2_phys_0|dedicated usage|0\n"
           "garbage line\n")
    r = parse_counter_output(out)
    assert r["shared_mib"] == 300.0 and r["dedicated_mib"] == 14336.0 and len(r["raw"]) == 2
    assert spill_check(r, 148.0) == {"spill": False, "excess_mib": 152.0, "margin_mib": 256.0}
    assert spill_check(r, 40.0)["spill"] is True
    assert spill_check(r, None) is None
    assert spill_check({"error": "x"}, 10.0) is None
    assert expected_clean_shared(148, 49152, 131072) == pytest.approx(148 + 80)
    assert expected_clean_shared(148, 49152, None) == 148
    cmd = counter_command([42])
    assert "pid_42_*" in cmd and "Shared Usage" in cmd and "Dedicated Usage" in cmd


def test_tasklist_csv():
    out = '"llama-server.exe","5512","Console","1","1,234 K"\n"other.exe","1","Console","1","1 K"\n'
    assert parse_tasklist_csv(out, "llama-server.exe") == [5512]
    assert parse_tasklist_csv("INFO: No tasks are running which match the specified criteria.\n", "x.exe") == []


def _res(scen, gpu=None, server=None):
    return {"scenarios": scen, "gpu": gpu or {"available": False, "reason": "off"},
            "server": server or {"models": {"status": 200}}}


def _agent(hits, volatile=False, ok=True):
    reqs = [{"ok": ok, "turn": i + 1, "prefix_hit": h, "completion_tokens": 3} for i, h in enumerate(hits)]
    return {"name": "agent-volatile" if volatile else "agent", "kind": "agent_session",
            "config": {"volatile_top": volatile}, "requests": reqs, "expected_requests": len(hits),
            "summary": {"errors": []}}


def test_checks_warn_on_zero_or_unknown_prefix_hit():
    c = {x["check"]: x for x in run_checks(_res([_agent([0.0, 0.8, 0.9])]))}
    assert c["prefix_reuse"]["status"] == "pass"
    c = {x["check"]: x for x in run_checks(_res([_agent([0.0, 0.0, 0.9])]))}
    assert c["prefix_reuse"]["status"] == "warn" and "turn(s) [2]" in c["prefix_reuse"]["detail"]
    c = {x["check"]: x for x in run_checks(_res([_agent([None, None, None])]))}
    assert c["prefix_reuse"]["status"] == "warn" and "unknown" in c["prefix_reuse"]["detail"]


def test_checks_fail_on_missing_requests_and_zero_tokens():
    a = _agent([0.1, 0.5])
    a["expected_requests"] = 3
    checks = run_checks(_res([a]))
    assert verdict(checks) == "fail"
    assert any(c["check"] == "request_count" and c["status"] == "fail" for c in checks)
    b = _agent([0.1, 0.5])
    for r in b["requests"]:
        r["completion_tokens"] = 0
    assert any(c["check"] == "tokens_generated" and c["status"] == "fail" for c in run_checks(_res([b])))
    assert verdict(run_checks(_res([]))) == "fail"


def test_checks_gpu_used():
    gpu = {"available": True, "samples": [{"label": "baseline", "dedicated_mib": 0.0, "shared_mib": 1.0}]}
    c = {x["check"]: x for x in run_checks(_res([_agent([0, 0.9])], gpu=gpu))}
    assert c["gpu_used"]["status"] == "fail"
    gpu["samples"][0]["dedicated_mib"] = 14000.0
    c = {x["check"]: x for x in run_checks(_res([_agent([0, 0.9])], gpu=gpu))}
    assert c["gpu_used"]["status"] == "pass"
    gpu["samples"] = [{"label": "baseline", "error": "no process"}]
    c = {x["check"]: x for x in run_checks(_res([_agent([0, 0.9])], gpu=gpu))}
    assert c["gpu_used"]["status"] == "warn"


def test_volatile_comparison_check():
    checks = run_checks(_res([_agent([0.0, 0.9, 0.9]), _agent([0.0, 0.01, 0.01], volatile=True)]))
    c = {x["check"]: x for x in checks}
    assert c["volatile_cache_loss"]["status"] == "pass"


def test_config_validation():
    assert validate_config(BUILTIN_SCENARIOS) == []
    errs = validate_config({"scenarios": [{"kind": "nope"}, {"kind": "long_context", "sizes": [-1], "depths": [2]}]})
    assert len(errs) == 3
    assert validate_config({"x": 1})


def test_compare_marks_direction():
    def res(ttft, tps, label):
        return {"label": label, "target": {"base_url": "u", "model": "m"}, "scenarios": [
            {"name": "s", "groups": {"g": {"ttft_s": {"n": 1, "p50": ttft}, "decode_tps_client": {"n": 1, "mean": tps}}}}]}
    md, rows = compare(res(0.2, 40.0, "A"), res(0.1, 50.0, "B"))
    by = {r["metric"]: r for r in rows}
    assert by["TTFT p50 ms"]["delta"] == pytest.approx(-100.0) and by["TTFT p50 ms"]["better"] is True
    assert by["decode tok/s (client)"]["delta_pct"] == pytest.approx(25.0) and by["decode tok/s (client)"]["better"]
    assert "| s | g | TTFT p50 ms |" in md and "better" in md
