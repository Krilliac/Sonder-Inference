import json
import os

import pytest

import bench_http
from fake_server import FakeServer
from httpbench.client import Target, trace_text
from httpbench.metrics import derive
from httpbench.sse import SSEParser


# ---------------------------------------------------------------------- parser

def test_sse_basic_and_done():
    p = SSEParser()
    ev = p.feed(b'data: {"a":1}\n\ndata: [DONE]\n\n')
    assert [e.data for e in ev] == ['{"a":1}', "[DONE]"]
    assert ev[1].is_done and not ev[0].is_done


def test_sse_split_at_every_byte_and_crlf():
    raw = b'event: tick\r\ndata: one\r\ndata:two\r\n\r\n: comment\r\n\r\ndata: x\r\n\r\n'
    for step in (1, 2, 3, 7):
        p = SSEParser()
        out = []
        for i in range(0, len(raw), step):
            out.extend(p.feed(raw[i:i + step]))
        out.extend(p.flush())
        assert [(e.event, e.data) for e in out] == [("tick", "one\ntwo"), ("message", "x")], step


def test_sse_bare_cr_and_trailing_event_without_blank_line():
    p = SSEParser()
    out = p.feed(b"data: a\r\rdata: b\n")
    out += p.flush()
    assert [e.data for e in out] == ["a", "b"]


def test_sse_utf8_split_inside_multibyte_char():
    raw = "data: héllo €\n\n".encode()
    p = SSEParser()
    out = []
    for i in range(len(raw)):
        out.extend(p.feed(raw[i:i + 1]))
    assert out[0].data == "héllo €"


# ---------------------------------------------------------------------- client vs fake server

def _chat(t: Target, content: str, **kw):
    body = {"model": "fake-model", "messages": [{"role": "user", "content": content}], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": 32}
    body.update(kw)
    return t.stream(t.oai("/chat/completions"), body)


def test_stream_chat_metrics_against_fake_server():
    with FakeServer(delay=0.005) as srv:
        t = Target(srv.url)
        tr = _chat(t, "hello there, what next?")
        m = derive(tr)
        assert m["ok"], m["error"]
        assert tr.done and tr.status == 200 and tr.model == "fake-model"
        content, _ = trace_text(tr)
        assert content.startswith("Next I would run")
        assert m["completion_tokens"] == len(content.split()) and m["completion_tokens_source"] == "usage"
        assert m["ttft_s"] > 0 and m["e2e_s"] > m["ttft_s"]
        assert len(m["itl_s"]) == m["chunks"] - 1 > 0
        # each chunk waited ~5 ms on the server
        assert all(x >= 0.003 for x in m["itl_s"])
        assert m["cached_tokens"] == 0 and m["prefix_hit"] == 0.0  # cold, reported as 0
        tr2 = _chat(t, "hello there, what next? And then?")
        m2 = derive(tr2)
        assert 0.3 < m2["prefix_hit"] < 1.0 and m2["cached_tokens_source"] == "timings.cache_n"


def test_stream_multi_token_chunks_and_usage_cache_report():
    with FakeServer(chunk_tokens=3, cache_report="usage", sse_crlf=True) as srv:
        t = Target(srv.url + "/v1")
        _chat(t, "prefix prefix prefix")
        m = derive(_chat(t, "prefix prefix prefix and more"))
        assert m["ok"]
        assert m["cached_tokens_source"] == "usage.prompt_tokens_details.cached_tokens"
        assert m["completion_tokens"] > m["chunks"]  # 3 tokens per chunk


def test_stream_unknown_cache_is_none_not_zero():
    with FakeServer(cache_report="none") as srv:
        t = Target(srv.url)
        _chat(t, "abc")
        m = derive(_chat(t, "abc def"))
        assert m["ok"] and m["prefix_hit"] is None and m["cached_tokens"] is None


def test_stream_http_error_and_midstream_error():
    with FakeServer(status_override=400) as srv:
        m = derive(_chat(Target(srv.url), "x"))
        assert not m["ok"] and m["status"] == 400 and "overridden" in m["error"]
    with FakeServer(fail_after_chunks=2) as srv:
        m = derive(_chat(Target(srv.url), "x"))
        assert not m["ok"] and "backend exploded" in m["error"] and m["chunks"] == 2


def test_stream_connection_refused_is_recorded():
    t = Target("http://127.0.0.1:1", timeout=2)
    m = derive(_chat(t, "x"))
    assert not m["ok"] and m["error"]


def test_stream_raw_completion_route():
    with FakeServer(draft=True) as srv:
        t = Target(srv.url + "/v1")  # /completion hangs off the root, not /v1
        assert t.root("/completion") == "/completion"
        body = {"prompt": "Note: the access phrase for vault Cobalt is red-fox-1234.\nWhat is the access phrase "
                          "for vault Cobalt mentioned above?", "n_predict": 16, "stream": True, "cache_prompt": True}
        tr = t.stream(t.root("/completion"), body, api="completion")
        m = derive(tr)
        assert m["ok"], m["error"]
        assert "red-fox-1234" in trace_text(tr)[0]
        assert m["draft_acceptance"] == pytest.approx(0.7)
        assert m["prompt_tokens"] > 0


def test_target_url_normalisation():
    t = Target("http://127.0.0.1:8080")
    assert (t.oai("/chat/completions"), t.root("/props")) == ("/v1/chat/completions", "/props")
    t = Target("http://127.0.0.1:11437/v1/")
    assert (t.oai("/models"), t.root("/props")) == ("/v1/models", "/props")
    t = Target("https://host/api/v1")
    assert (t.oai("/models"), t.root("/v1/sonder/health"), t.port) == ("/api/v1/models", "/api/v1/sonder/health", 443)
    with pytest.raises(ValueError):
        Target("127.0.0.1:8080")


# ---------------------------------------------------------------------- end to end

SMALL = {
    "schema": "sonder.bench.http.scenarios/1",
    "defaults": {"max_tokens": 24, "temperature": 0, "seed": 1},
    "scenarios": [
        {"name": "agent-stable", "kind": "agent_session", "turns": 3, "system_tokens": 600, "tools": 4,
         "turn_tokens": 60},
        {"name": "agent-volatile-top", "kind": "agent_session", "turns": 3, "system_tokens": 600, "tools": 4,
         "turn_tokens": 60, "volatile_top": True},
        {"name": "fanout", "kind": "concurrency_sweep", "levels": [1, 2, 4], "waves": 1, "shared_prefix_tokens": 300},
        {"name": "long", "kind": "long_context", "sizes": [1500, 3000, 999999], "depths": [0.1, 0.5, 0.9]},
        {"name": "long-raw", "kind": "long_context", "sizes": [1200], "depths": [0.5], "api": "completion"},
        {"name": "repeat", "kind": "repeat_prompt", "prompt_tokens": 800, "repeats": 3},
    ],
}


def _run(tmp_path, srv, *extra, cfg=SMALL):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    prefix = str(tmp_path / "out" / "run")
    rc = bench_http.main(["run", "--base-url", srv.url, "--scenarios", str(cfg_path), "--out", prefix,
                          "--quiet", "--label", "t", *extra])
    with open(prefix + ".json", encoding="utf-8") as f:
        res = json.load(f)
    with open(prefix + ".md", encoding="utf-8") as f:
        md = f.read()
    return rc, res, md, prefix


def test_end_to_end_run_writes_results_and_passes_checks(tmp_path):
    with FakeServer(n_ctx=8192) as srv:
        rc, res, md, prefix = _run(tmp_path, srv, "--api-key", "sekrit", "--no-think")
        assert srv.state.requests[-1]["auth"] == "Bearer sekrit"
        chat = [r for r in srv.state.requests if r["path"] == "/v1/chat/completions"]
        assert chat and all(r["body"]["chat_template_kwargs"] == {"enable_thinking": False} for r in chat)
        assert all(r["body"]["model"] == "fake-model" for r in chat)
    assert res["schema"] == "sonder.bench.http.results/1"
    checks = {(c["check"], c.get("scenario")): c for c in res["checks"]}
    assert res["verdict"] == "pass", [c for c in res["checks"] if c["status"] not in ("pass", "skip")]
    assert rc == 0
    by = {s["name"]: s for s in res["scenarios"]}
    # request counts: 3 + 3 + (1 warmup + 1 + 2 + 4) + 2 sizes * 3 depths + 1 + 3
    assert [len(by[n]["requests"]) for n in ("agent-stable", "agent-volatile-top", "fanout", "long", "long-raw", "repeat")] \
        == [3, 3, 8, 6, 1, 3]
    assert checks[("prefix_reuse", "agent-stable")]["status"] == "pass"
    assert checks[("volatile_cache_loss", None)]["status"] == "pass"
    assert checks[("recall", "long")]["detail"].startswith("6/6")
    assert checks[("recall", "long-raw")]["status"] == "pass"
    assert checks[("gpu_used", None)]["status"] == "skip"
    assert checks[("server_identity", None)]["status"] == "pass"
    assert by["long"]["skipped"] and by["long"]["skipped"][0]["size"] == 999999
    assert res["server"]["build_info"] == "b0000-fake" and res["server"]["n_ctx"] == 8192
    assert res["server"]["props"]["body"]["chat_template_chars"] == 100
    assert res["prompt_sizing"] == "tokenize"
    # long-context prompts land near their target size
    for r in by["long"]["requests"]:
        assert 0.8 * r["size"] <= r["prompt_tokens"] <= r["size"]
    # repeat prompt: turn 2+ nearly all cached
    reps = by["repeat"]["requests"]
    assert reps[1]["prefix_hit"] > 0.9 and reps[2]["prefix_hit"] > 0.9
    assert by["fanout"]["groups"]["c4"]["concurrency"] == 4
    assert by["fanout"]["groups"]["c4"]["aggregate_decode_tps"] > 0
    assert "| agent-stable | turn2 |" in md and "**Verdict: PASS**" in md
    assert "skipped" in md

    # A/B compare of the run against itself: zero deltas, exit 0
    rc = bench_http.main(["compare", prefix + ".json", prefix + ".json", "--out", str(tmp_path / "ab.md")])
    assert rc == 0
    ab = (tmp_path / "ab.md").read_text(encoding="utf-8")
    assert "| agent-stable | turn2 | TTFT p50 ms |" in ab and "0.000" in ab


def test_end_to_end_unknown_cache_warns_and_strict_fails(tmp_path):
    cfg = {"scenarios": [SMALL["scenarios"][0], SMALL["scenarios"][-1]]}
    with FakeServer(cache_report="none", props=False, tokenize=False) as srv:
        rc, res, md, _ = _run(tmp_path, srv, cfg=cfg)
        assert rc == 0 and res["verdict"] == "warn"
        c = {(x["check"], x.get("scenario")): x for x in res["checks"]}
        assert c[("prefix_reuse", "agent-stable")]["status"] == "warn"
        assert "unknown" in c[("prefix_reuse", "agent-stable")]["detail"]
        assert res["prompt_sizing"].startswith("estimate")
        assert "unknown" in md
        rc, res, _, _ = _run(tmp_path, srv, "--strict", cfg=cfg)
        assert rc == 1


def test_end_to_end_cache_disabled_warns_zero_hit(tmp_path):
    cfg = {"scenarios": [SMALL["scenarios"][0]]}
    with FakeServer(cache=False) as srv:
        rc, res, _, _ = _run(tmp_path, srv, cfg=cfg)
    c = {(x["check"], x.get("scenario")): x for x in res["checks"]}
    assert c[("prefix_reuse", "agent-stable")]["status"] == "warn"
    assert "stopped working" in c[("prefix_reuse", "agent-stable")]["detail"]


def test_end_to_end_server_errors_fail_the_run(tmp_path):
    cfg = {"scenarios": [SMALL["scenarios"][-1]]}
    with FakeServer(status_override=503) as srv:
        rc, res, _, _ = _run(tmp_path, srv, cfg=cfg)
    assert rc == 1 and res["verdict"] == "fail"
    c = {(x["check"], x.get("scenario")): x for x in res["checks"]}
    assert c[("request_count", "repeat")]["status"] == "fail"
    assert "0/3" in c[("request_count", "repeat")]["detail"]


def test_cli_usage_errors(tmp_path, capsys):
    assert bench_http.main(["run", "--base-url", "nope", "--out", str(tmp_path / "x")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"scenarios": [{"kind": "bogus"}]}))
    assert bench_http.main(["run", "--base-url", "http://127.0.0.1:1", "--scenarios", str(bad)]) == 2
    assert bench_http.main(["run", "--base-url", "http://127.0.0.1:1", "--only", "nope"]) == 2
    assert bench_http.main(["scenarios"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert {s["kind"] for s in printed["scenarios"]} == {"agent_session", "concurrency_sweep", "long_context",
                                                         "repeat_prompt"}


def test_example_config_is_valid():
    from httpbench.runner import load_config
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(here, "scenarios.example.json"))
    assert len(cfg["scenarios"]) >= 4
