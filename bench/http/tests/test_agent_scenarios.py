from httpbench.client import Target
import pytest
from httpbench.runner import BUILTIN_SCENARIOS, Runner, validate_config
from httpbench.scenarios import (alternate_reuse, max_tokens_for, pinned_thinking,
                                  scenario_headers, stall_failures)


def test_new_builtin_scenarios_are_declared_and_validate():
    names = {s["name"] for s in BUILTIN_SCENARIOS["scenarios"]}
    assert {"agent-alternate", "priority-contention", "concurrency-stall"} <= names
    assert validate_config({"scenarios": [{"kind": "agent_alternate", "headers": {"X-Test": "yes"},
                                             "priority": "interactive"}]}) == []
    assert validate_config({"scenarios": [{"kind": "agent_alternate", "headers": ["bad"]}]})


def test_headers_and_thinking_budget_preserve_explicit_override():
    assert scenario_headers({"headers": {"X-Sonder-Agent-Id": "a"}}, {"X-Default": "d"}) == {
        "X-Default": "d", "X-Sonder-Agent-Id": "a"}
    assert pinned_thinking(["thinking pinned by server policy"])
    assert not pinned_thinking(["enable_thinking=on was overridden by server pin (off)"])
    assert max_tokens_for({"kind": "agent_session"}, {"max_tokens": 64}, ["thinking pinned"]) == 1500
    assert max_tokens_for({"kind": "agent_session", "max_tokens": 32, "_max_tokens_explicit": True},
                          {"max_tokens": 64}, ["thinking pinned"]) == 32


def test_alternate_reuse_uses_actual_measured_prompt_length():
    rows = [{"agent": "A", "turn": 2, "prompt_tokens": 100, "cached_tokens": 91},
            {"agent": "A", "turn": 3, "prompt_tokens": 200, "cached_tokens": 180},
            {"agent": "B", "turn": 2, "prompt_tokens": 1000, "cached_tokens": 0}]
    result = alternate_reuse(rows)
    assert result["pass"] and result["ratios"] == [0.91, 0.9]
    assert not alternate_reuse([{**rows[0], "cached_tokens": None}])["pass"]


def test_alternate_runner_preserves_abc_order_and_reuses_fixed_prefixes():
    from fake_server import FakeServer

    with FakeServer(keyed_cache=True, delay=0) as server:
        runner = Runner(Target(server.url), defaults={"max_tokens": 4})
        result = runner.run({"defaults": {"max_tokens": 4}, "scenarios": [{
            "name": "alt", "kind": "agent_alternate", "turns": 3,
            "agent_tokens": {"A": 2000, "B": 1000, "C": 3000}}]})
        rows = result["scenarios"][0]["requests"]
        assert [(r["agent"], r["turn"]) for r in rows] == [
            ("A", 1), ("B", 1), ("C", 1), ("A", 2), ("B", 2), ("C", 2),
            ("A", 3), ("B", 3), ("C", 3)]
        assert result["scenarios"][0]["alternate_reuse"]["pass"]
        assert {r["body"].get("prompt_cache_key") for r in server.state.requests if "body" in r and r["body"].get("prompt_cache_key")} == {
            f"{runner.run_id}:alt:agent-A", f"{runner.run_id}:alt:agent-B", f"{runner.run_id}:alt:agent-C"}


def test_alternate_runner_fails_when_cache_is_disabled():
    from fake_server import FakeServer

    with FakeServer(cache=False, delay=0) as server:
        runner = Runner(Target(server.url), defaults={"max_tokens": 4})
        result = runner.run({"defaults": {"max_tokens": 4}, "scenarios": [{
            "kind": "agent_alternate", "turns": 2,
            "agent_tokens": {"A": 20, "B": 20, "C": 20}}]})
        assert result["scenarios"][0]["alternate_reuse"]["pass"] is False


def test_stall_detection_catches_503_and_canary_leaks():
    rows = [{"status": 200, "marker": "A", "_content": "ok", "e2e_s": 251.0},
            {"status": 503, "marker": "B", "_content": "A"},
            {"status": 200, "marker": "C", "_content": "ok"}]
    failures = stall_failures(rows, 251.0)
    assert any("wall" in x for x in failures)
    assert any("503" in x for x in failures)
    assert any("marker leak" in x for x in failures)


def test_stall_uses_request_wall_not_prompt_preparation_time():
    rows = [{"ok": True, "status": 200, "e2e_s": 1, "marker": f"M{i}"} for i in range(3)]
    assert stall_failures(rows, 999) == []
    rows[0]["e2e_s"] = 251
    assert any("wall" in x for x in stall_failures(rows, 1))


def test_pinned_thinking_first_request_retries_and_preserves_reasoning_history():
    from fake_server import FakeServer
    with FakeServer(pinned_thinking=True, reasoning_only=True, delay=0) as server:
        runner = Runner(Target(server.url), defaults={"max_tokens": 4}, log=lambda _: None)
        result = runner.run({"scenarios": [{"kind": "agent_session", "turns": 2,
                                             "system_tokens": 400, "tools": 0, "turn_tokens": 20}]})
        posts = [r["body"] for r in server.state.requests if r["path"] == "/v1/chat/completions"]
        assert [p["max_tokens"] for p in posts] == [4, 1500, 1500]
        history = [m for m in posts[-1]["messages"] if m["role"] == "assistant"]
        assert history[0]["reasoning_content"].startswith("Next I would")
        assert history[0]["content"] == ""
        assert result["scenarios"][0]["requests"][0]["thinking_retry"]["initial"]["completion_tokens"] == 4


def test_recall_reads_reasoning_without_overriding_explicit_cap():
    from fake_server import FakeServer
    with FakeServer(pinned_thinking=True, reasoning_only=True, delay=0) as server:
        result = Runner(Target(server.url), log=lambda _: None).run({"scenarios": [{
            "kind": "long_context", "sizes": [500], "depths": [0.5], "max_tokens": 20,
            "_max_tokens_explicit": True}]})
        row = result["scenarios"][0]["requests"][0]
        assert row["recall_correct"] and row["found_in_reasoning"]
        assert "thinking_retry" not in row
        assert [r["body"]["max_tokens"] for r in server.state.requests
                if r["path"] == "/v1/chat/completions"] == [20]


def test_priority_contention_has_distinct_keys_priorities_and_cold_nonce():
    from fake_server import FakeServer
    with FakeServer(delay=0.01) as server:
        result = Runner(Target(server.url), log=lambda _: None).run({"scenarios": [{
            "kind": "priority_contention", "background_tokens": 500, "interactive_tokens": 200,
            "max_tokens": 16}]})
        posts = [r["body"] for r in server.state.requests if r["path"] == "/v1/chat/completions"]
        assert [p["priority"] for p in posts] == ["background", "interactive"]
        assert len({p["prompt_cache_key"] for p in posts}) == 2
        scenario = result["scenarios"][0]
        assert scenario["overlapped"] and scenario["interactive_ttft_s"] > 0
        assert scenario["cold_nonce"] in posts[0]["messages"][0]["content"]


@pytest.mark.parametrize("leak,status", [(None, None), ("reasoning", None), ("content", None), (None, 503)])
def test_stall_fake_server_checks_content_reasoning_and_503(monkeypatch, leak, status):
    import fake_server
    def answer(prompt, last):
        marker = last.split("marker=", 1)[1].split(";", 1)[0]
        return marker if not leak else "STALL_CANARY_0 STALL_CANARY_1 STALL_CANARY_2"
    monkeypatch.setattr(fake_server, "answer_for", answer)
    with fake_server.FakeServer(delay=0, reasoning_only=leak == "reasoning", status_override=status) as server:
        result = Runner(Target(server.url), log=lambda _: None).run({"scenarios": [{
            "kind": "concurrency_stall", "prefix_tokens": 100, "max_tokens": 20}]})
        scenario = result["scenarios"][0]
        assert len(scenario["requests"]) == 3
        assert len({r["prompt_cache_key"] for r in scenario["requests"]}) == 3
        assert (result["verdict"] == "fail") == (bool(leak) or status == 503)


def test_legacy_request_defaults_and_new_controls_for_both_apis():
    runner = Runner(Target("http://127.0.0.1:1"))
    assert "max_tokens" not in runner.body("chat", {}, prompt="x")[1]
    assert runner.body("completion", {}, prompt="x")[1]["n_predict"] == 64
    for api in ("chat", "completion"):
        body = runner.body(api, {"prompt_cache_key": "key", "priority": "background"}, prompt="x")[1]
        assert body["prompt_cache_key"] == "key" and body["priority"] == "background"
