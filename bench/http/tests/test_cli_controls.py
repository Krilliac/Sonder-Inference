"""Public command-line controls for reproducible multi-agent HTTP runs."""

import json

import pytest

import bench_http
from fake_server import FakeServer


def test_cli_accepts_repeatable_headers_and_observation_paths():
    args = bench_http.build_parser().parse_args([
        "run", "--base-url", "http://127.0.0.1:1", "--header", "X-Run=a=b",
        "--header", "X-Agent=B", "--metrics-url", "http://127.0.0.1:2/metrics",
        "--child-log", "child.log",
    ])
    assert args.header == [("X-Run", "a=b"), ("X-Agent", "B")]
    assert args.metrics_url.endswith("/metrics") and args.child_log == "child.log"


@pytest.mark.parametrize("header", ["missing-separator", "=value", "X Bad=value", "X=one\r\ntwo"])
def test_cli_rejects_invalid_header_before_sending(header):
    with pytest.raises(SystemExit) as exc:
        bench_http.build_parser().parse_args(["run", "--base-url", "http://127.0.0.1:1", "--header", header])
    assert exc.value.code == 2


def test_cli_controls_reach_fake_and_sensitive_values_are_not_saved(tmp_path, monkeypatch):
    config = {"scenarios": [{"kind": "repeat_prompt", "repeats": 1, "prompt_tokens": 100,
                              "headers": {"X-Agent": "scenario", "X-Secret": "scenario-secret"},
                              "prompt_cache_key": "agent-one", "priority": "interactive"}]}
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    prefix = tmp_path / "result"
    with FakeServer() as srv:
        argv = ["run", "--base-url", srv.url, "--scenarios", str(cfg), "--out", str(prefix),
                "--header", "X-Agent=global", "--header", "X-Secret=global-secret",
                "--api-key", "test-secret", "--quiet"]
        monkeypatch.setattr("sys.argv", ["bench_http.py", *argv])
        assert bench_http.main(argv) == 0
        request = [r for r in srv.state.requests if r["path"] == "/v1/chat/completions"][0]
        assert request["headers"]["X-Agent"] == "scenario"
        assert request["headers"]["X-Secret"] == "scenario-secret"
        assert request["body"]["prompt_cache_key"] == "agent-one"
        assert request["body"]["priority"] == "interactive"
        assert request["auth"] == "Bearer test-secret"
    saved = prefix.with_suffix(".json").read_text(encoding="utf-8")
    assert all(secret not in saved for secret in ("scenario-secret", "global-secret", "test-secret"))


def test_explicit_cli_max_tokens_is_marked_for_thinking_override():
    args = bench_http.build_parser().parse_args([
        "run", "--base-url", "http://127.0.0.1:1", "--max-tokens", "17",
    ])
    assert args.max_tokens == 17


def test_metrics_are_scraped_around_each_request_and_compared(tmp_path, capsys):
    cfg = tmp_path / "metrics-config.json"
    cfg.write_text(json.dumps({"scenarios": [{"kind": "repeat_prompt", "repeats": 2,
                                             "prompt_tokens": 100}]}), encoding="utf-8")
    prefix = tmp_path / "metrics-result"
    with FakeServer() as srv:
        rc = bench_http.main(["run", "--base-url", srv.url, "--scenarios", str(cfg),
                              "--metrics-url", srv.url + "/metrics", "--api-key", "probe-secret",
                              "--out", str(prefix), "--quiet"])
        assert rc == 0
        scrapes = [r for r in srv.state.get_requests if r["path"] == "/metrics"]
        assert len(scrapes) >= 4
        assert all("Authorization" not in r["headers"] for r in scrapes)
    result = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    requests = result["scenarios"][0]["requests"]
    assert len(requests) == 2
    assert all(r["metrics"]["accepted_per_position"] == {"0": 3.0, "1": 2.0} for r in requests)
    capsys.readouterr()
    assert bench_http.main(["compare", str(prefix) + ".json", str(prefix) + ".json"]) == 0
    assert "accepted position 0" in capsys.readouterr().out


def test_unavailable_observations_warn_instead_of_reporting_zero(tmp_path):
    cfg = tmp_path / "missing-config.json"
    cfg.write_text(json.dumps({"scenarios": [{"kind": "repeat_prompt", "repeats": 1,
                                             "prompt_tokens": 100}]}), encoding="utf-8")
    prefix = tmp_path / "missing-result"
    with FakeServer() as srv:
        rc = bench_http.main(["run", "--base-url", srv.url, "--scenarios", str(cfg),
                              "--metrics-url", srv.url + "/absent", "--child-log", str(tmp_path / "absent.log"),
                              "--out", str(prefix), "--quiet", "--strict"])
    result = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert rc == 1 and result["verdict"] == "warn"
    checks = {c["check"]: c for c in result["checks"]}
    assert checks["metrics_available"]["status"] == "warn"
    assert checks["child_log_available"]["status"] == "warn"


def test_child_log_run_counts_exclude_old_events_and_compare(tmp_path, monkeypatch, capsys):
    import fake_server
    log = tmp_path / "child.log"
    log.write_text("making room\n" * 10, encoding="utf-8")
    original = fake_server.answer_for
    def logged_answer(prompt, last):
        with log.open("a", encoding="utf-8") as stream:
            stream.write("making room\nselected slot by id\n")
        return original(prompt, last)
    monkeypatch.setattr(fake_server, "answer_for", logged_answer)
    cfg = tmp_path / "log-config.json"
    cfg.write_text(json.dumps({"scenarios": [{"kind": "repeat_prompt", "repeats": 2,
                                             "prompt_tokens": 100}]}), encoding="utf-8")
    prefix = tmp_path / "log-result"
    with FakeServer() as srv:
        assert bench_http.main(["run", "--base-url", srv.url, "--scenarios", str(cfg),
                                "--child-log", str(log), "--out", str(prefix), "--quiet"]) == 0
    result = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert result["child_log"]["counters"]["making_room"] == 2
    assert result["scenarios"][0]["child_log"]["counters"]["selected_slot_by_id"] == 2
    capsys.readouterr()
    assert bench_http.main(["compare", str(prefix) + ".json", str(prefix) + ".json"]) == 0
    assert "child log making_room" in capsys.readouterr().out


def test_redaction_handles_abbreviated_and_equals_options():
    saved = bench_http._redact_argv(["run", "--head=X-Test=secret1", "--api-k", "secret2", "--label", "a"])
    assert saved == ["run", "--head=<redacted>", "--api-k", "<redacted>", "--label", "a"]


def test_metrics_url_rejects_non_http_without_running_a_request():
    assert bench_http.main(["run", "--base-url", "http://127.0.0.1:1", "--metrics-url", "file:///tmp/file"]) == 2
