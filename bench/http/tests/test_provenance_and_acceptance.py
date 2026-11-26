"""Unit and fake-server integration tests for reproducible provenance, separated metrics,
and acceptance-matrix scenarios in bench/http."""

from __future__ import annotations

import hashlib
import json

import pytest

import bench_http
from fake_server import FakeServer
from httpbench.client import Target
from httpbench.metrics import (Chunk, StreamTrace, classify_cache_state, derive,
                               group_summary, split_cold_warm_summary)
from httpbench.observation import (PROVENANCE_SCHEMA, REQUIRED_PROVENANCE_FIELDS,
                                   build_provenance, corpus_revision_for, sha256_file,
                                   validate_provenance, validate_sha256_hex)
from httpbench.report import compare
from httpbench.runner import ACCEPTANCE_MATRIX_SCENARIOS, BUILTIN_SCENARIOS, Runner, validate_config
from httpbench.scenarios import CORPUS_TAGS, overlap_count, trailing_summary, warm_tool_return_reuse


# ---------------------------------------------------------------------- metrics separation

def test_derive_separates_committed_vs_proposed_and_client_vs_backend_ms():
    trace = StreamTrace(
        status=200,
        done=True,
        end_t=0.500,
        chunks=[Chunk(0.120, "a "), Chunk(0.220, "b "), Chunk(0.340, "c "), Chunk(0.500, "d ")],
        usage={"prompt_tokens": 100, "completion_tokens": 4},
        timings={
            "prompt_n": 20,
            "cache_n": 80,
            "prompt_ms": 14.5,
            "predicted_ms": 62.5,
            "prompt_per_second": 1379.3,
            "predicted_per_second": 64.0,
            "draft_n": 12,
            "draft_n_accepted": 3,
        },
    )
    m = derive(trace)
    # Committed target tokens are strictly separated from proposed/rejected draft tokens.
    assert m["committed_output_tokens"] == 4
    assert m["proposed_draft_tokens"] == 12
    assert m["accepted_draft_tokens"] == 3
    assert m["rejected_draft_tokens"] == 9
    assert m["committed_output_tokens"] != m["proposed_draft_tokens"]

    # Backend compute (ms) is strictly separated from client-observed wire latencies (ms).
    assert m["backend_prefill_ms"] == pytest.approx(14.5)
    assert m["backend_decode_ms"] == pytest.approx(62.5)
    assert m["backend_compute_ms"] == pytest.approx(77.0)
    assert m["client_observed_ms"]["ttft_ms"] == pytest.approx(120.0)
    assert m["client_observed_ms"]["e2e_ms"] == pytest.approx(500.0)
    assert m["client_observed_ms"]["itl_ms"]["n"] == 3
    assert m["client_observed_ms"]["itl_ms"]["p50"] == pytest.approx(120.0)
    assert m["client_observed_ms"]["e2e_ms"] > m["backend_compute_ms"]


def test_derive_leaves_missing_draft_and_partial_backend_timings_as_unknown():
    trace = StreamTrace(
        status=200,
        done=True,
        end_t=0.250,
        chunks=[Chunk(0.050, "x"), Chunk(0.150, "y")],
        usage={"prompt_tokens": 30, "completion_tokens": 2},
        timings={"prompt_ms": 10.0},  # predicted_ms absent -> backend_compute_ms must NOT be a floor
    )
    m = derive(trace)
    assert m["committed_output_tokens"] == 2
    assert m["proposed_draft_tokens"] is None
    assert m["accepted_draft_tokens"] is None
    assert m["rejected_draft_tokens"] is None
    assert m["backend_prefill_ms"] == pytest.approx(10.0)
    assert m["backend_decode_ms"] is None
    assert m["backend_compute_ms"] is None

    summary = group_summary([m])
    assert summary["committed_output_tokens_total"] == 2
    assert summary["proposed_draft_tokens_total"] is None
    assert summary["backend_compute_ms"]["total_ms"] == {"n": 0}
    assert summary["backend_compute_ms"]["prefill_ms"]["p50"] == pytest.approx(10.0)
    assert summary["client_observed_ms"]["ttft_ms"]["p50"] == pytest.approx(50.0)
    assert summary["client_observed_ms"]["itl_ms"]["p50"] == pytest.approx(100.0)


def test_cold_vs_warm_classification_and_independent_aggregation():
    cold = {
        "ok": True,
        "cache_state": "cold",
        "ttft_s": 0.400,
        "e2e_s": 0.800,
        "itl_s": [0.020, 0.040],
        "committed_output_tokens": 10,
        "proposed_draft_tokens": 16,
        "accepted_draft_tokens": 8,
        "backend_prefill_ms": 120.0,
        "backend_decode_ms": 180.0,
        "backend_compute_ms": 300.0,
        "prefix_hit": 0.0,
    }
    warm = {
        "ok": True,
        "cache_state": "warm",
        "ttft_s": 0.050,
        "e2e_s": 0.300,
        "itl_s": [0.010, 0.015],
        "committed_output_tokens": 12,
        "proposed_draft_tokens": 20,
        "accepted_draft_tokens": 15,
        "backend_prefill_ms": 8.0,
        "backend_decode_ms": 140.0,
        "backend_compute_ms": 148.0,
        "prefix_hit": 0.92,
    }
    unknown = {"ok": True, "ttft_s": 0.1, "e2e_s": 0.2, "itl_s": [], "completion_tokens": 3}

    assert classify_cache_state({"turn": 1}) == "cold"
    assert classify_cache_state({"turn": 2}) == "warm"
    assert classify_cache_state({"repeat": 1}) == "cold"
    assert classify_cache_state({"repeat": 3}) == "warm"
    assert classify_cache_state({"cached_tokens": 0}) == "cold"
    assert classify_cache_state({"cached_tokens": 128}) == "warm"
    assert classify_cache_state(unknown) == "unclassified"

    split = split_cold_warm_summary([cold, warm, unknown])
    assert split["cold"]["ok"] == 1
    assert split["warm"]["ok"] == 1
    assert split["unclassified"]["ok"] == 1
    assert split["cold"]["client_observed_ms"]["ttft_ms"]["p50"] == pytest.approx(400.0)
    assert split["warm"]["client_observed_ms"]["ttft_ms"]["p50"] == pytest.approx(50.0)
    assert split["cold"]["backend_compute_ms"]["total_ms"]["p50"] == pytest.approx(300.0)
    assert split["warm"]["backend_compute_ms"]["total_ms"]["p50"] == pytest.approx(148.0)
    assert split["cold"]["committed_output_tokens_total"] == 10
    assert split["cold"]["proposed_draft_tokens_total"] == 16
    assert split["warm"]["committed_output_tokens_total"] == 12
    assert split["warm"]["proposed_draft_tokens_total"] == 20
    assert split["cold"]["prefix_hit"]["mean"] == pytest.approx(0.0)
    assert split["warm"]["prefix_hit"]["mean"] == pytest.approx(0.92)


# ---------------------------------------------------------------------- provenance schema

def test_provenance_schema_validation_and_file_digests(tmp_path):
    model_file = tmp_path / "model.gguf"
    draft_file = tmp_path / "draft.gguf"
    exe_file = tmp_path / "llama-server.bin"
    model_file.write_bytes(b"fake-target-gguf-payload")
    draft_file.write_bytes(b"fake-draft-gguf-payload")
    exe_file.write_bytes(b"fake-upstream-binary")

    expected_model_sha = hashlib.sha256(b"fake-target-gguf-payload").hexdigest()
    expected_draft_sha = hashlib.sha256(b"fake-draft-gguf-payload").hexdigest()
    expected_exe_sha = "sha256:" + hashlib.sha256(b"fake-upstream-binary").hexdigest()
    assert sha256_file(model_file) == expected_model_sha
    assert validate_sha256_hex(expected_model_sha.upper()) == expected_model_sha
    with pytest.raises(ValueError):
        validate_sha256_hex("not-a-sha256")

    cfg1 = {"scenarios": [{"name": "s", "kind": "single_request", "corpus_tags": ["prose"]}]}
    cfg2 = {"scenarios": [{"name": "s", "kind": "single_request", "corpus_tags": ["code"]}]}
    assert corpus_revision_for(cfg1) == corpus_revision_for(dict(cfg1))
    assert corpus_revision_for(cfg1) != corpus_revision_for(cfg2)

    minimal_results = {
        "target": {"base_url": "http://127.0.0.1:8080/v1", "model": "fake-model", "defaults": {}, "extra_body": {}},
        "server": {"n_ctx": 4096, "ctx_source": "server", "build_info": "b11195"},
        "prompt_sizing": "tokenize",
        "scenarios": [{
            "name": "s",
            "expected_requests": 2,
            "requests": [
                {"ok": True, "cache_state": "cold", "model": "fake-model", "proposed_draft_tokens": 10, "cached_tokens": 0},
                {"ok": True, "cache_state": "warm", "model": "fake-model", "proposed_draft_tokens": 10, "cached_tokens": 50},
            ],
        }],
    }
    prov = build_provenance(
        minimal_results,
        cfg1,
        checks=[{"check": "request_count", "status": "pass", "detail": "2/2"}],
        verdict_value="pass",
        overrides={
            "source_commit": "8af151105327bba0fcc581f5e8f12e04acf554a5",
            "upstream_exe": str(exe_file),
            "model_file": str(model_file),
            "draft_file": str(draft_file),
            "resolved_settings": {"spec_type": "suffix"},
            "effective_settings": {"active_spec": "suffix"},
        },
    )
    assert prov["schema"] == PROVENANCE_SCHEMA
    assert set(REQUIRED_PROVENANCE_FIELDS) <= set(prov)
    assert prov["model_sha256"] == expected_model_sha
    assert prov["draft_sha256"] == expected_draft_sha
    assert prov["upstream_revision_or_digest"] == expected_exe_sha
    assert prov["resolved_settings"]["spec_type"] == "suffix"
    assert prov["effective_settings"]["active_spec"] == "suffix"
    assert prov["effective_settings"]["draft_observed_active"] is True
    assert prov["effective_settings"]["prefix_cache_observed_active"] is True
    assert prov["sample_count"] == {
        "expected": 2, "completed": 2, "ok": 2, "failed": 0, "cold": 1, "warm": 1, "unclassified": 0
    }
    assert prov["validation_outcome"]["provenance_complete"] is True
    assert validate_provenance(prov, strict=True) == []

    # Mutations caught by strict validator
    bad = dict(prov, model_sha256="unknown")
    assert validate_provenance(bad, strict=False) == []
    assert validate_provenance(bad, strict=True)
    bad_counts = dict(prov, sample_count={**prov["sample_count"], "cold": 2})
    assert validate_provenance(bad_counts)
    bad_bool = dict(prov, sample_count={**prov["sample_count"], "ok": True})
    assert validate_provenance(bad_bool)
    missing_field = {k: v for k, v in prov.items() if k != "timing_definition"}
    assert validate_provenance(missing_field)


# ---------------------------------------------------------------------- acceptance-matrix scenarios

def test_acceptance_matrix_scenarios_validate_and_reject_bad_configs():
    assert validate_config(ACCEPTANCE_MATRIX_SCENARIOS) == []
    assert validate_config(BUILTIN_SCENARIOS) == []
    names = {s["name"] for s in ACCEPTANCE_MATRIX_SCENARIOS["scenarios"]}
    assert names == {
        "single-request",
        "simultaneous-siblings",
        "staggered-arrivals",
        "long-trailing",
        "warm-tool-return",
    }

    bad_cfg = {
        "scenarios": [
            {"name": "b1", "kind": "single_request", "corpus_tags": ["invalid_tag"], "warm_repeat": "yes"},
            {"name": "b2", "kind": "simultaneous_siblings", "siblings": 1, "corpus_tag": "nope"},
            {"name": "b3", "kind": "staggered_arrivals", "arrivals": 1, "stagger_ms": -5},
            {"name": "b4", "kind": "long_trailing", "short_peers": 0, "short_max_tokens": 0},
            {"name": "b5", "kind": "warm_tool_return", "turns": 1},
        ]
    }
    errs = validate_config(bad_cfg)
    assert len(errs) >= 9


def test_acceptance_matrix_end_to_end_against_fake_server(tmp_path, capsys):
    model_file = tmp_path / "target.gguf"
    model_file.write_bytes(b"target-model-bytes")
    model_sha = hashlib.sha256(b"target-model-bytes").hexdigest()

    cfg = {
        "schema": "sonder.bench.http.scenarios/1",
        "defaults": {"max_tokens": 12, "temperature": 0, "seed": 42},
        "scenarios": [
            {"name": "single-request", "kind": "single_request",
             "corpus_tags": list(CORPUS_TAGS), "prompt_tokens": 120, "warm_repeat": True, "max_tokens": 8},
            {"name": "simultaneous-siblings", "kind": "simultaneous_siblings",
             "siblings": 3, "shared_prefix_tokens": 200, "corpus_tag": "code", "max_tokens": 8},
            {"name": "staggered-arrivals", "kind": "staggered_arrivals",
             "arrivals": 3, "stagger_ms": 12.0, "prompt_tokens": 120, "corpus_tag": "prose", "max_tokens": 8},
            {"name": "long-trailing", "kind": "long_trailing",
             "short_peers": 2, "short_max_tokens": 3, "long_max_tokens": 12,
             "shared_prefix_tokens": 120, "corpus_tag": "code"},
            {"name": "warm-tool-return", "kind": "warm_tool_return",
             "turns": 3, "system_tokens": 300, "tools": 4, "turn_tokens": 40,
             "corpus_tag": "tool_json", "max_tokens": 8},
        ],
    }
    cfg_path = tmp_path / "acceptance.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    prefix = tmp_path / "out" / "accept"

    with FakeServer(draft=True, keyed_cache=True, delay=0.008) as srv:
        rc = bench_http.main([
            "run",
            "--base-url", srv.url,
            "--scenarios", str(cfg_path),
            "--label", "accept-arm",
            "--source-commit", "8af151105327bba0fcc581f5e8f12e04acf554a5",
            "--model-file", str(model_file),
            "--draft-sha256", "none",
            "--resolved-settings", '{"spec_type":"suffix","slots":4}',
            "--require-provenance",
            "--out", str(prefix),
            "--quiet",
        ])

    assert rc == 0
    res = json.loads((tmp_path / "out" / "accept.json").read_text(encoding="utf-8"))
    md = (tmp_path / "out" / "accept.md").read_text(encoding="utf-8")
    assert res["verdict"] == "pass", [c for c in res["checks"] if c["status"] not in ("pass", "skip")]

    # Provenance checks
    prov = res["provenance"]
    assert validate_provenance(prov, strict=True) == []
    assert prov["source_commit"] == "8af151105327bba0fcc581f5e8f12e04acf554a5"
    assert prov["upstream_revision_or_digest"] == "b0000-fake"
    assert prov["model_sha256"] == model_sha
    assert prov["draft_sha256"] == "none"
    assert prov["resolved_settings"]["spec_type"] == "suffix"
    assert prov["effective_settings"]["draft_observed_active"] is True
    assert prov["effective_settings"]["prefix_cache_observed_active"] is True
    assert prov["sample_count"]["ok"] == prov["sample_count"]["expected"]
    assert prov["sample_count"]["cold"] > 0 and prov["sample_count"]["warm"] > 0
    assert prov["sample_count"]["unclassified"] == 0
    assert prov["validation_outcome"]["verdict"] == "pass"
    assert prov["validation_outcome"]["provenance_complete"] is True

    # Separated run-level cold vs warm and corpus tags
    assert res["cold_vs_warm"]["cold"]["ok"] == prov["sample_count"]["cold"]
    assert res["cold_vs_warm"]["warm"]["ok"] == prov["sample_count"]["warm"]
    assert res["cold_vs_warm"]["cold"]["committed_output_tokens_total"] > 0
    assert res["cold_vs_warm"]["cold"]["proposed_draft_tokens_total"] > 0
    assert res["cold_vs_warm"]["cold"]["backend_compute_ms"]["total_ms"]["n"] > 0
    assert res["cold_vs_warm"]["cold"]["client_observed_ms"]["ttft_ms"]["p95"] is not None
    assert set(res["by_corpus_tag"]) == set(CORPUS_TAGS)

    by = {s["name"]: s for s in res["scenarios"]}
    # 1. single-request: 4 corpus tags x (cold + warm) = 8 requests
    sr = by["single-request"]
    assert len(sr["requests"]) == 8
    assert sr["cold_vs_warm"]["cold"]["ok"] == 4
    assert sr["cold_vs_warm"]["warm"]["ok"] == 4
    assert sr["cold_vs_warm"]["warm"]["prefix_hit"]["mean"] > 0.8
    assert set(sr["by_corpus_tag"]) == set(CORPUS_TAGS)

    # 2. simultaneous-siblings: 1 cold warmup + 3 concurrent warm siblings
    sib = by["simultaneous-siblings"]
    assert len(sib["requests"]) == 4
    assert sib["cold_vs_warm"]["cold"]["ok"] == 1
    assert sib["cold_vs_warm"]["warm"]["ok"] == 3
    assert sib["overlapped_siblings"] >= 1
    assert sib["sibling_failures"] == []

    # 3. staggered-arrivals: 3 staggered arrivals with monotonic intended delays
    stag = by["staggered-arrivals"]
    assert len(stag["requests"]) == 3
    assert [r["arrival_index"] for r in stag["requests"]] == [0, 1, 2]
    assert [r["intended_stagger_ms"] for r in stag["requests"]] == [0.0, 12.0, 24.0]
    assert stag["requests"][0]["t_start_s"] <= stag["requests"][1]["t_start_s"] <= stag["requests"][2]["t_start_s"]
    assert stag["overlapped_pairs"] >= 1

    # 4. long-trailing: 2 short peers finish strictly before 1 long trailing request
    lt = by["long-trailing"]
    assert len(lt["requests"]) == 3
    assert lt["trailing"]["peers_ok"] == 2
    assert lt["trailing"]["trailing_ok"] is True
    assert lt["trailing"]["peers_finished_before_trailing"] is True
    assert lt["trailing"]["solo_tail_s"] > 0.0

    # 5. warm-tool-return: 1 cold turn + 2 warm tool-return turns with prefix reuse
    wtr = by["warm-tool-return"]
    assert len(wtr["requests"]) == 3
    assert wtr["cold_vs_warm"]["cold"]["ok"] == 1
    assert wtr["cold_vs_warm"]["warm"]["ok"] == 2
    assert wtr["warm_tool_reuse"]["pass"] is True
    assert all(ratio > 0.5 for ratio in wtr["warm_tool_reuse"]["ratios"])

    # Markdown report & A/B compare surface provenance, cold/warm, and separated metrics
    assert "## Provenance" in md
    assert "## Cold vs Warm Summary" in md
    assert model_sha in md

    diff_md, rows = compare(res, res)
    assert "Provenance A:" in diff_md
    metrics_seen = {r["metric"] for r in rows}
    assert {"backend compute p50 ms", "committed output tokens", "proposed draft tokens"} <= metrics_seen
    assert any(r["group"] == "(cold)" for r in rows)
    assert any(r["group"] == "(warm)" for r in rows)

    # CLI scenarios --preset acceptance prints exactly the 5 acceptance-matrix scenarios
    capsys.readouterr()
    assert bench_http.main(["scenarios", "--preset", "acceptance"]) == 0
    preset = json.loads(capsys.readouterr().out)
    assert [s["kind"] for s in preset["scenarios"]] == [
        "single_request",
        "simultaneous_siblings",
        "staggered_arrivals",
        "long_trailing",
        "warm_tool_return",
    ]


def test_require_provenance_fails_when_digests_are_unknown(tmp_path):
    cfg = {"scenarios": [{"name": "sr", "kind": "single_request", "corpus_tags": ["prose"],
                          "prompt_tokens": 80, "warm_repeat": False, "max_tokens": 6}]}
    cfg_path = tmp_path / "sr.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    prefix = tmp_path / "strict_fail"
    with FakeServer(delay=0) as srv:
        rc = bench_http.main([
            "run", "--base-url", srv.url, "--scenarios", str(cfg_path),
            "--require-provenance", "--out", str(prefix), "--quiet",
        ])
    assert rc == 1
    res = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert res["verdict"] == "fail"
    checks = {c["check"]: c for c in res["checks"]}
    assert checks["provenance_complete"]["status"] == "fail"
    assert "model_sha256" in checks["provenance_complete"]["detail"]
    assert res["provenance"]["validation_outcome"]["provenance_complete"] is False


def test_warm_tool_return_warns_when_cache_unknown_or_disabled():
    with FakeServer(cache_report="none", delay=0) as srv:
        res = Runner(Target(srv.url), log=lambda _: None).run({
            "scenarios": [{"name": "wtr", "kind": "warm_tool_return", "turns": 3,
                           "system_tokens": 200, "tools": 2, "turn_tokens": 20, "max_tokens": 6}]
        })
    checks = {c["check"]: c for c in res["checks"]}
    assert checks["warm_tool_return_reuse"]["status"] == "warn"
    assert "unknown" in checks["warm_tool_return_reuse"]["detail"]

    with FakeServer(cache=False, delay=0) as srv:
        res2 = Runner(Target(srv.url), log=lambda _: None).run({
            "scenarios": [{"name": "wtr", "kind": "warm_tool_return", "turns": 3,
                           "system_tokens": 200, "tools": 2, "turn_tokens": 20, "max_tokens": 6}]
        })
    checks2 = {c["check"]: c for c in res2["checks"]}
    assert checks2["warm_tool_return_reuse"]["status"] == "warn"


def test_scenario_pure_helpers_overlap_trailing_and_tool_reuse():
    assert overlap_count([{"t_start_s": 0.0, "e2e_s": 1.0}, {"t_start_s": 0.5, "e2e_s": 1.0}]) == 1
    assert overlap_count([{"t_start_s": 0.0, "e2e_s": 0.5}, {"t_start_s": 0.5, "e2e_s": 0.5}]) == 0
    tr = trailing_summary(
        [{"ok": True, "t_start_s": 0.0, "e2e_s": 0.2}, {"ok": True, "t_start_s": 0.0, "e2e_s": 0.3}],
        {"ok": True, "t_start_s": 0.0, "e2e_s": 1.0},
    )
    assert tr["peers_finished_before_trailing"] is True
    assert tr["solo_tail_s"] == pytest.approx(0.7)
    reuse = warm_tool_return_reuse([
        {"ok": True, "turn": 1, "cache_state": "cold", "prompt_tokens": 100, "cached_tokens": 0},
        {"ok": True, "turn": 2, "cache_state": "warm", "prompt_tokens": 120, "cached_tokens": 90},
    ])
    assert reuse["pass"] is True and reuse["ratios"] == [pytest.approx(0.75)]


def test_cli_rejects_bad_provenance_flags(tmp_path):
    assert bench_http.main([
        "run", "--base-url", "http://127.0.0.1:1", "--model-sha256", "bad-hex",
    ]) == 2
    assert bench_http.main([
        "run", "--base-url", "http://127.0.0.1:1", "--model-file", str(tmp_path / "missing.gguf"),
    ]) == 2
    assert bench_http.main([
        "run", "--base-url", "http://127.0.0.1:1", "--resolved-settings", "[1,2,3]",
    ]) == 2
