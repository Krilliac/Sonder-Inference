import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from httpbench.observation import ChildLogProbe, MetricsProbe, metrics_delta
from httpbench.report import compare, render_markdown


class _MetricsHandler(BaseHTTPRequestHandler):
    body = b""

    def log_message(self, *_args):
        pass

    def do_GET(self):
        data = type(self).body
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _metrics_server(body: str):
    _MetricsHandler.body = body.encode()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/metrics"


def test_metrics_probe_parses_labels_scientific_values_and_full_body():
    body = "\n".join([
        '# HELP spec_decode_accepted_tokens_per_pos_total x',
        'spec_decode_accepted_tokens_per_pos_total{position="0",slot="0"} 4',
        'spec_decode_accepted_tokens_per_pos_total{slot="0",position="1"} 1e1',
        'spec_decode_accepted_tokens_per_pos_total{position="2"} 0.5',
        'unrelated_metric 99',
    ])
    server, url = _metrics_server(body)
    try:
        result = MetricsProbe(url).snapshot()
        assert result["status"] == 200
        assert result["accepted_per_position"] == {"0": 4.0, "1": 10.0, "2": 0.5}
    finally:
        server.shutdown()
        server.server_close()


def test_metrics_delta_reports_new_positions_and_resets_without_negative_counts():
    assert metrics_delta(
        {"accepted_per_position": {"0": 9, "1": 3}},
        {"accepted_per_position": {"0": 12, "1": 1, "2": 4}},
    ) == {"accepted_per_position": {"0": 3.0}, "errors": [], "resets": ["1"], "unknown": ["2"]}
    assert metrics_delta({"error": "timeout"}, {"accepted_per_position": {"0": 1}})["errors"]


def test_metrics_probe_accepts_llamacpp_prefix_and_reports_malformed_duplicates():
    body = 'llamacpp:spec_decode_accepted_tokens_per_pos_total{position="0"} 7\n'
    body += 'llamacpp:spec_decode_accepted_tokens_per_pos_total{position="0"} 8\n'
    body += 'llamacpp:spec_decode_accepted_tokens_per_pos_total{slot="x"} nope\n'
    body += "# filler\n" * 1000
    server, url = _metrics_server(body)
    try:
        result = MetricsProbe(url).snapshot()
        assert result["accepted_per_position"] == {"0": 7.0}
        assert result["parse_errors"] == 2
    finally:
        server.shutdown()
        server.server_close()


def _scratch_log():
    return os.path.join(os.getcwd(), f".l5-observation-{uuid.uuid4().hex}.log")


def test_child_log_counts_only_appended_interval_and_repeated_progress():
    path = _scratch_log()
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("startup\nprogress = 1.00\n")
    probe = ChildLogProbe(path)
    before = probe.snapshot()
    with open(path, "a", encoding="utf-8") as stream:
        stream.write("making room\nselected slot by id 2\n")
        stream.write("slot=2 progress = 1.00\nslot=2 progress = 1.00\n")
        stream.write("forcing full prompt re-processing\nexceeds cache size limit\n")
    try:
        delta = probe.delta(before)
        assert delta["errors"] == []
        assert delta["counters"]["making_room"] == 1
        assert delta["counters"]["selected_slot_by_id"] == 1
        assert delta["counters"]["repeated_progress_1_00"] == 1
        assert delta["counters"]["forcing_full_prompt_re_processing"] == 1
        assert delta["counters"]["exceeds_cache_size_limit"] == 1
    finally:
        os.unlink(path)


def test_child_log_truncation_is_explicit():
    path = _scratch_log()
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("making room\n")
    probe = ChildLogProbe(path)
    before = probe.snapshot()
    try:
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("new\n")
        result = probe.delta(before)
        assert result["reset"] and result["errors"]
    finally:
        os.unlink(path)


def test_child_log_large_existing_file_uses_eof_cursor_and_caps_interval():
    path = _scratch_log()
    try:
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("x" * (8 * 1024 * 1024 + 100))
        probe = ChildLogProbe(path, max_bytes=32)
        before = probe.snapshot()
        with open(path, "a", encoding="utf-8") as stream:
            stream.write("making room\n" * 200)
        result = probe.delta(before)
        assert result["partial"] and result["errors"]
        assert result["counters"]["making_room"] < 200
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_child_log_missing_before_snapshot_is_unknown():
    path = _scratch_log()
    probe = ChildLogProbe(path)
    before = probe.snapshot()
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("making room\n")
    try:
        result = probe.delta(before)
        assert result["errors"] and not result["counters"]
    finally:
        os.unlink(path)


def test_report_keeps_old_results_and_prints_observation_counters():
    def result(label, accepted, child):
        return {"label": label, "target": {"base_url": "u", "model": "m"}, "scenarios": [{
            "name": "s", "groups": {"g": {"ttft_s": {"n": 1, "p50": .1},
            "accepted_per_position": {"0": accepted}, "child_log": {"counters": {"making_room": child}}}}
        }]}
    md = render_markdown(result("A", 3, 1))
    assert "accepted/pos 0:3" in md and "child log making_room=1" in md
    diff, rows = compare(result("A", 3, 1), result("B", 5, 2))
    assert "accepted position 0" in diff and "child log making_room" in diff
    assert any(row["metric"] == "accepted position 0" and row["delta"] == 2.0 for row in rows)


def test_observation_comparison_preserves_counter_present_only_in_a():
    a = {"target": {"base_url": "u"}, "scenarios": [],
         "metrics": {"accepted_per_position": {"0": 5}},
         "child_log": {"counters": {"making_room": 2}}}
    b = {"target": {"base_url": "u"}, "scenarios": []}
    _, rows = compare(a, b)
    assert {row["metric"] for row in rows} == {"accepted position 0", "child log making_room"}
    assert all(row["a"] is not None and row["b"] is None and row["delta"] is None for row in rows)


def test_real_slot_log_does_not_confuse_different_tasks(tmp_path):
    log = tmp_path / "child.log"
    log.write_text("boot\n", encoding="utf-8")
    probe = ChildLogProbe(str(log))
    before = probe.snapshot()
    with log.open("a", encoding="utf-8") as stream:
        stream.write("slot update_slots: id 0 | task 14 | prompt processing progress = 1.00\n")
        stream.write("slot update_slots: id 0 | task 15 | prompt processing progress = 1.00\n")
        stream.write("slot update_slots: id 0 | task 15 | prompt processing progress = 1.00\n")
        stream.write("slot update_slots: id 0 | task 15 | prompt processing progress = 1.001\n")
    result = probe.delta(before)
    assert result["counters"]["repeated_progress_1_00"] == 1
    assert result["counters"]["progress_1_00"] == 3


def test_metric_missing_and_parse_errors_are_not_successful_deltas():
    assert metrics_delta({"accepted_per_position": {}}, {"accepted_per_position": {}})["errors"]
    bad = {"accepted_per_position": {"0": 5}, "parse_errors": 1}
    assert metrics_delta(bad, {"accepted_per_position": {"0": 10}})["errors"]
