import gc
import threading

import pytest

import sonder_inference as si


def test_engine_basics(engine):
    assert engine.device_count >= 1
    assert not engine.closed
    assert "Engine" in repr(engine)


def test_generate_collects_text_and_stats(engine, model):
    with engine.create_session(model, si.SamplingConfig.greedy(5)) as s:
        chunks = []
        r = s.generate("hello from python", on_token=chunks.append)
    assert r.completed and not r.cancelled
    assert r.outcome == si.Outcome.COMPLETED
    assert r.completion_tokens == 5 and r.chunks == 5
    assert r.prompt_tokens == 3
    assert len(chunks) == 5 and "".join(chunks) == r.text and r.text
    assert r.ttft_ms is not None and r.ttft_ms >= 0 and r.total_ms >= r.ttft_ms


def test_seeded_runs_are_deterministic(engine, model):
    cfg = si.SamplingConfig.greedy(12, seed=7)
    texts = []
    for _ in range(2):
        with engine.create_session(model, cfg) as s:
            texts.append(s.generate("determinism check").text)
    assert texts[0] == texts[1] and texts[0]


def test_new_sampling_fields_accepted_by_session(engine, model):
    cfg = si.SamplingConfig.greedy(6).replace(
        typical_p=0.95, presence_penalty=0.3, frequency_penalty=0.2, repeat_last_n=16,
        num_ctx=2048, logit_bias={3: 1.5})
    with engine.create_session(model, cfg) as s:
        r = s.generate("sampling fields")
    assert r.completed and r.completion_tokens == 6


def test_invalid_sampling_rejected_at_session_create(engine, model):
    with pytest.raises(si.InvalidArgumentError, match="frequency_penalty"):
        engine.create_session(model, si.SamplingConfig(frequency_penalty=5.0))


def test_callback_false_stops_early(engine, model):
    seen = []

    def on_token(t):
        seen.append(t)
        return len(seen) < 2  # False on the second chunk

    with engine.create_session(model, si.SamplingConfig.greedy(20)) as s:
        r = s.generate("stop early", on_token=on_token)
    assert len(seen) == 2
    assert r.text == "".join(seen)


def test_callback_exception_propagates_and_session_is_reusable(engine, model):
    class Boom(Exception):
        pass

    def on_token(_t):
        raise Boom("from callback")

    with engine.create_session(model, si.SamplingConfig.greedy(8)) as s:
        with pytest.raises(Boom):
            s.generate("x", on_token=on_token)
        assert s.generate("again").completed


def test_cancel_from_callback(engine, model):
    with engine.create_session(model, si.SamplingConfig.greedy(50)) as s:
        count = [0]

        def on_token(_t):
            count[0] += 1
            if count[0] == 3:
                s.cancel()

        r = s.generate("cancel me", on_token=on_token)
    assert r.cancelled
    assert r.completion_tokens < 50


def test_stream_yields_chunks_and_result(engine, model):
    with engine.create_session(model, si.SamplingConfig.greedy(7)) as s:
        with s.stream("stream please") as it:
            chunks = list(it)
        assert it.result is not None and it.result.completed
        assert "".join(chunks) == it.result.text and len(chunks) == 7


def test_stream_close_early(engine, model):
    with engine.create_session(model, si.SamplingConfig.greedy(500)) as s:
        it = s.stream("long stream")
        first = next(it)
        assert first
        it.close()
        with pytest.raises(StopIteration):
            next(it)
        assert s.generate("still usable").completed


def test_generate_runs_without_holding_the_gil(engine, model):
    # Several sessions on separate threads; ctypes releases the GIL during the C call.
    results = []

    def work(i):
        with engine.create_session(model, si.SamplingConfig.greedy(10, seed=i)) as s:
            results.append(s.generate(f"thread {i}").completion_tokens)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [10, 10, 10, 10]


def test_utf8_is_decoded(engine, model):
    with engine.create_session(model, si.SamplingConfig.greedy(3)) as s:
        r = s.generate("héllo wörld ✓")
    assert r.completed and isinstance(r.text, str)


def test_chat_is_a_stub(engine, model):
    with engine.create_session(model) as s:
        with pytest.raises(si.UnsupportedError, match="chat"):
            s.chat([{"role": "user", "content": "hi"}])
        with pytest.raises(NotImplementedError):
            s.chat([si.ChatMessage("user", "hi")])


def test_close_is_idempotent_and_closed_handles_raise(engine, model):
    s = engine.create_session(model)
    s.close()
    s.close()
    assert s.closed
    with pytest.raises(si.InvalidStateError, match="closed"):
        s.generate("x")


def test_engine_close_closes_children():
    e = si.Engine(telemetry_level=si.TelemetryLevel.OFF)
    e.register_mock_backend()
    m = e.load_model("mock", "mock:tiny")
    s = e.create_session(m, si.SamplingConfig.greedy(2))
    e.close()
    assert e.closed and m.closed and s.closed
    with pytest.raises(si.InvalidStateError):
        e.load_model("mock", "mock:tiny")


def test_model_can_be_released_before_session(engine):
    m = engine.load_model("mock", "mock:tiny")
    s = engine.create_session(m, si.SamplingConfig.greedy(3))
    m.close()
    assert s.generate("model released").completion_tokens == 3
    s.close()


def test_garbage_collection_releases_handles():
    e = si.Engine(telemetry_level=si.TelemetryLevel.OFF)
    e.register_mock_backend()
    s = e.create_session(e.load_model("mock", "mock:tiny"), si.SamplingConfig.greedy(2))
    s.generate("gc")
    del s, e
    gc.collect()  # finalizers run in a safe order; must not crash


def test_model_from_other_engine_rejected(engine):
    with si.Engine(telemetry_level=si.TelemetryLevel.OFF) as other:
        other.register_mock_backend()
        with other.load_model("mock", "mock:tiny") as m:
            with pytest.raises(si.InvalidArgumentError, match="different engine"):
                engine.create_session(m)


def test_embedded_nul_rejected_for_every_c_string(engine, model, tmp_path):
    # c_char_p stops at the first NUL, so these would be silently truncated.
    with engine.create_session(model, si.SamplingConfig.greedy(3)) as s:
        with pytest.raises(ValueError, match="NUL"):
            s.generate("visible\0hidden")
        with pytest.raises(ValueError, match="NUL"):
            s.stream("visible\0hidden")
        assert s.generate("still usable").completed
    with pytest.raises(ValueError, match="NUL"):
        engine.load_model("mock", "mock:tiny\0evil")
    with pytest.raises(ValueError, match="NUL"):
        engine.load_model("mo\0ck", "mock:tiny")
    with pytest.raises(ValueError, match="NUL"):
        engine.register_ollama_backend("http://127.0.0.1:1\0/x")
    with pytest.raises(ValueError, match="NUL"):
        si.Engine(telemetry_jsonl_path=str(tmp_path / "t.jsonl") + "\0.txt")


def test_telemetry_file(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    with si.Engine(telemetry_level=si.TelemetryLevel.STANDARD, telemetry_jsonl_path=str(path)) as e:
        e.register_mock_backend()
        with e.load_model("mock", "mock:tiny") as m, e.create_session(m, si.SamplingConfig.greedy(3)) as s:
            s.generate("telemetry")
    assert path.stat().st_size > 0
