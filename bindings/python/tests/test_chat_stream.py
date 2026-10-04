"""Structured chat iterators against the real C ABI; deterministic mock data."""
import json
import itertools
import threading
import weakref

import pytest

import sonder_inference as si
from sonder_inference._lib import Library
from test_stream_backpressure import long_backend  # noqa: F401 - pytest fixture


@pytest.mark.parametrize("capture_text", [False, True])
def test_chat_stream_result_metadata_privacy_and_cursor(tmp_path, capture_text):
    path = tmp_path / "chat-stream.jsonl"
    private = "synthetic-chat-stream-private-canary"
    metadata = si.SessionMetadata(session_id="chat-stream:1", run_id="chat-stream-run:1",
                                  agent_id="chat-stream-agent:1", task_id="chat-stream-task:1")
    with si.Engine(telemetry_level=si.TelemetryLevel.DEEP, telemetry_jsonl_path=str(path),
                   capture_text=capture_text) as engine:
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(4), metadata=metadata) as session:
                with session.chat_stream([si.ChatMessage("system", "synthetic system message"),
                                          {"role": "user", "content": private}]) as stream:
                    chunks = list(stream)
                assert stream.result is not None and stream.result.completed
                assert stream.result.completion_tokens == len(chunks) == 4
                assert stream.result.text == "".join(chunks)
                assert not stream._thread.is_alive()
    raw = path.read_text()
    assert private not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    queued = [event for event in events if event["event_type"] == "request.queued"]
    assert len(queued) == 1 and queued[0]["attributes"]["kind"] == "chat"
    assert queued[0]["attributes"]["messages"] == 2
    scoped = [event for event in events if event.get("session_id") == metadata.session_id]
    assert scoped and all(event["producer"].get("synthetic") is True for event in scoped)
    assert all(all(event.get(key) == value for key, value in metadata.__dict__.items()) for event in scoped)
    tokens = [event for event in events if event["event_type"] == "inference.token.generated"]
    assert len(tokens) == 4
    if capture_text:
        assert "".join(event["attributes"]["text"] for event in tokens) == stream.result.text
    else:
        assert all("text" not in event["attributes"] for event in tokens)
    assert len({event["event_id"] for event in events}) == len(events)
    assert [event["sequence"] for event in events] == sorted({event["sequence"] for event in events})


@pytest.mark.parametrize("messages,error", [
    ([], si.InvalidArgumentError),
    ([{"role": "user"}], si.InvalidArgumentError),
    ([{"role": "user", "content": "x\0y"}], ValueError),
    ([{"role": "us\0er", "content": "x"}], ValueError),
    ([{"role": "user", "content": 1}], TypeError),
    (["user: x"], TypeError),
    ([{"role": "user", "content": "x", "images": []}], si.UnsupportedError),
])
def test_chat_stream_preparation_rejects_invalid_input_and_preserves_reuse(engine, model, messages, error):
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        with pytest.raises(error):
            session.chat_stream(messages)
        with session.chat_stream([si.ChatMessage("user", "reuse")]) as stream:
            assert len(list(stream)) == 2 and stream.result.completed


@pytest.mark.parametrize("role", ["unknown", "assistant"])
def test_chat_stream_native_validation_propagates_and_joins(engine, model, role):
    with engine.create_session(model) as session:
        with session.chat_stream([si.ChatMessage(role, "synthetic invalid conversation")]) as stream:
            with pytest.raises(si.InvalidArgumentError):
                list(stream)
        assert not stream._thread.is_alive() and stream.result is None
        assert session.generate("reuse").completed


def test_chat_stream_iterable_count_bound_and_closed_session(engine, model):
    consumed = [0]

    def forever():
        while True:
            consumed[0] += 1
            yield si.ChatMessage("user", "")

    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        with session.chat_stream(itertools.repeat(si.ChatMessage("user", ""), 1024)) as stream:
            assert len(list(stream)) == 2 and stream.result.completed
        with pytest.raises(si.InvalidArgumentError, match="1024"):
            session.chat_stream(forever())
        assert consumed == [1025]
    with pytest.raises(si.InvalidStateError, match="closed"):
        session.chat_stream([si.ChatMessage("user", "closed")])


def test_chat_stream_prepares_caller_snapshot_and_routes_structured_chat(lib, long_backend):
    url, requests = long_backend
    with si.Engine(telemetry_level=si.TelemetryLevel.OFF) as engine:
        engine.register_ollama_backend(url)
        with engine.load_model("ollama", "synthetic-long-chat") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(2048)) as session:
                original = session._request
                entered, release = threading.Event(), threading.Event()

                def gated_request(*args, **kwargs):
                    entered.set()
                    assert release.wait(10)
                    return original(*args, **kwargs)

                session._request = gated_request
                messages = [{"role": "system", "content": "synthetic original system"},
                            {"role": "user", "content": "synthetic original user ✓"}]
                with session.chat_stream(iter(messages)) as stream:
                    try:
                        assert entered.wait(10)
                        messages[0]["content"] = "mutated system"
                        messages[1]["role"] = "assistant"
                        messages.clear()
                    finally:
                        release.set()
                    chunks = list(stream)
                assert len(chunks) == 2048 and stream.result.completed
                assert "".join(chunks) == stream.result.text == "".join(f"{i}✓ " for i in range(2048))
                assert not stream._thread.is_alive()
                body = requests.get(timeout=5)
                assert body["messages"] == [{"role": "system", "content": "synthetic original system"},
                                            {"role": "user", "content": "synthetic original user ✓"}]


def test_chat_stream_legacy_missing_export_rejects_before_iteration(lib):
    class WithoutChat:
        def __getattr__(self, name):
            if name == "sonder_session_chat":
                raise AttributeError(name)
            return getattr(lib.cdll, name)

    legacy = Library(WithoutChat(), lib.path)
    with si.Engine(library=legacy, telemetry_level=si.TelemetryLevel.OFF) as engine:
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
                assert session.generate("legacy generation").completed
                with pytest.raises(si.UnsupportedError, match="sonder_session_chat"):
                    session.chat_stream([si.ChatMessage("user", "unsupported")])


def test_chat_stream_native_inputs_live_until_call_finishes(engine, model):
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        prepare, request = session._chat_args, session._request
        entered, release = threading.Event(), threading.Event()
        refs = []

        def observed_args(messages):
            args = prepare(messages)
            refs.append(weakref.ref(args[0]))
            return args

        def gated_request(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return request(*args, **kwargs)

        session._chat_args, session._request = observed_args, gated_request
        with session.chat_stream([si.ChatMessage("user", "synthetic owned input")]) as stream:
            try:
                assert entered.wait(10)
                assert refs[0]() is not None
            finally:
                release.set()
            assert len(list(stream)) == 2 and stream.result.completed
            assert not stream._thread.is_alive()
            assert refs[0]() is None


def test_public_token_stream_constructor_remains_compatible(engine, model):
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        with si.TokenStream(session, "synthetic direct constructor") as stream:
            assert stream._prompt == "synthetic direct constructor"
            assert len(list(stream)) == 2
        assert stream.result.completed and not stream._thread.is_alive()
