"""Real shared-library chat behavior; mock output makes no model-quality claim."""
import itertools
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import sonder_inference as si
from sonder_inference._lib import Library


@pytest.mark.parametrize("messages", [[], [{"role": "assistant", "content": "hi"}],
    [{"role": "unknown", "content": "hi"}], [{"role": "user"}], [{"content": "hi"}]])
def test_chat_invalid_conversation_leaves_session_usable(engine, model, messages):
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        with pytest.raises(si.InvalidArgumentError):
            session.chat(messages)
        assert session.chat([si.ChatMessage("user", "again")]).completion_tokens == 2


@pytest.mark.parametrize("message, error", [
    ({"role": "user", "content": "x\0y"}, ValueError),
    ({"role": "us\0er", "content": "hi"}, ValueError),
    ({"role": "user", "content": 1}, TypeError),
    ("user: hi", TypeError),
    ({"role": "user", "content": "hi", "images": []}, si.UnsupportedError),
    ({"role": "assistant", "content": "hi", "tool_calls": []}, si.UnsupportedError),
    ({"role": "assistant", "content": "hi", "reasoning_content": "private"}, si.UnsupportedError)])
def test_chat_rejects_truncation_and_unsupported_fields(engine, model, message, error):
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        with pytest.raises(error):
            session.chat([message])


def test_chat_count_bound_and_bounded_iterable_consumption(engine, model):
    count = [0]

    def forever():
        while True:
            count[0] += 1
            yield si.ChatMessage("user", "")

    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        assert session.chat(itertools.repeat(si.ChatMessage("user", ""), 1024)).completed
        with pytest.raises(si.InvalidArgumentError, match="1024"):
            session.chat(forever())
        assert count == [1025]
        assert session.generate("still usable").completed


def test_chat_stop_exception_cancel_and_reuse(engine, model):
    messages = [si.ChatMessage("user", "hi")]
    with engine.create_session(model, si.SamplingConfig.greedy(12)) as session:
        chunks = []

        def stop(text):
            chunks.append(text)
            return len(chunks) < 2

        stopped = session.chat(messages, on_token=stop)
        assert stopped.chunks == len(chunks) == 2 and stopped.text == "".join(chunks)

        def boom(_text):
            raise RuntimeError("chat callback failure")

        with pytest.raises(RuntimeError, match="chat callback failure"):
            session.chat(messages, on_token=boom)
        cancelled = session.chat(messages, on_token=lambda _text: session.cancel())
        assert cancelled.cancelled and cancelled.completion_tokens < 12
        assert session.chat(messages).completion_tokens == 12
        assert session.generate("after chat").completion_tokens == 12


def test_chat_parallel_sessions_reuse(engine, model):
    def work(index):
        with engine.create_session(model, si.SamplingConfig.greedy(5, seed=index)) as session:
            for _ in range(8):
                assert session.chat([si.ChatMessage("user", "héllo ✓")]).completion_tokens == 5
            return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(work, range(4)))


@pytest.mark.parametrize("capture_text", [False, True])
def test_chat_telemetry_consent_kind_and_cursor(tmp_path, capture_text):
    path = tmp_path / "chat.jsonl"
    private = "chat-private-canary-20261004"
    with si.Engine(telemetry_level=si.TelemetryLevel.DEEP, telemetry_jsonl_path=str(path),
                   capture_text=capture_text) as engine:
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(4)) as session:
                result = session.chat([si.ChatMessage("user", private)])
    raw = path.read_text()
    events = [json.loads(line) for line in raw.splitlines()]
    queued = [event for event in events if event["event_type"] == "request.queued"]
    assert len(queued) == 1 and queued[0]["attributes"]["kind"] == "chat"
    assert private not in raw
    tokens = [event for event in events if event["event_type"] == "inference.token.generated"]
    assert len(tokens) == 4
    if capture_text:
        assert "".join(event["attributes"]["text"] for event in tokens) == result.text
    else:
        assert all("text" not in event["attributes"] for event in tokens)
    assert len({event["event_id"] for event in events}) == len(events)
    sequences = [event["sequence"] for event in events]
    assert sequences == sorted(set(sequences))


def test_legacy_abi_library_without_chat_still_generates(lib):
    # A proxy hides only the additive symbol; every other operation calls the
    # actual shared library. Local qualification also loads archived old .so.
    class WithoutChat:
        def __getattr__(self, name):
            if name == "sonder_session_chat":
                raise AttributeError(name)
            return getattr(lib.cdll, name)

    legacy = Library(WithoutChat(), lib.path)
    assert legacy.abi_version == 1
    with si.Engine(library=legacy, telemetry_level=si.TelemetryLevel.OFF) as engine:
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
                assert session.generate("legacy generation").completion_tokens == 2
                with pytest.raises(si.UnsupportedError, match="sonder_session_chat"):
                    session.chat([si.ChatMessage("user", "hi")])
