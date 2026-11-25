"""C ABI v1.x bindings: extended stats, chunk callbacks, tools, embed, and dll_dir cleanup."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import sonder_inference as si
from sonder_inference._lib import Library


def test_dll_dir_handle_closed_when_library_init_fails(monkeypatch, tmp_path):
    """If Library(...) raises after ctypes.CDLL succeeds (e.g. ABI mismatch),
    the Windows os.add_dll_directory handle must be closed and not leaked."""
    fake_lib = tmp_path / "sonder_inference.dll"
    fake_lib.write_bytes(b"placeholder")
    closed: list[bool] = []

    class FakeDllDir:
        def close(self) -> None:
            closed.append(True)

    class FakeAbiFn:
        restype: Any = None
        argtypes: Any = None

        def __call__(self) -> int:
            return 999

    class BadAbiCdll:
        sonder_abi_version = FakeAbiFn()

    monkeypatch.setattr(si._lib, "_library", None)
    before_handles = list(si._lib._dll_dir_handles)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(si._lib.os, "add_dll_directory", lambda _p: FakeDllDir(), raising=False)
    monkeypatch.setattr(si._lib.ctypes, "CDLL", lambda _p: BadAbiCdll())

    with pytest.raises(si.LibraryNotFoundError, match="ABI version 999"):
        si.load_library(fake_lib)

    assert closed == [True]
    assert si._lib._dll_dir_handles == before_handles


def test_generation_result_v1x_stats_and_stop_reason(engine, model):
    with engine.create_session(model, si.SamplingConfig.greedy(4)) as session:
        res = session.generate("hello v1x stats")
        assert res.completed
        assert res.completion_tokens == 4
        assert res.stop_reason == si.StopReason.MAX_TOKENS
        assert res.cached_tokens is None
        assert res.draft_tokens is None
        assert res.draft_accepted_tokens is None
        assert res.reasoning == ""
        assert res.tool_calls == ()

        early = session.generate("stop early", lambda _t: False)
        assert early.stop_reason == si.StopReason.CALLBACK


def test_generate_ex_with_v1x_controls_and_callbacks(engine, model):
    tokens: list[str] = []
    reasoning: list[str] = []
    deltas: list[si.ToolCallDelta] = []
    meta = si.RequestMetadata(
        parent_request_id="v1x-py:1",
        response_format_type="json_object",
        separate_reasoning=True,
        enable_thinking=True,
        reasoning_effort="low",
        reasoning_budget_tokens=128,
    )
    with engine.create_session(model, si.SamplingConfig.greedy(5)) as session:
        res = session.generate(
            "generate with controls",
            tokens.append,
            metadata=meta,
            on_reasoning=reasoning.append,
            on_tool_call_delta=deltas.append,
        )
        assert res.completed
        assert res.completion_tokens == 5
        assert "".join(tokens) == res.text
        assert res.stop_reason == si.StopReason.MAX_TOKENS
        assert deltas == []


def test_chat_ex_native_tool_calling_and_delta_reassembly(engine, model):
    tool = si.ToolDefinition(
        name="get_weather",
        description="Get weather for a city",
        parameters_json=json.dumps({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }),
    )
    meta = si.RequestMetadata(
        tools=[tool],
        tool_choice=si.ToolChoice(mode=si.ToolChoiceMode.FUNCTION, function_name="get_weather"),
    )
    deltas: list[si.ToolCallDelta] = []
    with engine.create_session(model, si.SamplingConfig.greedy(8)) as session:
        res = session.chat(
            [si.ChatMessage("user", "Check weather in Austin")],
            metadata=meta,
            on_tool_call_delta=deltas.append,
        )
        assert res.completed
        assert len(deltas) == 2
        assert deltas[0].name == "get_weather"
        assert deltas[0].id is not None and deltas[0].id.startswith("call_mock_")
        assert deltas[1].arguments_delta
        assert len(res.tool_calls) == 1
        call = res.tool_calls[0]
        assert call.name == "get_weather"
        assert call.id == deltas[0].id
        parsed = json.loads(call.arguments_json)
        assert "city" in parsed

        # Callback exception propagation through _request_ex
        def boom(_delta: si.ToolCallDelta) -> None:
            raise RuntimeError("delta-boom")

        with pytest.raises(RuntimeError, match="delta-boom"):
            session.chat(
                [si.ChatMessage("user", "Check weather")],
                metadata=meta,
                on_tool_call_delta=boom,
            )
        # Session remains reusable after callback exception
        after = session.generate("after exception")
        assert after.completed


def test_request_metadata_v1x_validation(engine, model):
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        with pytest.raises(si.InvalidArgumentError, match="response_format_type"):
            session.generate("x", metadata=si.RequestMetadata(response_format_type="xml"))
        with pytest.raises(si.InvalidArgumentError, match="response_format_schema"):
            session.generate(
                "x",
                metadata=si.RequestMetadata(
                    response_format_type="json_object",
                    response_format_schema="{}",
                ),
            )
        with pytest.raises(si.InvalidArgumentError, match="reasoning_budget_tokens"):
            session.generate("x", metadata=si.RequestMetadata(reasoning_budget_tokens=-1))
        with pytest.raises(si.InvalidArgumentError, match="reasoning_budget_tokens"):
            session.generate("x", metadata=si.RequestMetadata(reasoning_budget_tokens=True))
        with pytest.raises(si.InvalidArgumentError, match="function_name"):
            session.chat(
                [si.ChatMessage("user", "hi")],
                metadata=si.RequestMetadata(tool_choice=si.ToolChoice(mode=si.ToolChoiceMode.FUNCTION)),
            )
        with pytest.raises(si.InvalidArgumentError, match="non-empty"):
            session.chat(
                [si.ChatMessage("user", "hi")],
                metadata=si.RequestMetadata(tools=[si.ToolDefinition(name="")]),
            )


def test_session_embed_and_validation(engine, model):
    with engine.create_session(model) as session:
        res = session.embed(["alpha beta", "gamma delta epsilon"])
        assert isinstance(res, si.EmbedResult)
        assert len(res.embeddings) == 2
        assert res.dimensions == 16
        assert len(res.embeddings[0]) == 16
        assert len(res.embeddings[1]) == 16
        assert res.embeddings[0] != res.embeddings[1]
        assert res.prompt_tokens >= 2

        # Deterministic across calls
        res2 = session.embed(["alpha beta"])
        assert res2.embeddings[0] == res.embeddings[0]

        with pytest.raises(TypeError, match="sequence of str"):
            session.embed("not a sequence")  # type: ignore[arg-type]
        with pytest.raises(si.InvalidArgumentError, match="at least one"):
            session.embed([])
        with pytest.raises(si.InvalidArgumentError, match="at most"):
            session.embed(["x"] * (si._structs.MAX_EMBED_INPUTS + 1))


def test_v1x_symbols_raise_unsupported_on_older_library(lib):
    class WithoutV1x:
        def __getattr__(self, name: str) -> Any:
            if name in ("sonder_session_generate_ex", "sonder_session_chat_ex", "sonder_session_embed"):
                raise AttributeError(name)
            return getattr(lib.cdll, name)

    legacy = Library(WithoutV1x(), lib.path)  # type: ignore[arg-type]
    with si.Engine(library=legacy, telemetry_level=si.TelemetryLevel.OFF) as engine:
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
                # Plain v1 generate still works on an older library using 48-byte stats.
                plain = session.generate("legacy ok")
                assert plain.completed
                assert plain.stop_reason == si.StopReason.NONE

                with pytest.raises(si.UnsupportedError, match="sonder_session_generate_ex"):
                    session.generate("x", on_reasoning=lambda _r: None)
                with pytest.raises(si.UnsupportedError, match="sonder_session_chat_ex"):
                    session.chat(
                        [si.ChatMessage("user", "hi")],
                        metadata=si.RequestMetadata(tools=[si.ToolDefinition("f")]),
                    )
                with pytest.raises(si.UnsupportedError, match="sonder_session_embed"):
                    session.embed(["hello"])
