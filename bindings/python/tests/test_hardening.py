"""Regression tests for C ABI parity, safety, concurrency, and lifecycle hardening."""

from __future__ import annotations

import ctypes
import gc
import json
import math
import sys
import threading
import time
import types
import weakref
from pathlib import Path
from typing import Any

import pytest

import sonder_inference as si
from sonder_inference import _lib as L
from sonder_inference import _structs as S
from sonder_inference.errors import SonderError, Status


# ---------------------------------------------------------------------------
# Struct layout, sizes, and enum parity against include/sonder_inference.h


def test_all_c_abi_struct_layouts_and_sizes():
    # Pointer-width aware checks for 64-bit platforms (LP64 / LLP64).
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(S.EngineOptions) == S.ENGINE_OPTIONS_SIZE == 24
        assert S.EngineOptions.struct_size.offset == 0
        assert S.EngineOptions.telemetry_level.offset == 4
        assert S.EngineOptions.telemetry_jsonl_path.offset == 8
        assert S.EngineOptions.capture_text.offset == 16

        assert ctypes.sizeof(S.CLogitBias) == S.LOGIT_BIAS_SIZE == 8
        assert S.CLogitBias.token.offset == 0
        assert S.CLogitBias.bias.offset == 4

        assert ctypes.sizeof(S.CSamplingConfigV1) == S.SAMPLING_CONFIG_V1_SIZE == 48
        assert ctypes.sizeof(S.CSamplingConfig) == S.SAMPLING_CONFIG_SIZE == 80
        assert S.CSamplingConfig.struct_size.offset == 0
        assert S.CSamplingConfig.has_seed.offset == 24
        assert S.CSamplingConfig.seed.offset == 32
        assert S.CSamplingConfig.max_tokens.offset == 40
        assert S.CSamplingConfig.typical_p.offset == 44
        assert S.CSamplingConfig.presence_penalty.offset == 48
        assert S.CSamplingConfig.frequency_penalty.offset == 52
        assert S.CSamplingConfig.repeat_last_n.offset == 56
        assert S.CSamplingConfig.num_ctx.offset == 60
        assert S.CSamplingConfig.logit_bias.offset == 64
        assert S.CSamplingConfig.logit_bias_count.offset == 72

        assert ctypes.sizeof(S.CGenerationStats) == S.GENERATION_STATS_SIZE == 48
        assert S.CGenerationStats.struct_size.offset == 0
        assert S.CGenerationStats.outcome.offset == 4
        assert S.CGenerationStats.prompt_tokens.offset == 8
        assert S.CGenerationStats.completion_tokens.offset == 16
        assert S.CGenerationStats.chunks.offset == 24
        assert S.CGenerationStats.ttft_ms.offset == 32
        assert S.CGenerationStats.total_ms.offset == 40

        assert ctypes.sizeof(S.CChatMessage) == S.CHAT_MESSAGE_SIZE == 24
        assert S.CChatMessage.struct_size.offset == 0
        assert S.CChatMessage.role.offset == 8
        assert S.CChatMessage.content.offset == 16

        assert ctypes.sizeof(S.CSessionMetadata) == S.SESSION_METADATA_SIZE == 40
        assert S.CSessionMetadata.struct_size.offset == 0
        assert S.CSessionMetadata.session_id.offset == 8
        assert S.CSessionMetadata.run_id.offset == 16
        assert S.CSessionMetadata.agent_id.offset == 24
        assert S.CSessionMetadata.task_id.offset == 32

        assert ctypes.sizeof(S.CRequestMetadataV1) == S.REQUEST_METADATA_V1_SIZE == 16
        assert ctypes.sizeof(S.CRequestMetadata) == S.REQUEST_METADATA_SIZE == 24
        assert S.CRequestMetadata.struct_size.offset == 0
        assert S.CRequestMetadata.parent_request_id.offset == 8
        assert S.CRequestMetadata.workload_class.offset == 16
        assert S.CRequestMetadata.priority.offset == 20

        assert ctypes.sizeof(S.CGpuInfo) == S.GPU_INFO_SIZE == 496

    assert S.MAX_CORRELATION_ID_BYTES == 128
    assert S.MAX_CHAT_MESSAGES == 1024
    assert [int(x) for x in Status] == list(range(12))
    assert [int(x) for x in si.TelemetryLevel] == [0, 1, 2, 3]
    assert [int(x) for x in si.Outcome] == [0, 1, 2, 3]
    assert [int(x) for x in si.WorkloadClass] == list(range(8))


def test_py_typed_marker_is_present():
    pkg_dir = Path(si.__file__).resolve().parent
    assert (pkg_dir / "py.typed").is_file()


# ---------------------------------------------------------------------------
# Library loading, ABI version checks, status fallback, and probe hygiene


def test_incompatible_abi_version_checked_before_declaring_v1_symbols():
    """An ABI v2 library that removed v1 symbols must report an ABI mismatch, not AttributeError."""

    fake_v2 = types.SimpleNamespace(sonder_abi_version=lambda: 2)
    with pytest.raises(si.LibraryNotFoundError, match=r"C ABI version 2 is not supported"):
        L.Library(fake_v2, "fake_v2.so")  # type: ignore[arg-type]


def test_missing_required_symbols_raise_library_not_found_error(lib):
    class EmptyCdLL:
        pass

    with pytest.raises(si.LibraryNotFoundError, match="sonder_abi_version"):
        L.Library(EmptyCdLL(), "empty.so")  # type: ignore[arg-type]

    class MissingEngineCreate:
        def __getattr__(self, name: str) -> Any:
            if name == "sonder_engine_create":
                raise AttributeError(name)
            return getattr(lib.cdll, name)

    with pytest.raises(si.LibraryNotFoundError, match="sonder_engine_create"):
        L.Library(MissingEngineCreate(), lib.path)  # type: ignore[arg-type]


def test_find_library_does_not_swallow_abi_mismatch(monkeypatch, tmp_path):
    fake_v99 = types.SimpleNamespace(sonder_abi_version=lambda: 99)

    monkeypatch.setattr(L, "_library", None)
    monkeypatch.delenv(si.ENV_LIBRARY, raising=False)
    monkeypatch.delenv(si.ENV_LIB_DIR, raising=False)
    monkeypatch.setattr(L, "_candidates", lambda _path: [])
    monkeypatch.setattr(L.ctypes.util, "find_library", lambda _name: "libsonder_inference.so")
    monkeypatch.setattr(L.ctypes, "CDLL", lambda _found: fake_v99)
    with pytest.raises(si.LibraryNotFoundError, match="C ABI version 99 is not supported"):
        si.load_library()


def test_load_library_accepts_different_case_on_windows(lib):
    if sys.platform != "win32":
        pytest.skip("Windows case-insensitive path test")
    swapped = lib.path.swapcase()
    if Path(swapped).is_file():
        assert si.load_library(swapped) is lib


def test_supports_extended_sampling_probe_clears_last_error_and_is_thread_safe(lib):
    fresh = L.Library(lib.cdll, lib.path)
    results: list[bool] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def probe() -> None:
        try:
            barrier.wait(5)
            results.append(fresh.supports_extended_sampling)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=probe) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
        assert not t.is_alive()
    assert not errors
    assert results == [True] * 6
    # The intentional typical_p=0 probe failure must not leak into last_error().
    assert fresh.last_error() == ""


def test_status_string_fallback_and_ok_status_error_code(lib):
    assert lib.status_string(int(Status.TIMEOUT)) == "timeout"

    class EmptyLastErrorLib:
        def last_error(self) -> str:
            return ""

        def status_string(self, rc: int) -> str:
            return lib.status_string(rc)

    with pytest.raises(si.SonderTimeoutError, match="timeout"):
        si.api._check(EmptyLastErrorLib(), int(Status.TIMEOUT))  # type: ignore[arg-type]

    class OkSubclassError(SonderError):
        status = Status.OK

    e = OkSubclassError("ok-sentinel")
    assert e.status == Status.OK
    assert e.code == 0


# ---------------------------------------------------------------------------
# Input validation: float32 overflow, PathLike, falsy non-strings, types


def test_float32_overflow_rejected_instead_of_silent_inf_ban(lib):
    # -1e300 is finite and far outside [-100, 100]; casting directly to c_float
    # would silently turn it into -inf, which the C++ validator accepts as a ban.
    with pytest.raises(si.InvalidArgumentError, match="overflows float32"):
        si.SamplingConfig(logit_bias={1: -1e300}).validate(lib)
    with pytest.raises(si.InvalidArgumentError, match="overflows float32"):
        si.SamplingConfig(logit_bias={1: 1e300}).validate(lib)
    with pytest.raises(si.InvalidArgumentError, match="temperature"):
        si.SamplingConfig(temperature=1e300).validate(lib)
    # Explicit -math.inf token ban remains valid.
    si.SamplingConfig(logit_bias={1: -math.inf}).validate(lib)


@pytest.mark.parametrize("bad_bias", [0, False, "", "ab", b"ab", bytearray(b"ab")])
def test_logit_bias_rejects_falsy_and_string_types(lib, bad_bias):
    with pytest.raises(TypeError):
        si.SamplingConfig(logit_bias=bad_bias).validate(lib)  # type: ignore[arg-type]


def test_engine_pathlike_and_strict_argument_validation(lib, tmp_path):
    path = tmp_path / "pathlike_telemetry.jsonl"
    with si.Engine(telemetry_level=si.TelemetryLevel.METRICS, telemetry_jsonl_path=path) as engine:
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
                assert session.generate("pathlike").completed
    assert path.stat().st_size > 0

    for bad_path in (0, False, [], b"foo.jsonl"):
        with pytest.raises(TypeError):
            si.Engine(telemetry_level=si.TelemetryLevel.OFF,
                      telemetry_jsonl_path=bad_path)  # type: ignore[arg-type]

    for bad_level in (True, False, "off"):
        with pytest.raises(TypeError):
            si.Engine(telemetry_level=bad_level)  # type: ignore[arg-type]
    with pytest.raises(si.InvalidArgumentError, match="telemetry_level"):
        si.Engine(telemetry_level=99)

    with si.Engine(telemetry_level=si.TelemetryLevel.OFF) as engine:
        for bad_url in (0, False, [], 123):
            with pytest.raises(TypeError):
                engine.register_ollama_backend(bad_url)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Model"):
            engine.create_session(None)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Model"):
            engine.create_session("mock:tiny")  # type: ignore[arg-type]

        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with pytest.raises(TypeError, match="SamplingConfig"):
                engine.create_session(model, {"temperature": 0.0})  # type: ignore[arg-type]


def test_chat_and_callback_eager_type_validation(engine, model, monkeypatch):
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        for bad_messages in ("hello", "", b"hello", {"role": "user", "content": "hi"}, si.ChatMessage("user", "hi")):
            with pytest.raises(TypeError):
                session.chat(bad_messages)  # type: ignore[arg-type]
            with pytest.raises(TypeError):
                session.chat_stream(bad_messages)  # type: ignore[arg-type]

        # Non-callable on_token must fail before invoking the native library.
        called = []
        orig_use = session._use

        def fail_if_entered():
            called.append(True)
            return orig_use()

        monkeypatch.setattr(session, "_use", fail_if_entered)
        with pytest.raises(TypeError, match="callable"):
            session.generate("hi", on_token=123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="callable"):
            session.chat([si.ChatMessage("user", "hi")], on_token="bad")  # type: ignore[arg-type]
        assert not called


def test_null_native_handle_is_rejected():
    fake_lib = types.SimpleNamespace(cdll=types.SimpleNamespace(destroy=lambda _p: None))

    class DummyHandle(si.api._Handle):
        _destroy_fn = "destroy"

    with pytest.raises(si.InternalError, match="null native handle"):
        DummyHandle(fake_lib, 0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Incremental UTF-8 decoding, callback stop/exception guard, NaN ttft_ms


def test_incremental_utf8_boundaries_null_chunks_and_callback_guards(engine, model):
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        # 1) Split 4-byte emoji + 3-byte checkmark one byte at a time, plus
        #    zero-length and NULL pointer chunks, plus invalid byte + truncated tail.
        payloads = [
            b"",
            None,
            b"\xf0", b"\x9f", b"\x9a", b"\x80",  # 🚀 split across 4 callbacks
            b"\xe2", b"\x9c", b"\x93",            # ✓ split across 3 callbacks
            b"ok\xff!",                           # invalid UTF-8 byte mid-stream
            b"\xe2\x9c",                          # truncated trailing sequence at EOF
        ]

        def fake_native(_handle, _prompt, cb, _user, stats_ptr):
            for chunk in payloads:
                if chunk is None:
                    rc = cb(None, None, 4)
                else:
                    buf = ctypes.create_string_buffer(chunk) if chunk else None
                    ptr = ctypes.addressof(buf) if buf is not None else None
                    rc = cb(None, ptr, len(chunk))
                assert rc == 0
            stats = stats_ptr._obj
            stats.outcome = int(si.Outcome.COMPLETED)
            stats.prompt_tokens = 1
            stats.completion_tokens = 3
            stats.chunks = 3
            stats.ttft_ms = float("nan")
            stats.total_ms = 1.5
            return 0

        seen: list[str] = []
        res = session._request(fake_native, (b"p",), seen.append)
        assert res.completed
        assert res.text == "🚀✓ok\ufffd!\ufffd"
        assert "".join(seen) == res.text
        assert seen[0] == "🚀"
        assert seen[1] == "✓"
        assert res.ttft_ms is None  # NaN ttft_ms normalized to None

        # 2) Misbehaving native caller invoking callback after stop (return False)
        #    or after exception must not re-enter Python on_token or flush tail.
        calls_after_stop: list[str] = []

        def fake_ignores_stop(_handle, _prompt, cb, _user, stats_ptr):
            for chunk in (b"first", b"second", b"\xe2\x9c"):
                buf = ctypes.create_string_buffer(chunk)
                cb(None, ctypes.addressof(buf), len(chunk))
            stats = stats_ptr._obj
            stats.outcome = int(si.Outcome.COMPLETED)
            stats.ttft_ms = 0.5
            stats.total_ms = 1.0
            return 0

        def stop_on_first(text: str) -> bool:
            calls_after_stop.append(text)
            return False

        res2 = session._request(fake_ignores_stop, (b"p",), stop_on_first)
        assert calls_after_stop == ["first"]
        assert res2.text == "first"

        calls_after_exc: list[str] = []

        def raise_on_first(text: str) -> None:
            calls_after_exc.append(text)
            raise RuntimeError("boom-once")

        with pytest.raises(RuntimeError, match="boom-once"):
            session._request(fake_ignores_stop, (b"p",), raise_on_first)
        assert calls_after_exc == ["first"]


# ---------------------------------------------------------------------------
# Telemetry callback re-entrant clear -> re-register fix


def test_telemetry_callback_reentrant_clear_and_reregister_delivers_events(lib):
    lock = threading.Lock()
    cb1_lines: list[str] = []
    cb2_lines: list[str] = []
    switched = [False]

    with si.Engine(telemetry_level=si.TelemetryLevel.METRICS) as engine:
        def cb2(line: str) -> None:
            with lock:
                cb2_lines.append(line)

        def cb1(line: str) -> None:
            with lock:
                cb1_lines.append(line)
                if not switched[0]:
                    switched[0] = True
                    # Clearing then re-registering from inside the callback must
                    # re-attach the native trampoline rather than leaving it detached.
                    engine.set_telemetry_callback(None)
                    engine.set_telemetry_callback(cb2)

        engine.set_telemetry_callback(cb1)
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(3)) as session:
                assert session.generate("first triggers reentrant clear+reregister").completed
                assert session.generate("second must reach cb2").completed
                engine.set_telemetry_callback(None)

    with lock:
        assert len(cb1_lines) >= 1
        assert len(cb2_lines) >= 1
        events2 = [json.loads(line) for line in cb2_lines]
        assert any(e["event_type"] == "request.completed" for e in events2)


# ---------------------------------------------------------------------------
# Concurrent Engine child creation vs close() (WeakSet thread-safety)


def test_concurrent_child_creation_and_engine_close_never_corrupts_weakset(lib):
    for _ in range(8):
        engine = si.Engine(telemetry_level=si.TelemetryLevel.OFF)
        engine.register_mock_backend()
        model = engine.load_model("mock", "mock:tiny")
        stop = threading.Event()
        unexpected: list[BaseException] = []

        def creator() -> None:
            while not stop.is_set():
                try:
                    s = engine.create_session(model, si.SamplingConfig.greedy(1))
                    s.close()
                except si.InvalidStateError:
                    return
                except BaseException as exc:  # noqa: BLE001
                    unexpected.append(exc)
                    return

        workers = [threading.Thread(target=creator, daemon=True) for _ in range(4)]
        for w in workers:
            w.start()
        time.sleep(0.02)
        engine.close()
        stop.set()
        for w in workers:
            w.join(5)
            assert not w.is_alive()
        assert not unexpected, unexpected
        assert engine.closed and model.closed


# ---------------------------------------------------------------------------
# TokenStream eager validation, abandoned GC cleanup, and slow-prefill close


def test_token_stream_direct_constructor_validates_eagerly(engine, model, monkeypatch):
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        def forbidden(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("worker should not start on invalid input")

        monkeypatch.setattr(si.TokenStream, "_start", forbidden)
        with pytest.raises(ValueError, match="NUL"):
            si.TokenStream(session, "bad\0prompt")
        with pytest.raises(TypeError):
            si.TokenStream(session, 123)  # type: ignore[arg-type]

    # Session is now closed.
    with pytest.raises(si.InvalidStateError, match="closed"):
        si.TokenStream(session, "closed")


def test_abandoned_backpressured_token_stream_is_gc_cancelled_not_leaked(engine, model):
    """Breaking out of a stream without .close() must cancel the worker on GC."""
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        cancelled_seen = threading.Event()

        def infinite_producer(_function: Any, _args: tuple[Any, ...],
                              on_token: Any = None) -> si.GenerationResult:
            for i in range(500):
                if on_token is not None and on_token(f"tok{i}") is False:
                    cancelled_seen.set()
                    return si.GenerationResult(
                        text="", outcome=si.Outcome.CANCELLED, prompt_tokens=1,
                        completion_tokens=i, chunks=i, ttft_ms=0.1, total_ms=0.5)
            return si.GenerationResult(
                text="", outcome=si.Outcome.COMPLETED, prompt_tokens=1,
                completion_tokens=500, chunks=500, ttft_ms=0.1, total_ms=0.5)

        orig_request = session._request
        session._request = infinite_producer  # type: ignore[method-assign]
        try:
            stream = session.stream("abandoned")
            worker_thread = stream._thread
            stream_ref = weakref.ref(stream)

            # Wait until the 64-chunk buffer is full and the worker is backpressured.
            deadline = time.monotonic() + 5.0
            while stream._queue.qsize() < 64 and time.monotonic() < deadline:
                time.sleep(0.005)
            assert stream._queue.qsize() == 64
            assert worker_thread.is_alive()

            # Read one chunk and abandon the stream WITHOUT calling stream.close().
            assert next(stream) == "tok0"
            del stream
            gc.collect()

            assert stream_ref() is None, "worker thread kept a strong reference to TokenStream"
            assert cancelled_seen.wait(5.0), "abandoned TokenStream finalizer did not cancel worker"
            worker_thread.join(5.0)
            assert not worker_thread.is_alive()
        finally:
            session._request = orig_request  # type: ignore[method-assign]

        assert session.generate("reusable after abandoned stream").completed


def test_token_stream_close_reasserts_cancel_during_slow_prefill(engine, model):
    """If close() races before a slow-prefill call enters C (0 tokens emitted),
    close() must keep re-asserting cancel() so the prefill aborts promptly."""
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        entered_prefill = threading.Event()
        prefill_cancelled = threading.Event()
        orig_cancel = session.cancel
        orig_request = session._request

        def tracking_cancel() -> None:
            if entered_prefill.is_set():
                prefill_cancelled.set()
            orig_cancel()

        def slow_prefill_request(_function: Any, _args: tuple[Any, ...],
                                 _on_token: Any = None) -> si.GenerationResult:
            # Delay entering "native prefill" so the first close() cancel fires while idle.
            time.sleep(0.05)
            entered_prefill.set()
            assert prefill_cancelled.wait(3.0), "close() did not re-send cancel() during slow prefill"
            return si.GenerationResult(
                text="", outcome=si.Outcome.CANCELLED, prompt_tokens=1,
                completion_tokens=0, chunks=0, ttft_ms=None, total_ms=50.0)

        session.cancel = tracking_cancel  # type: ignore[method-assign]
        session._request = slow_prefill_request  # type: ignore[method-assign]
        try:
            stream = session.stream("slow prefill")
            started = time.monotonic()
            stream.close()
            elapsed = time.monotonic() - started
            assert elapsed < 2.0
            assert prefill_cancelled.is_set()
            assert stream.result is not None and stream.result.cancelled
            assert not stream._thread.is_alive()
        finally:
            session.cancel = orig_cancel  # type: ignore[method-assign]
            session._request = orig_request  # type: ignore[method-assign]
