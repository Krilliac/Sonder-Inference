"""close() racing an in-flight generate()/stream() on another thread.

The C ABI's sonder_session_destroy / sonder_engine_destroy free the handle
immediately, so destroying while sonder_session_generate runs is a
use-after-free. Each scenario runs in a child interpreter: a native crash (or
a hang) then shows up as a failed test with the exit code, never as a skip
and never as a torn-down pytest run.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import sonder_inference as si

_PRELUDE = textwrap.dedent("""
    import faulthandler, threading, time
    faulthandler.enable()
    import sonder_inference as si

    def make():
        e = si.Engine(telemetry_level=si.TelemetryLevel.OFF)
        e.register_mock_backend()
        m = e.load_model("mock", "mock:tiny")
        return e, m

    def slow_first_chunk_worker(s, max_tokens_note=""):
        started, cb_exited = threading.Event(), threading.Event()
        box = {}

        def on_token(_t):
            if not started.is_set():
                started.set()
                time.sleep(0.5)  # the closer runs while we are inside the C call
                cb_exited.set()

        def run():
            try:
                box["result"] = s.generate("race", on_token=on_token)
            except BaseException as e:
                box["error"] = e

        t = threading.Thread(target=run)
        t.start()
        assert started.wait(30), "generation never produced a chunk"
        return t, cb_exited, box
""")


def _run(body: str) -> None:
    script = _PRELUDE + textwrap.dedent(body) + "\nprint('SCENARIO-OK', flush=True)\n"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "SCENARIO-OK" in proc.stdout, (
        f"child exited with {proc.returncode:#x}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def test_session_close_waits_for_generate_on_other_thread(lib):
    _run("""
        e, m = make()
        s = e.create_session(m, si.SamplingConfig.greedy(2000))
        t, cb_exited, box = slow_first_chunk_worker(s)
        s.close()
        # close() must not destroy the C session while generate is still inside it.
        assert cb_exited.is_set(), "close() returned while generate() was still running"
        t.join(30)
        assert not t.is_alive()
        assert "error" not in box, box
        assert box["result"].cancelled, box["result"]  # close cancels the in-flight call
        assert box["result"].completion_tokens < 2000
        assert s.closed
        for call in (lambda: s.generate("after"), s.cancel, lambda: s.stream("after")):
            try:
                call()
            except si.InvalidStateError as exc:
                assert "closed" in str(exc)
            else:
                raise AssertionError("call after close did not raise")
        s.close()  # idempotent
        e.close()
    """)


def test_engine_close_waits_for_session_mid_call(lib):
    _run("""
        e, m = make()
        s = e.create_session(m, si.SamplingConfig.greedy(2000))
        t, cb_exited, box = slow_first_chunk_worker(s)
        e.close()
        assert cb_exited.is_set(), "Engine.close() returned while a session was mid-call"
        t.join(30)
        assert "error" not in box, box
        assert box["result"].cancelled
        assert e.closed and m.closed and s.closed
        try:
            s.generate("after")
        except si.InvalidStateError:
            pass
        else:
            raise AssertionError("generate after Engine.close did not raise")
    """)


def test_session_close_during_long_stream(lib):
    _run("""
        # Slow-yield on the worker so the generation is still inside the C
        # call when the session is closed (the mock backend is otherwise fast).
        push = si.TokenStream._push
        close_returned = threading.Event()
        late = []

        def slow_push(self, t):
            time.sleep(0.002)
            if close_returned.is_set():  # the C call is still delivering chunks
                late.append(t)
            return push(self, t)

        si.TokenStream._push = slow_push
        e, m = make()
        s = e.create_session(m, si.SamplingConfig.greedy(2000))
        it = s.stream("long stream")
        chunks = [next(it)]

        def close_it():
            s.close()
            close_returned.set()

        closer = threading.Thread(target=close_it)
        closer.start()
        for c in it:  # drains until the cancelled generation finishes
            chunks.append(c)
            time.sleep(0.001)
        closer.join(30)
        assert not closer.is_alive()
        assert not late, f"{len(late)} chunk(s) delivered after close() returned"
        assert it.result is not None and it.result.cancelled
        assert len(chunks) < 2000
        assert s.closed
        e.close()
    """)


def test_concurrent_close_calls_are_safe(lib):
    _run("""
        e, m = make()
        s = e.create_session(m, si.SamplingConfig.greedy(2000))
        t, cb_exited, box = slow_first_chunk_worker(s)
        closers = [threading.Thread(target=s.close) for _ in range(4)] + [threading.Thread(target=e.close)]
        for c in closers:
            c.start()
        for c in closers:
            c.join(30)
            assert not c.is_alive()
        assert cb_exited.is_set()
        t.join(30)
        assert "error" not in box, box
        assert s.closed and e.closed
    """)


def test_close_from_own_callback_is_rejected_not_deadlocked(engine, model):
    s = engine.create_session(model, si.SamplingConfig.greedy(50))
    seen = []

    def on_token(_t):
        seen.append(_t)
        s.close()

    with pytest.raises(si.InvalidStateError, match="callback"):
        s.generate("self close", on_token=on_token)
    assert len(seen) == 1 and not s.closed
    assert s.generate("still usable").completed
    s.close()
