"""Close ordering at a fake external-call boundary; no native library loaded."""
from __future__ import annotations

import threading
import types
import weakref

import pytest

from sonder_inference.api import Engine, InvalidStateError, Model, Session, _Handle


class FakeHandle(_Handle):
    _destroy_fn = "destroy"

    def __init__(self, destroy, interrupt=lambda _handle: None):
        self.interrupt = interrupt
        super().__init__(types.SimpleNamespace(cdll=types.SimpleNamespace(destroy=destroy)), 1)

    def _interrupt(self, handle):
        self.interrupt(handle)


def spawn(function):
    errors = []

    def run():
        try:
            function()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, errors


def joined(worker):
    thread, errors = worker
    thread.join(3)
    assert not thread.is_alive()
    assert not errors


def test_concurrent_close_waits_for_external_interrupt():
    active = threading.Event()
    finish_call = threading.Event()
    interrupt_entered = threading.Event()
    release_interrupt = threading.Event()
    second_started = threading.Event()
    second_returned = threading.Event()
    trace = []

    def interrupt(_handle):
        trace.append("interrupt-start")
        interrupt_entered.set()
        assert release_interrupt.wait(3)
        trace.append("interrupt-end")

    handle = FakeHandle(lambda _handle: trace.append("destroy"), interrupt)

    def call():
        with handle._use():
            active.set()
            assert finish_call.wait(3)

    def second_close():
        second_started.set()
        handle.close()
        second_returned.set()

    workers = []
    try:
        workers.append(spawn(call))
        assert active.wait(3)
        workers.append(spawn(handle.close))
        assert interrupt_entered.wait(3)
        finish_call.set()
        joined(workers[0])
        workers.append(spawn(second_close))
        assert second_started.wait(3)
        assert not second_returned.wait(0.1)
    finally:
        finish_call.set()
        release_interrupt.set()
        for worker in workers:
            joined(worker)
        handle.close()
    assert trace == ["interrupt-start", "interrupt-end", "destroy"]
    assert handle._ptr is None


def test_all_close_callers_wait_for_one_external_destroy():
    entered = threading.Event()
    release = threading.Event()
    returned = [threading.Event() for _ in range(4)]
    started = [threading.Event() for _ in range(4)]
    destroys = []

    def destroy(_handle):
        destroys.append("start")
        entered.set()
        assert release.wait(3)
        destroys.append("end")

    handle = FakeHandle(destroy)

    def close(index):
        started[index].set()
        handle.close()
        returned[index].set()

    workers = []
    try:
        workers.append(spawn(lambda: close(0)))
        assert entered.wait(3)
        workers.extend(spawn(lambda i=i: close(i)) for i in range(1, 4))
        assert all(event.wait(3) for event in started)
        assert handle.closed
        assert not any(event.wait(0.02) for event in returned)
    finally:
        release.set()
        for worker in workers:
            joined(worker)
        handle.close()
    assert destroys == ["start", "end"]
    assert all(event.is_set() for event in returned)
    assert handle._ptr is None


def test_engine_close_waits_for_child_destroy_in_session_model_engine_order():
    entered = threading.Event()
    release = threading.Event()
    engine_returned = threading.Event()
    engine_started = threading.Event()
    trace = []

    def session_destroy(_handle):
        trace.append("session-start")
        entered.set()
        assert release.wait(3)
        trace.append("session-end")

    lib = types.SimpleNamespace(cdll=types.SimpleNamespace(
        sonder_session_destroy=session_destroy,
        sonder_model_release=lambda _handle: trace.append("model"),
        sonder_engine_destroy=lambda _handle: trace.append("engine")))
    engine = Engine.__new__(Engine)
    _Handle.__init__(engine, lib, 1)
    model = Model.__new__(Model)
    _Handle.__init__(model, lib, 2)
    session = Session.__new__(Session)
    _Handle.__init__(session, lib, 3)
    engine._children = weakref.WeakSet([model, session])

    def engine_close():
        engine_started.set()
        engine.close()
        engine_returned.set()

    workers = []
    try:
        workers.append(spawn(session.close))
        assert entered.wait(3)
        workers.append(spawn(engine_close))
        assert engine_started.wait(3)
        assert not engine_returned.wait(0.1)
    finally:
        release.set()
        for worker in workers:
            joined(worker)
        engine.close()
    assert trace == ["session-start", "session-end", "model", "engine"]
    assert engine._ptr is model._ptr is session._ptr is None


def test_reentrant_close_rejection_does_not_begin_closure():
    trace = []
    handle = FakeHandle(lambda _handle: trace.append("destroy"), lambda _handle: trace.append("interrupt"))
    try:
        with handle._use():
            with pytest.raises(InvalidStateError, match="inside one of its own calls"):
                handle.close()
            assert not handle.closed
            assert not trace
        handle.close()
        assert trace == ["destroy"]
    finally:
        handle.close()


@pytest.mark.parametrize("closers", [1, 4, 8])
def test_repeated_close_contention_preserves_external_call_order(closers):
    # Exercise 96 resource lifetimes and 416 close callers across this matrix.
    # These are Python wrapper/mock-boundary controls, not native throughput.
    for _ in range(32):
        active = threading.Event()
        finish_call = threading.Event()
        start_closing = threading.Event()
        trace = []

        def interrupt(_handle):
            trace.append("interrupt")
            finish_call.set()

        handle = FakeHandle(lambda _handle: trace.append("destroy"), interrupt)

        def call():
            with handle._use():
                active.set()
                assert finish_call.wait(3)

        def close():
            assert start_closing.wait(3)
            handle.close()

        workers = []
        try:
            workers.append(spawn(call))
            assert active.wait(3)
            workers.extend(spawn(close) for _ in range(closers))
            start_closing.set()
            for worker in workers:
                joined(worker)
            assert trace.count("destroy") == 1
            assert trace[-1] == "destroy"
            assert handle._ptr is None and not handle._active
            with pytest.raises(InvalidStateError, match="is closed"):
                with handle._use():
                    pytest.fail("a closed handle admitted another call")
        finally:
            finish_call.set()
            start_closing.set()
            for worker in workers:
                joined(worker)
            handle.close()
