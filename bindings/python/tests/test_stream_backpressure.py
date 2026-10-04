"""Real shared-library stream delivery and stalled-consumer lifecycle checks."""
from __future__ import annotations

import textwrap
import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from test_close_race import _run


@pytest.fixture
def long_backend_url():
    show = (Path(__file__).resolve().parents[3] /
            "src/backends/ollama/tests/fixtures/show.json").read_bytes()
    records = [{"response": f"{i}✓ ", "done": False} for i in range(2048)]
    records.append({"response": "", "done": True, "prompt_eval_count": 3, "eval_count": 2048})
    response = b"".join((json.dumps(x, ensure_ascii=False) + "\n").encode() for x in records)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if self.path not in ("/api/show", "/api/generate"):
                self.send_error(404)
                return
            data = show if self.path == "/api/show" else response
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if self.path == "/api/show" else "application/x-ndjson")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # a deliberately cancelled native client

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


def _native_make(url):
    return f"""
        def make():
            e = si.Engine(telemetry_level=si.TelemetryLevel.OFF)
            e.register_ollama_backend('{url}')
            return e, e.load_model('ollama', 'synthetic-long-stream')
    """


_WAIT_FULL = """
    deadline = time.monotonic() + 10
    while it._queue.qsize() < 64 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert it._queue.qsize() >= 64, 'producer did not fill the buffer'
    it._thread.join(0.1)
    assert it._queue.qsize() <= 64, f'unbounded backlog: {it._queue.qsize()}'
    assert it._thread.is_alive(), 'producer should wait for the paused consumer'
    assert it.result is None
"""
_WAIT_FULL = textwrap.indent(textwrap.dedent(_WAIT_FULL), "        ")


def test_stalled_consumer_bounds_backlog_and_drains_without_loss(lib, long_backend_url):
    _run(_native_make(long_backend_url) + """
        e, m = make()
        s = e.create_session(m, si.SamplingConfig.greedy(2048))
        it = s.stream('synthetic bounded delivery')
    """ + _WAIT_FULL + """
        chunks = list(it)
        assert len(chunks) == 2048
        assert ''.join(chunks) == it.result.text == ''.join(f'{i}✓ ' for i in range(2048))
        assert it.result.completed and it.result.completion_tokens == 2048
        assert not it._thread.is_alive()
        assert s.generate('reuse').completed
        e.close()
    """)


@pytest.mark.parametrize('action', ['stream.close', 'session.cancel', 'session.close', 'engine.close'])
def test_stalled_consumer_lifecycle_never_waits_for_drain(lib, action, long_backend_url):
    calls = {'stream.close': 'it.close()', 'session.cancel': 's.cancel()',
             'session.close': 's.close()', 'engine.close': 'e.close()'}
    _run(_native_make(long_backend_url) + """
        e, m = make()
        s = e.create_session(m, si.SamplingConfig.greedy(2048))
        it = s.stream('synthetic paused cancellation')
    """ + _WAIT_FULL + f"""
        started = time.monotonic()
        {calls[action]}
        it._thread.join(5)
        assert not it._thread.is_alive(), 'worker still waiting for the consumer'
        assert time.monotonic() - started < 5
        assert it.result is not None and it.result.cancelled
        if '{action}' == 'stream.close':
            assert list(it) == []
        else:
            chunks = list(it)
            assert len(chunks) <= 64
            assert it.result.text.startswith(''.join(chunks))
        if not s.closed:
            assert s.generate('reuse after cancellation').completed
        e.close()
    """)


def test_stream_error_after_queued_chunks_is_propagated(lib):
    _run("""
        e, m = make()
        s = e.create_session(m)
        def failing_generate(prompt, on_token=None):
            for i in range(100):
                if on_token(str(i)) is False:
                    return
            raise ValueError('synthetic backend failure')
        s.generate = failing_generate
        it = s.stream('synthetic worker failure')
        chunks = []
        try:
            while True:
                chunks.append(next(it))
        except ValueError as exc:
            assert str(exc) == 'synthetic backend failure'
        else:
            raise AssertionError('worker failure was hidden')
        assert chunks == [str(i) for i in range(100)]
        assert not it._thread.is_alive()
        it.close()
        e.close()
    """)


@pytest.mark.parametrize('action', ['cancel', 'close'])
def test_stream_stop_before_native_request_enters(lib, action):
    _run(f"""
        e, m = make()
        s = e.create_session(m, si.SamplingConfig.greedy(12))
        original = s.generate
        entered, release = threading.Event(), threading.Event()
        def gated_generate(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)
        s.generate = gated_generate
        it = s.stream('synthetic start race')
        assert entered.wait(10)
        if '{action}' == 'cancel':
            s.cancel()
            release.set()
            list(it)
        else:
            closer = threading.Thread(target=it.close)
            closer.start()
            assert it._closed.wait(10)
            release.set()
            closer.join(5)
            assert not closer.is_alive()
        it._thread.join(5)
        assert not it._thread.is_alive()
        assert it.result is not None and it.result.cancelled
        s.generate = original
        assert s.generate('reuse').completed
        e.close()
    """)
