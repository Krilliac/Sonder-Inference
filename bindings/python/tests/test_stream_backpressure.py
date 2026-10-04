"""Real shared-library stream delivery and stalled-consumer lifecycle checks."""
from __future__ import annotations

import textwrap
import json
import queue
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from test_close_race import _run


@pytest.fixture
def long_backend():
    show = (Path(__file__).resolve().parents[3] /
            "src/backends/ollama/tests/fixtures/show.json").read_bytes()
    records = [{"response": f"{i}✓ ", "done": False} for i in range(2048)]
    records.append({"response": "", "done": True, "prompt_eval_count": 3, "eval_count": 2048})
    response = b"".join((json.dumps(x, ensure_ascii=False) + "\n").encode() for x in records)
    chat_records = [{"message": {"role": "assistant", "content": f"{i}✓ "}, "done": False}
                    for i in range(2048)]
    chat_records.append({"message": {"content": ""}, "done": True, "prompt_eval_count": 3, "eval_count": 2048})
    chat_response = b"".join((json.dumps(x, ensure_ascii=False) + "\n").encode() for x in chat_records)
    requests = queue.Queue(maxsize=8)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path not in ("/api/show", "/api/generate", "/api/chat"):
                self.send_error(404)
                return
            if self.path == "/api/chat":
                try:
                    requests.put_nowait(json.loads(body))
                except queue.Full:
                    pass  # bounded test observation, never block the native client
            data = show if self.path == "/api/show" else chat_response if self.path == "/api/chat" else response
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
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


@pytest.fixture
def long_backend_url(long_backend):
    return long_backend[0]


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


def _iterator(mode, prompt):
    return f"s.chat_stream([si.ChatMessage('user', {prompt!r})])" if mode == 'chat' else f"s.stream({prompt!r})"


@pytest.mark.parametrize('mode', ['generate', 'chat'])
def test_stalled_consumer_bounds_backlog_and_drains_without_loss(lib, long_backend_url, mode):
    _run(_native_make(long_backend_url) + f"""
        e, m = make()
        s = e.create_session(m, si.SamplingConfig.greedy(2048))
        it = {_iterator(mode, 'synthetic bounded delivery')}
    """ + _WAIT_FULL + """
        chunks = list(it)
        assert len(chunks) == 2048
        assert ''.join(chunks) == it.result.text == ''.join(str(i) + '✓ ' for i in range(2048))
        assert it.result.completed and it.result.completion_tokens == 2048
        assert not it._thread.is_alive()
        assert s.generate('reuse').completed
        e.close()
    """)


@pytest.mark.parametrize('action', ['stream.close', 'session.cancel', 'session.close', 'engine.close'])
@pytest.mark.parametrize('mode', ['generate', 'chat'])
def test_stalled_consumer_lifecycle_never_waits_for_drain(lib, action, long_backend_url, mode):
    calls = {'stream.close': 'it.close()', 'session.cancel': 's.cancel()',
             'session.close': 's.close()', 'engine.close': 'e.close()'}
    _run(_native_make(long_backend_url) + f"""
        e, m = make()
        s = e.create_session(m, si.SamplingConfig.greedy(2048))
        it = {_iterator(mode, 'synthetic paused cancellation')}
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


@pytest.mark.parametrize('mode', ['generate', 'chat'])
def test_stream_error_after_queued_chunks_is_propagated(lib, mode):
    _run(f"""
        e, m = make()
        s = e.create_session(m)
        def failing_request(function, args, on_token=None):
            for i in range(100):
                if on_token(str(i)) is False:
                    return
            raise ValueError('synthetic backend failure')
        s._request = failing_request
        it = {_iterator(mode, 'synthetic worker failure')}
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
@pytest.mark.parametrize('mode', ['generate', 'chat'])
def test_stream_stop_before_native_request_enters(lib, action, mode):
    _run(f"""
        e, m = make()
        s = e.create_session(m, si.SamplingConfig.greedy(12))
        original = s._request
        entered, release = threading.Event(), threading.Event()
        def gated_request(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)
        s._request = gated_request
        it = {_iterator(mode, 'synthetic start race')}
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
        s._request = original
        assert s.generate('reuse').completed
        e.close()
    """)
