"""Request parent lineage through the real shared library and bounded iterators."""
from dataclasses import FrozenInstanceError
import gc
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import weakref

import pytest

import sonder_inference as si
from sonder_inference._lib import Library


METHODS = ('generate', 'chat', 'stream', 'chat_stream')
PRIVATE = 'request-parent-private-prompt-canary'


def request(session, method, metadata=None):
    value = [si.ChatMessage('user', PRIVATE)] if 'chat' in method else PRIVATE
    result = getattr(session, method)(value, metadata=metadata)
    if 'stream' in method:
        with result as stream:
            chunks = list(stream)
        assert ''.join(chunks) == stream.result.text
        assert not stream._thread.is_alive()
        return stream.result
    return result


@pytest.mark.parametrize('capture_text', [False, True])
def test_parent_is_request_local_across_all_methods_and_default_reuse(tmp_path, capture_text):
    path = tmp_path / 'parents.jsonl'
    expected = []
    outputs = []
    identity = si.SessionMetadata(session_id='parent-sdk:1', run_id='parent-run:1',
                                  agent_id='parent-agent:1', task_id='parent-task:1')
    with si.Engine(telemetry_level=si.TelemetryLevel.DEEP, telemetry_jsonl_path=str(path),
                   capture_text=capture_text) as engine:
        engine.register_mock_backend()
        with engine.load_model('mock', 'mock:tiny') as model:
            with engine.create_session(model, si.SamplingConfig.greedy(4), metadata=identity) as session:
                for method in METHODS:
                    for metadata in (si.RequestMetadata(f'parent:{method}'), si.RequestMetadata(), None):
                        expected.append(metadata.parent_request_id if metadata else None)
                        result = request(session, method, metadata)
                        assert result.completed and result.completion_tokens == 4
                        outputs.append(result.text)
    raw = path.read_text()
    assert PRIVATE not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    queued = [event for event in events if event['event_type'] == 'request.queued']
    assert len(queued) == 12
    parents = {event['request_id']: parent for event, parent in zip(queued, expected)}
    assert len(parents) == 12 and all(key.startswith('req-') for key in parents)
    lifecycle = [event for event in events if event['event_type'].startswith('request.')]
    assert len(lifecycle) == 36
    for event in events:
        attrs = event['attributes']
        if event['event_type'].startswith('request.'):
            parent = parents[event['request_id']]
            assert attrs.get('parent_request_id') == parent
            assert ('parent_request_id' in attrs) == (parent is not None)
        else:
            assert 'parent_request_id' not in attrs
        if event.get('session_id') == identity.session_id:
            assert all(event.get(key) == value for key, value in identity.__dict__.items())
            assert event['producer'].get('synthetic') is True
    tokens = [event for event in events if event['event_type'] == 'inference.token.generated']
    assert len(tokens) == 48
    assert all(('text' in event['attributes']) == capture_text for event in tokens)
    if capture_text:
        assert ''.join(event['attributes']['text'] for event in tokens) == ''.join(outputs)
    assert [event['sequence'] for event in events] == list(range(len(events)))
    assert len({event['event_id'] for event in events}) == len(events)
    assert len({event['producer']['instance_id'] for event in events}) == 1


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('value', ['', 'a' * 129, 'x\0y', 'private/id', 'héllo', 'line\nbreak', 7])
def test_invalid_metadata_fails_eagerly_without_worker_callback_or_submission(engine, model, monkeypatch,
                                                                            method, value):
    with engine.create_session(model) as session:
        def forbidden(*args, **kwargs):
            raise AssertionError('invalid metadata entered worker/native request')
        monkeypatch.setattr(session, '_request', forbidden)
        with pytest.raises(si.InvalidArgumentError) as error:
            request(session, method, si.RequestMetadata(value))
        if isinstance(value, str) and value:
            assert value not in str(error.value)


@pytest.mark.parametrize('method', METHODS)
def test_wrong_metadata_type_and_boundary(engine, model, method):
    with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
        with pytest.raises(si.InvalidArgumentError, match='RequestMetadata'):
            request(session, method, {'parent_request_id': 'untyped'})
        assert request(session, method, si.RequestMetadata('a' * 128)).completed


def test_request_metadata_is_frozen():
    metadata = si.RequestMetadata('parent:original')
    with pytest.raises(FrozenInstanceError):
        metadata.parent_request_id = 'parent:changed'


@pytest.mark.parametrize('method', ['stream', 'chat_stream'])
def test_metadata_stream_closed_session_rejects_before_worker(engine, model, monkeypatch, method):
    session = engine.create_session(model)
    session.close()
    def forbidden(*args, **kwargs):
        raise AssertionError('closed session started a worker')
    monkeypatch.setattr(si.TokenStream, '_start', forbidden)
    with pytest.raises(si.InvalidStateError, match='closed'):
        request(session, method, si.RequestMetadata('parent:closed'))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('metadata', [si.RequestMetadata(), si.RequestMetadata('parent:explicit')])
def test_missing_additive_exports_are_explicitly_unsupported_before_worker(lib, method, metadata, monkeypatch):
    class WithoutRequestMetadata:
        def __getattr__(self, name):
            if name in ('sonder_session_generate_with_metadata', 'sonder_session_chat_with_metadata'):
                raise AttributeError(name)
            return getattr(lib.cdll, name)
    legacy = Library(WithoutRequestMetadata(), lib.path)
    with si.Engine(library=legacy, telemetry_level=si.TelemetryLevel.OFF) as engine:
        engine.register_mock_backend()
        with engine.load_model('mock', 'mock:tiny') as model:
            with engine.create_session(model, si.SamplingConfig.greedy(2)) as session:
                assert request(session, method).completed
                def forbidden(*args, **kwargs):
                    raise AssertionError('unsupported metadata entered request/worker')
                monkeypatch.setattr(session, '_request', forbidden)
                with pytest.raises(si.UnsupportedError, match='with_metadata'):
                    request(session, method, metadata)


@pytest.mark.parametrize('method', ['stream', 'chat_stream'])
def test_iterator_owns_prepared_parent_snapshot_until_native_return(engine, model, tmp_path, method):
    path = tmp_path / 'snapshot.jsonl'
    with si.Engine(telemetry_level=si.TelemetryLevel.DEEP, telemetry_jsonl_path=str(path)) as telemetry_engine:
        telemetry_engine.register_mock_backend()
        with telemetry_engine.load_model('mock', 'mock:tiny') as telemetry_model:
            with telemetry_engine.create_session(telemetry_model, si.SamplingConfig.greedy(2)) as session:
                prepare, native_request = session._request_args, session._request
                entered, release = threading.Event(), threading.Event()
                refs = []
                def observed_prepare(*args):
                    function, prepared = prepare(*args)
                    refs.append(weakref.ref(prepared[-1]))
                    return function, prepared
                def gated_request(*args, **kwargs):
                    entered.set()
                    assert release.wait(10)
                    return native_request(*args, **kwargs)
                session._request_args, session._request = observed_prepare, gated_request
                metadata = si.RequestMetadata('parent:original')
                value = [{'role': 'user', 'content': PRIVATE}] if 'chat' in method else PRIVATE
                stream = getattr(session, method)(value, metadata=metadata)
                try:
                    assert entered.wait(10)
                    assert refs[0]() is not None
                    # Deliberately bypass frozen API to prove worker preparation is a snapshot.
                    object.__setattr__(metadata, 'parent_request_id', 'parent:changed')
                    if isinstance(value, list):
                        value[0]['content'] = 'changed conversation'
                        value.clear()
                    del metadata
                    gc.collect()
                finally:
                    release.set()
                with stream:
                    assert len(list(stream)) == 2 and stream.result.completed
                    stream._thread.join(5)
                    assert not stream._thread.is_alive()
                    assert refs[0]() is None
    queued = [json.loads(line) for line in path.read_text().splitlines()
              if json.loads(line)['event_type'] == 'request.queued']
    assert len(queued) == 1
    assert queued[0]['attributes']['parent_request_id'] == 'parent:original'


@pytest.fixture
def failing_backend_url():
    show = (Path(__file__).resolve().parents[3] /
            'src/backends/ollama/tests/fixtures/show.json').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            if self.path == '/api/show':
                response = show
            elif self.path in ('/api/generate', '/api/chat'):
                records = [{'message': {'content': text}, 'done': False} if self.path == '/api/chat'
                           else {'response': text, 'done': False} for text in ('α ', '✓ ')]
                records.append({'error': 'synthetic transport failure'})
                response = b''.join((json.dumps(record, ensure_ascii=False) + '\n').encode() for record in records)
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json' if self.path == '/api/show' else 'application/x-ndjson')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


@pytest.mark.parametrize('method', METHODS)
def test_actual_native_backend_failure_retains_parent_and_drains_delivered_chunks(lib, tmp_path,
                                                                               failing_backend_url, method):
    path = tmp_path / 'failure.jsonl'
    chunks = []
    with si.Engine(telemetry_level=si.TelemetryLevel.DEEP, telemetry_jsonl_path=str(path)) as engine:
        engine.register_ollama_backend(failing_backend_url)
        with engine.load_model('ollama', 'synthetic-error-stream') as model:
            with engine.create_session(model, metadata=si.SessionMetadata(run_id='failure-run:1')) as session:
                value = [si.ChatMessage('user', PRIVATE)] if 'chat' in method else PRIVATE
                metadata = si.RequestMetadata('parent:failed')
                if 'stream' in method:
                    with getattr(session, method)(value, metadata=metadata) as stream:
                        with pytest.raises(si.BackendError, match='synthetic transport failure'):
                            while True:
                                chunks.append(next(stream))
                    assert not stream._thread.is_alive() and stream.result is None
                else:
                    with pytest.raises(si.BackendError, match='synthetic transport failure'):
                        getattr(session, method)(value, chunks.append, metadata=metadata)
    assert ''.join(chunks) == 'α ✓ '
    raw = path.read_text()
    assert PRIVATE not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    lifecycle = [event for event in events if event['event_type'].startswith('request.')]
    assert [event['event_type'] for event in lifecycle] == ['request.queued', 'request.started', 'request.failed']
    assert all(event['attributes']['parent_request_id'] == 'parent:failed' for event in lifecycle)
    assert all(event['run_id'] == 'failure-run:1' for event in lifecycle)


@pytest.mark.parametrize('method', ['generate', 'chat'])
def test_actual_native_cancellation_carries_parent_and_preserves_reuse(lib, tmp_path, method):
    path = tmp_path / 'cancel.jsonl'
    with si.Engine(telemetry_level=si.TelemetryLevel.DEEP, telemetry_jsonl_path=str(path)) as engine:
        engine.register_mock_backend()
        with engine.load_model('mock', 'mock:tiny') as model:
            with engine.create_session(model, si.SamplingConfig.greedy(8)) as session:
                value = [si.ChatMessage('user', PRIVATE)] if method == 'chat' else PRIVATE
                def cancel(_text):
                    session.cancel()
                result = getattr(session, method)(value, cancel, metadata=si.RequestMetadata('parent:cancelled'))
                assert result.cancelled and result.completion_tokens < 8
                assert session.generate('reused without parent').completed
    lifecycle = [json.loads(line) for line in path.read_text().splitlines()
                 if json.loads(line)['event_type'].startswith('request.')]
    assert [event['event_type'] for event in lifecycle] == ['request.queued', 'request.started', 'request.cancelled',
                                                           'request.queued', 'request.started', 'request.completed']
    assert all(event['attributes'].get('parent_request_id') == 'parent:cancelled' for event in lifecycle[:3])
    assert all('parent_request_id' not in event['attributes'] for event in lifecycle[3:])
