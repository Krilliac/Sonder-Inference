"""Session correlation must reach real producer envelopes without text capture."""
import json

import pytest

import sonder_inference as si


def test_session_metadata_correlates_generate_chat_and_stream(lib, tmp_path):
    path = tmp_path / 'metadata.jsonl'
    metadata = si.SessionMetadata(session_id='sdk-session:1', run_id='sdk-run:1',
                                  agent_id='sdk-agent:1', task_id='sdk-task:1')
    private = 'metadata-private-prompt-canary'
    with si.Engine(telemetry_level=si.TelemetryLevel.DEEP, telemetry_jsonl_path=str(path),
                   capture_text=False) as engine:
        engine.register_mock_backend()
        with engine.load_model('mock', 'mock:tiny') as model:
            with engine.create_session(model, si.SamplingConfig.greedy(4), metadata=metadata) as session:
                assert session.generate(private).completed
                assert session.chat([si.ChatMessage('user', private)]).completed
                with session.stream(private) as stream:
                    assert len(list(stream)) == 4
    raw = path.read_text()
    assert private not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    session_events = [event for event in events if event.get('session_id') == metadata.session_id]
    assert session_events
    assert len([event for event in session_events if event['event_type'] == 'request.queued']) == 3
    for event in session_events:
        assert event['run_id'] == metadata.run_id
        assert event['agent_id'] == metadata.agent_id
        assert event['task_id'] == metadata.task_id
    sequences = [event['sequence'] for event in events]
    assert sequences == sorted(set(sequences))
    assert len({event['event_id'] for event in events}) == len(events)
    assert all('text' not in event['attributes'] for event in events
               if event['event_type'] == 'inference.token.generated')


@pytest.mark.parametrize('value', ['', 'a' * 129, 'x\x00y', 'private/id', 'héllo', 'line\nbreak'])
def test_metadata_rejects_invalid_identifiers_without_echoing_value(engine, model, value):
    with pytest.raises(si.InvalidArgumentError) as error:
        engine.create_session(model, metadata=si.SessionMetadata(run_id=value))
    if value:
        assert value not in str(error.value)
    with engine.create_session(model) as session:
        assert session.generate('default after rejection').completed


def test_metadata_boundary_and_default_optional_fields(engine, model):
    with engine.create_session(model, metadata=si.SessionMetadata(run_id='a' * 128)) as session:
        assert session.generate('boundary').completed
    with engine.create_session(model, metadata=si.SessionMetadata()) as session:
        assert session.generate('all defaults').completed


def test_legacy_library_defaults_work_and_requested_metadata_is_explicitly_unsupported(lib):
    from sonder_inference._lib import Library

    class WithoutMetadata:
        def __getattr__(self, name):
            if name == 'sonder_session_create_with_metadata':
                raise AttributeError(name)
            return getattr(lib.cdll, name)

    legacy = Library(WithoutMetadata(), lib.path)
    with si.Engine(library=legacy, telemetry_level=si.TelemetryLevel.OFF) as engine:
        engine.register_mock_backend()
        with engine.load_model('mock', 'mock:tiny') as model:
            with engine.create_session(model, si.SamplingConfig.greedy(4)) as session:
                assert session.generate('legacy default').completion_tokens == 4
                assert session.chat([si.ChatMessage('user', 'legacy chat')]).completion_tokens == 4
            with pytest.raises(si.UnsupportedError, match='sonder_session_create_with_metadata'):
                engine.create_session(model, metadata=si.SessionMetadata(run_id='explicit-run'))


def test_metadata_type_and_model_ownership_remain_checked(engine, model):
    with pytest.raises(si.InvalidArgumentError, match='SessionMetadata'):
        engine.create_session(model, metadata={'run_id': 'untyped'})
    with si.Engine(telemetry_level=si.TelemetryLevel.OFF) as other:
        with pytest.raises(si.InvalidArgumentError, match='different engine'):
            other.create_session(model, metadata=si.SessionMetadata(run_id='run'))
