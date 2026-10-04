"""Actual SDK mock work must carry synthetic provenance into Observatory."""
import json

import pytest

import sonder_inference as si


@pytest.mark.parametrize('level', [si.TelemetryLevel.METRICS,
                                 si.TelemetryLevel.STANDARD, si.TelemetryLevel.DEEP])
def test_sdk_mock_provenance_survives_generate_chat_and_stream(lib, tmp_path, level):
    path = tmp_path / 'sdk-mock.jsonl'
    private = 'mock-provenance-private-canary'
    ids = si.SessionMetadata(session_id='mock-provenance-session',
                             run_id='mock-provenance-run', agent_id='mock-provenance-agent',
                             task_id='mock-provenance-task')
    with si.Engine(telemetry_level=level, telemetry_jsonl_path=str(path),
                   capture_text=False) as engine:
        engine.register_mock_backend()
        with engine.load_model('mock', 'mock:tiny') as model:
            with engine.create_session(model, si.SamplingConfig.greedy(4), metadata=ids) as session:
                assert session.generate(private).completion_tokens == 4
                assert session.chat([si.ChatMessage('user', private)]).completion_tokens == 4
                with session.stream(private) as stream:
                    assert len(list(stream)) == 4
    raw = path.read_text()
    assert private not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    scoped = [event for event in events if event['session_id'] == ids.session_id]
    assert len([event for event in scoped if event['event_type'] == 'request.completed']) == 3
    assert scoped and all(event['producer'].get('synthetic') is True for event in scoped)
    for event in scoped:
        assert event['run_id'] == ids.run_id
        assert event['agent_id'] == ids.agent_id
        assert event['task_id'] == ids.task_id
    backend_events = [event for event in events if event['attributes'].get('backend') == 'mock']
    assert backend_events and all(event['producer'].get('synthetic') is True for event in backend_events)
    # Engine/device observations do not independently identify backend work.
    for event in events:
        if event['event_type'] in ('engine.started', 'engine.stopped', 'device.memory.sample'):
            assert 'synthetic' not in event['producer']
    sequences = [event['sequence'] for event in events]
    assert sequences == sorted(set(sequences))
    assert len({event['event_id'] for event in events}) == len(events)
    assert all('text' not in event['attributes'] for event in events
               if event['event_type'] == 'inference.token.generated')
