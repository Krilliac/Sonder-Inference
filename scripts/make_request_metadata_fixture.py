#!/usr/bin/env python3
"""Actual native mock parent lineage JSONL for consumer qualification, no quality claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bindings/python/src'))
import sonder_inference as si  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    library = args.library.resolve(strict=True)
    si.load_library(str(library))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    private = 'request-parent-private-prompt-canary'
    receipts = []
    for capture in (False, True):
        path = args.out_dir / f'parent-capture-{str(capture).lower()}.jsonl'
        identity = si.SessionMetadata(session_id=f'fixture-session:{int(capture)}', run_id='fixture-run:1',
                                      agent_id='fixture-agent:1', task_id='fixture-task:1')
        expected = []
        results = []
        with si.Engine(telemetry_level=si.TelemetryLevel.DEEP, telemetry_jsonl_path=str(path),
                       capture_text=capture) as engine:
            engine.register_mock_backend()
            with engine.load_model('mock', 'mock:tiny') as model:
                with engine.create_session(model, si.SamplingConfig.greedy(4), metadata=identity) as session:
                    for method in ('generate', 'chat', 'stream', 'chat_stream', 'generate', 'chat_stream'):
                        index = len(expected)
                        metadata = (si.RequestMetadata(f'fixture-parent:{index}') if index < 4
                                    else si.RequestMetadata() if index == 4 else None)
                        expected.append(metadata.parent_request_id if metadata else None)
                        value = [si.ChatMessage('user', private)] if 'chat' in method else private
                        result = getattr(session, method)(value, metadata=metadata)
                        if 'stream' in method:
                            with result as stream:
                                chunks = list(stream)
                            assert ''.join(chunks) == stream.result.text
                            result = stream.result
                            assert not stream._thread.is_alive()
                        assert result.completed and result.completion_tokens == 4
                        results.append(result.text)
        raw = path.read_text()
        assert private not in raw
        events = [json.loads(line) for line in raw.splitlines()]
        queued = [event for event in events if event['event_type'] == 'request.queued']
        assert len(queued) == 6
        parents = {event['request_id']: parent for event, parent in zip(queued, expected)}
        assert len(parents) == 6
        lifecycle = [event for event in events if event['event_type'].startswith('request.')]
        assert len(lifecycle) == 18
        for event in lifecycle:
            parent = parents[event['request_id']]
            assert event['attributes'].get('parent_request_id') == parent
            assert ('parent_request_id' in event['attributes']) == (parent is not None)
        for event in events:
            if event['session_id'] == identity.session_id:
                assert all(event.get(key) == value for key, value in identity.__dict__.items())
                assert event['producer'].get('synthetic') is True
        tokens = [event for event in events if event['event_type'] == 'inference.token.generated']
        assert len(tokens) == 24
        assert all(('text' in event['attributes']) == capture for event in tokens)
        if capture:
            assert ''.join(event['attributes']['text'] for event in tokens) == ''.join(results)
        assert [event['sequence'] for event in events] == list(range(len(events)))
        producer = events[0]['producer']['instance_id']
        assert all(event['producer']['instance_id'] == producer for event in events)
        assert all(event['event_id'] == f"{producer}-{event['sequence']}" for event in events)
        receipts.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'events': len(events), 'requests': len(queued), 'token_events': len(tokens),
                         'capture_text': capture, 'session_metadata': identity.__dict__, 'request_parents': parents,
                         'private_prompt_absent': True, 'producer_instance_id': producer})
    receipt = {'status': 'native-producer-qualified-consumer-pending', 'synthetic': True,
               'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(), 'fixtures': receipts}
    (args.out_dir / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
