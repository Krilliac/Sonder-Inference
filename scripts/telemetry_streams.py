"""Per-stream checks shared by the telemetry qualification harnesses.

An engine numbers each sequence stream on its own: the main stream
(`producer.stream` absent) uses `<instance>-<n>` event ids, and the periodic
`samples` stream (`producer.stream == "samples"`) uses `<instance>-s<n>`.
"""
from __future__ import annotations

from typing import Any


def check_event_streams(events: list[dict[str, Any]], *, contiguous: bool = True) -> str:
    """Assert each stream is a contiguous 0..n-1 run (strictly increasing when
    `contiguous` is false) with event ids matching the stream's format.

    Returns the single producer instance id shared by every event.
    """
    instances = {event['producer']['instance_id'] for event in events}
    assert len(instances) == 1, instances
    instance = next(iter(instances))
    by_stream: dict[str | None, list[dict[str, Any]]] = {}
    for event in events:
        by_stream.setdefault(event['producer'].get('stream'), []).append(event)
    assert set(by_stream) <= {None, 'samples'}, set(by_stream)
    for stream, group in by_stream.items():
        sequences = [event['sequence'] for event in group]
        if contiguous:
            assert sequences == list(range(len(group))), (stream, sequences[:16])
        else:
            assert sequences == sorted(set(sequences)), (stream, sequences[:16])
        prefix = f'{instance}-s' if stream == 'samples' else f'{instance}-'
        assert all(event['event_id'] == f"{prefix}{event['sequence']}" for event in group), stream
    assert len({event['event_id'] for event in events}) == len(events)
    return instance
