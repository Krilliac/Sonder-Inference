#!/usr/bin/env python3
"""Bounded mixed SDK parent-lineage stability qualification; synthetic work only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bindings/python/src'))
import sonder_inference as si  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=8)
    parser.add_argument('--requests-per-worker', type=int, default=16)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fixture-dir', type=Path)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 16 or not 1 <= args.requests_per_worker <= 32:
        parser.error('cycles must be 1..16 and requests-per-worker 1..32')
    library = args.library.resolve(strict=True)
    si.load_library(str(library))
    root = Path(__file__).resolve().parents[1]
    private = 'request-parent-private-prompt-canary'
    scenarios = []
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='sonder-request-parent-') as temporary:
        for workers in (1, 2, 4):
            latency = []
            creation = []
            requests = events_total = synthetic_events = 0
            before = time.monotonic()
            for cycle in range(args.cycles):
                path = Path(temporary) / f'{workers}-{cycle}.jsonl'
                run = f'sdk-run:{workers}:{cycle}'
                expected = {f'sdk-session:{workers}:{cycle}:{worker}':
                            {'run_id': run, 'agent_id': f'sdk-agent:{worker}', 'task_id': f'sdk-task:{cycle}:{worker}'}
                            for worker in range(workers)}
                with si.Engine(telemetry_level=si.TelemetryLevel.DEEP,
                               telemetry_jsonl_path=str(path), capture_text=False) as engine:
                    engine.register_mock_backend()
                    with engine.load_model('mock', 'mock:tiny') as model:
                        def work(worker: int):
                            sid = f'sdk-session:{workers}:{cycle}:{worker}'
                            metadata = si.SessionMetadata(session_id=sid, **expected[sid])
                            created = time.monotonic()
                            session = engine.create_session(model, si.SamplingConfig.greedy(4), metadata=metadata)
                            create_ms = (time.monotonic() - created) * 1000
                            times = []
                            with session:
                                for index in range(args.requests_per_worker):
                                    requested = time.monotonic()
                                    parent = None if index % 3 == 2 else f'parent:{worker}:{index}'
                                    metadata = si.RequestMetadata(parent) if index % 3 != 2 else None
                                    if index % 4 == 0:
                                        result = session.generate(private, metadata=metadata)
                                    elif index % 4 == 1:
                                        result = session.chat([si.ChatMessage('user', private)], metadata=metadata)
                                    else:
                                        stream = (session.stream(private, metadata=metadata) if index % 4 == 2
                                                  else session.chat_stream([si.ChatMessage('user', private)], metadata=metadata))
                                        with stream:
                                            chunks = list(stream)
                                            result = stream.result
                                        assert result is not None and ''.join(chunks) == result.text
                                    assert result.completed and result.completion_tokens == 4
                                    times.append((time.monotonic() - requested) * 1000)
                            return create_ms, times
                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            for create_ms, times in pool.map(work, range(workers)):
                                creation.append(create_ms)
                                latency.extend(times)
                raw = path.read_text()
                assert private not in raw
                events = [json.loads(line) for line in raw.splitlines()]
                queued = [event for event in events if event['event_type'] == 'request.queued']
                assert len(queued) == workers * args.requests_per_worker
                assert Counter(event['session_id'] for event in queued) == {
                    sid: args.requests_per_worker for sid in expected}
                by_session = {sid: [] for sid in expected}
                for event in queued:
                    by_session[event['session_id']].append(event)
                parents = {}
                for sid, entries in by_session.items():
                    worker = int(sid.rsplit(':', 1)[1])
                    for index, event in enumerate(entries):
                        parent = None if index % 3 == 2 else f'parent:{worker}:{index}'
                        assert event['attributes'].get('parent_request_id') == parent
                        assert ('parent_request_id' in event['attributes']) == (parent is not None)
                        assert event['attributes']['kind'] == ('generate' if index % 4 in (0, 2) else 'chat')
                        parents[event['request_id']] = parent
                assert len(parents) == len(queued)
                lifecycle = [event for event in events if event['event_type'].startswith('request.')]
                assert len(lifecycle) == 3 * len(queued)
                assert Counter(event['event_type'] for event in lifecycle) == {
                    'request.queued': len(queued), 'request.started': len(queued), 'request.completed': len(queued)}
                for event in lifecycle:
                    parent = parents[event['request_id']]
                    assert event['attributes'].get('parent_request_id') == parent
                    assert ('parent_request_id' in event['attributes']) == (parent is not None)
                assert not any(event['event_type'] == 'telemetry.dropped' for event in events)
                correlated = [event for event in events if event.get('session_id') in expected]
                assert correlated
                for event in correlated:
                    assert all(event.get(key) == value for key, value in expected[event['session_id']].items())
                    assert event['producer'].get('synthetic') is True
                assert all('text' not in event['attributes'] for event in events
                           if event['event_type'] == 'inference.token.generated')
                sequences = [event['sequence'] for event in events]
                assert sequences == list(range(len(events)))
                assert len({event['producer']['instance_id'] for event in events}) == 1
                assert len({event['event_id'] for event in events}) == len(events)
                if args.fixture_dir and cycle == 0:
                    args.fixture_dir.mkdir(parents=True, exist_ok=True)
                    (args.fixture_dir / f'parents-{workers}-capture-off.jsonl').write_text(raw)
                requests += len(queued)
                events_total += len(events)
                synthetic_events += len(correlated)
            ordered = sorted(latency)
            scenarios.append({'workers': workers, 'cycles': args.cycles, 'sessions': workers * args.cycles,
                              'requests': requests, 'events': events_total,
                              'synthetic_session_events': synthetic_events, 'status': 'passed',
                              'median_request_ms': statistics.median(latency),
                              'p95_request_ms': ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
                              'median_session_create_ms': statistics.median(creation),
                              'elapsed_seconds': time.monotonic() - before})
    receipt = {'status': 'passed', 'synthetic': True, 'scope': 'Mixed actual mock SDK requests, parent/session lineage, privacy and producer cursor stability; synthetic SDK/engine overhead only, no provider/model quality.',
               'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'source_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)).hexdigest(),
               'worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)),
               'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'library': str(library), 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
               'scenarios': scenarios, 'requests': sum(s['requests'] for s in scenarios),
               'sessions': sum(s['sessions'] for s in scenarios), 'elapsed_seconds': time.monotonic() - start}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
