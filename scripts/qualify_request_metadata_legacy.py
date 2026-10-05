#!/usr/bin/env python3
"""Real older ABI-v1 defaults and eager explicit-parent rejection controls."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bindings/python/src'))
import sonder_inference as si  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    path = args.library.resolve(strict=True)
    library = si.Library(ctypes.CDLL(str(path)), str(path))
    assert library.abi_version == 1
    assert not library.has_symbol('sonder_session_generate_with_metadata')
    assert not library.has_symbol('sonder_session_chat_with_metadata')
    controls = []
    with si.Engine(library=library, telemetry_level=si.TelemetryLevel.OFF) as engine:
        engine.register_mock_backend()
        with engine.load_model('mock', 'mock:tiny') as model:
            with engine.create_session(model, si.SamplingConfig.greedy(4)) as session:
                for method in ('generate', 'chat', 'stream', 'chat_stream'):
                    value = [si.ChatMessage('user', 'synthetic compatibility')] if 'chat' in method else 'synthetic compatibility'
                    result = getattr(session, method)(value)
                    if 'stream' in method:
                        with result as stream:
                            assert len(list(stream)) == 4
                        result = stream.result
                        assert not stream._thread.is_alive()
                    assert result.completed and result.completion_tokens == 4
                    original = session._request
                    def forbidden(*args, **kwargs):
                        raise AssertionError('unsupported metadata entered request/worker')
                    session._request = forbidden
                    try:
                        for metadata in (si.RequestMetadata(), si.RequestMetadata('parent:explicit')):
                            try:
                                getattr(session, method)(value, metadata=metadata)
                            except si.UnsupportedError:
                                controls.append({'method': method, 'empty_metadata': metadata.parent_request_id is None,
                                                 'default': 'passed', 'explicit_metadata': 'unsupported-before-worker'})
                            else:
                                raise AssertionError('old library silently ignored metadata')
                    finally:
                        session._request = original
    receipt = {'status': 'passed', 'library': str(path), 'library_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'scope': 'Actual archived ABI-v1 shared library; no provider/model work', 'controls': controls}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
