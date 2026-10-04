#!/usr/bin/env python3
"""Bounded mock HTTP/Python stream qualification; no model performance claim."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 16:
        parser.error('--cycles must be 1..16')
    library = args.library.resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    environment = os.environ | {'SONDER_INFERENCE_LIBRARY': str(library),
                               'PYTHONPATH': str(root / 'bindings/python/src')}
    started = time.monotonic()
    runs = []
    stop = threading.Event()
    receipt = {'status': 'running', 'scope': 'Synthetic Python/C ABI/loopback HTTP delivery and lifecycle; '
               'elapsed time includes deliberate consumer pauses and test startup, not inference speed.',
               'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'library': str(library), 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
               'python_api_sha256': hashlib.sha256((root / 'bindings/python/src/sonder_inference/api.py').read_bytes()).hexdigest(),
               'cycles_per_worker': args.cycles, 'concurrency': [1, 2, 4]}

    def save(status: str) -> None:
        receipt.update(status=status, runs=runs, test_cases_passed=sum(run['test_cases_passed'] for run in runs),
                       elapsed_seconds=time.monotonic() - started)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary_out = args.out.with_suffix(args.out.suffix + '.tmp')
        temporary_out.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary_out.replace(args.out)

    save('running')
    with tempfile.TemporaryDirectory(prefix='sonder-python-stream-') as temporary:
        directory = Path(temporary)

        def worker(workers: int, index: int) -> list[dict]:
            results = []
            for cycle in range(args.cycles):
                if stop.is_set():
                    break
                stem = directory / f'w{workers}-{index}-{cycle}'
                xml = stem.with_suffix('.xml')
                before = time.monotonic()
                with stem.with_suffix('.log').open('w') as log:
                    process = subprocess.Popen(
                        [sys.executable, '-m', 'pytest', 'bindings/python/tests/test_stream_backpressure.py',
                         '-q', f'--junitxml={xml}'], cwd=root, env=environment,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name == 'posix'))
                    try:
                        code = process.wait(timeout=150)
                    except BaseException:
                        if os.name == 'posix':
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        else:
                            process.kill()
                        process.wait(timeout=10)
                        raise
                if code:
                    raise RuntimeError(f'stream qualification failed ({code}): {stem.with_suffix(".log").read_text()}')
                cases = list(ET.parse(xml).getroot().iter('testcase'))
                assert len(cases) == 8 and all(not list(case) for case in cases)
                results.append({'workers': workers, 'worker': index, 'cycle': cycle,
                                'test_cases_passed': len(cases), 'retries': 0,
                                'elapsed_seconds': time.monotonic() - before,
                                'cases': [{'name': case.attrib['name'], 'seconds': float(case.attrib['time'])}
                                          for case in cases]})
            return results

        for workers in (1, 2, 4):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(worker, workers, index) for index in range(workers)]
                for future in futures:
                    try:
                        runs.extend(future.result())
                    except BaseException:
                        stop.set()
                        save('failed')
                        raise
                    save('running')
    save('passed')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'runs'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
