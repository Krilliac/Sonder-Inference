#!/usr/bin/env python3
"""Bounded real-library MOCK request-cost/stability qualification; no model claims.

Run from the repo with PYTHONPATH=bindings/python/src. Each scenario has its
own child process, native library and 120-second deadline; the parent has a
600-second global deadline. A legacy library is optional and generates only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def child(args):
    import sonder_inference as si
    from sonder_inference._lib import Library

    library = Library(ctypes.CDLL(str(args.library.resolve())), str(args.library.resolve()))
    messages = [si.ChatMessage("user", "héllo ✓")]
    prompt = "User: héllo ✓\n\nAssistant:"
    samples = []
    completed = cancelled = 0
    start = time.perf_counter()
    with si.Engine(library=library, telemetry_level=si.TelemetryLevel.OFF) as engine:
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            def worker(index):
                timings = []
                ok = stops = 0
                for cycle in range(args.cycles):
                    with engine.create_session(model, si.SamplingConfig.greedy(8, seed=index + cycle)) as session:
                        call = (lambda callback=None: session.chat(messages, on_token=callback)
                                ) if args.mode == "chat" else (
                                    lambda callback=None: session.generate(prompt, on_token=callback))
                        for request in range(args.requests):
                            cancel = request % 16 == 0
                            before = time.perf_counter_ns()
                            result = call((lambda _text: session.cancel()) if cancel else None)
                            timings.append((time.perf_counter_ns() - before) / 1e6)
                            if cancel:
                                if not result.cancelled or result.completion_tokens >= 8:
                                    raise RuntimeError("cancellation contract failed")
                                stops += 1
                            else:
                                if not result.completed or result.chunks != 8 or not result.text:
                                    raise RuntimeError("completion contract failed")
                                ok += 1
                        # Reuse after each cycle's cancellations, including switching
                        # back to the unchanged generation export after chat.
                        if not session.generate(prompt).completed:
                            raise RuntimeError("reuse contract failed")
                        ok += 1
                return timings, ok, stops

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for timings, ok, stops in pool.map(worker, range(args.workers)):
                    samples.extend(timings)
                    completed += ok
                    cancelled += stops
    samples.sort()
    rss = None
    if sys.platform == "linux":
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"status": "passed", "scope": "Synthetic MOCK transport/request cost; no provider/model quality or inference speed claim",
            "library": str(args.library.resolve()), "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
            "abi_version": library.abi_version, "chat_export": library.has_symbol("sonder_session_chat"),
            "mode": args.mode, "workers": args.workers, "cycles": args.cycles,
            "measured_requests": len(samples), "completed": completed, "cancelled": cancelled,
            "sessions": args.workers * args.cycles, "elapsed_seconds": time.perf_counter() - start,
            "request_ms": {"median": statistics.median(samples),
                           "p95": samples[min(len(samples) - 1, int(len(samples) * 0.95))], "max": max(samples)},
            "process_peak_rss_kib_linux": rss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--legacy-library", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--requests-per-worker", dest="requests", type=int, default=256)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["generate", "chat"], default="chat", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(w < 1 or w > 16 for w in args.workers) or not 1 <= args.requests <= 1024 or not 1 <= args.cycles <= 16:
        parser.error("workers must be 1..16, requests 1..1024 and cycles 1..16")
    if args.child:
        args.workers = args.workers[0]
        print(json.dumps(child(args)))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "running", "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "scope": "Synthetic MOCK only", "scenarios": [], "errors": []}
    deadline = time.monotonic() + 600
    try:
        scenarios = [(args.library, "generate"), (args.library, "chat")]
        if args.legacy_library:
            scenarios.insert(0, (args.legacy_library, "generate"))
        for library, mode in scenarios:
            for workers in args.workers:
                command = [sys.executable, str(Path(__file__).resolve()), "--child", "--library", str(library.resolve()),
                           "--mode", mode, "--workers", str(workers), "--requests-per-worker", str(args.requests),
                           "--cycles", str(args.cycles), "--out", str(args.out)]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("global qualification deadline exceeded")
                result = subprocess.run(command, text=True, capture_output=True, timeout=min(120, remaining), env=os.environ.copy())
                if result.returncode:
                    raise RuntimeError(f"scenario {mode}/{workers} exited {result.returncode}: {result.stderr}")
                receipt["scenarios"].append(json.loads(result.stdout))
                args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt["status"] = "passed"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["errors"].append(str(error))
        raise
    finally:
        receipt["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "scenarios": len(receipt["scenarios"]),
                      "measured_requests": sum(s["measured_requests"] for s in receipt["scenarios"])}))


if __name__ == "__main__":
    main()
