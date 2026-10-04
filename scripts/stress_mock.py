"""Bounded mock-only HTTP concurrency and in-flight shutdown qualification."""
import argparse
import concurrent.futures
import json
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def percentile(values, fraction):
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]


def rss_kib(pid):
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return None


def request(base, index, stream, first_token=None, tokens=8):
    started = time.perf_counter()
    payload = {"model": "mock:tiny", "messages": [{"role": "user", "content": f"synthetic smoke {index}"}],
               "stream": stream, "max_tokens": tokens, "stream_options": {"include_usage": True}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        if stream:
            text = []
            usage = None
            done = False
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                value = raw[5:].strip()
                if value == b"[DONE]":
                    done = True
                    break
                event = json.loads(value)
                assert "error" not in event, event
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content")
                    if content:
                        text.append(content)
                        if first_token is not None:
                            first_token.set()
            assert done and text and usage is not None
        else:
            document = json.load(response)
            assert document["choices"][0]["message"]["content"]
            usage = document["usage"]
    assert usage["completion_tokens"] == tokens, usage
    return (time.perf_counter() - started) * 1000


def summary(binary, cycles, receipts, error=None, *, status="passed"):
    result = {"synthetic": True, "scope": "mock HTTP transport and lifecycle; no model performance claim",
              "binary": str(binary), "status": "failed" if error is not None else status,
              "requested_cycles": cycles, "completed_cycles": len(receipts), "cycles": receipts,
              "total_requests": sum(cycle["requests"] for cycle in receipts),
              "errors": int(error is not None)}
    if error is not None:
        result["failure"] = {"type": type(error).__name__, "message": str(error)[:1000]}
    return result


def write_receipt(path, result, *, echo=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    if echo:
        print(json.dumps(result))


def qualify(binary, cycles, receipts=None, *, mock_delay_ms=10, on_cycle=None):
    if receipts is None:
        receipts = []
    for cycle in range(cycles):
        with tempfile.TemporaryDirectory(prefix="sonder-inference-stress-") as tmp:
            ready = Path(tmp) / "ready.json"
            with (Path(tmp) / "server.log").open("w") as log:
                process = subprocess.Popen([str(binary), "serve", "--backend", "mock", "--model", "mock:tiny",
                                            "--mock-delay-ms", str(mock_delay_ms), "--port", "0", "--ready-file", str(ready),
                                            "--shutdown-grace-ms", "2000"], stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 10
                    while not ready.exists():
                        assert process.poll() is None, "server exited before readiness"
                        assert time.monotonic() < deadline, "startup exceeded ten seconds"
                        time.sleep(0.02)
                    base = json.loads(ready.read_text())["url"]
                    while True:
                        try:
                            with urllib.request.urlopen(base + "/v1/sonder/health", timeout=2) as response:
                                health = json.load(response)
                                health_status = response.status
                        except urllib.error.HTTPError as error:
                            if error.code != 503:
                                raise
                            health = json.load(error)
                            health_status = error.code
                        if health["status"] == "ready":
                            assert health_status == 200, "ready health did not return HTTP 200"
                            break
                        assert time.monotonic() < deadline, "health exceeded startup budget"
                        time.sleep(0.02)
                    assert health["synthetic"] is True
                    started = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                        futures = [pool.submit(request, base, i, i % 2 == 0) for i in range(64)]
                        durations = [future.result(timeout=15) for future in futures]
                    mixed_elapsed = time.perf_counter() - started
                    peak_rss = rss_kib(process.pid)
                    events = [threading.Event() for _ in range(4)]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                        draining = [pool.submit(request, base, 64 + i, True, events[i], 16) for i in range(4)]
                        assert all(event.wait(5) for event in events), "not all streams produced a first token"
                        with urllib.request.urlopen(base + "/v1/sonder/health", timeout=2) as response:
                            active_health = json.load(response)
                            assert response.status == 200
                        active_requests = sum(model["in_flight"] for model in active_health["models"])
                        assert active_requests == 4, "server did not retain all four admitted requests"
                        assert all(not future.done() for future in draining), "streams completed before shutdown"
                        shutdown_start = time.perf_counter()
                        process.send_signal(signal.SIGINT)
                        drain_durations = [future.result(timeout=5) for future in draining]
                        assert process.wait(timeout=5) == 0
                        shutdown_ms = (time.perf_counter() - shutdown_start) * 1000
                    assert not ready.exists(), "ready file survived shutdown"
                    receipts.append({"cycle": cycle + 1, "requests": 68, "errors": 0,
                                     "server_commit": health["commit"], "instance_id": health["instance_id"],
                                     "mixed_workers": 8, "mixed_streamed": 32, "mixed_nonstreamed": 32,
                                     "mock_delay_ms": mock_delay_ms,
                                     "mixed_elapsed_s": round(mixed_elapsed, 3),
                                     "latency_p50_ms": round(statistics.median(durations), 3),
                                     "latency_p95_ms": round(percentile(durations, .95), 3),
                                     "latency_max_ms": round(max(durations), 3), "peak_rss_kib": peak_rss,
                                     "shutdown_inflight_streams": len(drain_durations),
                                     "shutdown_health_in_flight": active_requests,
                                     "shutdown_completion_tokens": 16,
                                     "shutdown_ms": round(shutdown_ms, 3), "exit_code": 0, "ready_removed": True})
                    if on_cycle is not None:
                        on_cycle(summary(binary, cycles, receipts, status="running"))
                finally:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
    return summary(binary, cycles, receipts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--mock-delay-ms", type=int, default=10, choices=range(1, 51))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform == "win32":
        parser.error("this shutdown smoke requires POSIX signal delivery")
    binary = args.binary.resolve()
    completed = []

    def progress(result):
        write_receipt(args.out, result, echo=False)

    progress(summary(binary, args.cycles, completed, status="running"))
    try:
        result = qualify(binary, args.cycles, completed, mock_delay_ms=args.mock_delay_ms, on_cycle=progress)
    except Exception as error:
        write_receipt(args.out, summary(binary, args.cycles, completed, error))
        raise
    write_receipt(args.out, result)
