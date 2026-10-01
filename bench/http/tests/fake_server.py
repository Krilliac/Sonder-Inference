"""Tiny in-process fake of an OpenAI-compatible streaming server (stdlib only).

It imitates the llama-server behaviour the harness depends on: SSE chunks,
a final chunk with ``timings`` / ``usage``, a single-slot prefix cache whose
hit is reported as ``timings.cache_n`` or
``usage.prompt_tokens_details.cached_tokens`` (or not at all), ``/props``,
``/tokenize`` and the raw ``/completion`` route. It answers planted-fact
questions by looking the fact up in the prompt, so recall checks are real.

"Tokens" are 4-character slices, which keeps the arithmetic predictable.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def ntok(text: str) -> int:
    return (len(text) + 3) // 4


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class FakeState:
    def __init__(self, cache_report: str = "timings", chunk_tokens: int = 1, delay: float = 0.002,
                 n_ctx: int = 65536, draft: bool = False, props: bool = True, tokenize: bool = True,
                 sse_crlf: bool = False, fail_after_chunks: int | None = None, status_override: int | None = None,
                 cache: bool = True, keyed_cache: bool = False, pinned_thinking: bool = False,
                 reasoning_only: bool = False):
        self.cache_report = cache_report  # "timings" | "usage" | "none"
        self.chunk_tokens = chunk_tokens
        self.delay = delay
        self.n_ctx = n_ctx
        self.draft = draft
        self.props = props
        self.tokenize = tokenize
        self.sse_crlf = sse_crlf
        self.fail_after_chunks = fail_after_chunks
        self.status_override = status_override
        self.cache = cache
        self.keyed_cache = keyed_cache
        self.pinned_thinking = pinned_thinking
        self.reasoning_only = reasoning_only
        self.prompts_by_key: dict[str, str] = {}
        self.accepted_per_position = {"0": 0, "1": 0}
        self.last_prompt = ""
        self.requests: list[dict] = []
        self.get_requests: list[dict] = []
        self.lock = threading.Lock()


def answer_for(prompt: str, last: str) -> str:
    marker = re.search(r"marker=(STALL_CANARY_\d+);", last)
    if marker:
        return marker.group(1)
    m = re.search(r"access phrase for vault (\S+) mentioned", last)
    if m:
        f = re.search(r"access phrase for vault " + re.escape(m.group(1)) + r" is (\S+?)\.(\s|$)", prompt)
        if f:
            return f"The phrase is {f.group(1)}"
        return "I do not know"
    return "Next I would run the unit tests and read the failing module carefully before editing anything"


def make_handler(state: FakeState):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):  # quiet
            pass

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            with state.lock:
                state.get_requests.append({"path": self.path, "headers": dict(self.headers)})
            if self.path == "/metrics":
                with state.lock:
                    counters = dict(state.accepted_per_position)
                data = "\n".join(
                    f'llamacpp:spec_decode_accepted_tokens_per_pos_total{{position="{pos}"}} {count}'
                    for pos, count in counters.items()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
            if self.path == "/props" and state.props:
                return self._json(200, {"build_info": "b0000-fake", "model_path": "/models/fake.gguf",
                                        "total_slots": 1, "chat_template": "x" * 100,
                                        "default_generation_settings": {"n_ctx": state.n_ctx}})
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            return self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            with state.lock:
                state.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization"),
                                       "headers": dict(self.headers)})
            if self.path == "/tokenize" and state.tokenize:
                return self._json(200, {"tokens": list(range(ntok(body.get("content", ""))))})
            if self.path not in ("/v1/chat/completions", "/completion"):
                return self._json(404, {"error": {"message": "not found"}})
            if state.status_override:
                return self._json(state.status_override, {"error": {"message": "overridden", "type": "test"}})
            if self.path == "/completion":
                prompt = body.get("prompt", "")
                last = prompt
                max_tokens = int(body.get("n_predict", 16))
            else:
                if body.get("tools") is not None and body.get("reject_tools"):
                    return self._json(400, {"error": {"message": "tools unsupported"}})
                msgs = body.get("messages") or []
                prompt = "".join(f"<|{m['role']}|>{m['content']}" for m in msgs) + "<|assistant|>"
                last = msgs[-1]["content"] if msgs else ""
                max_tokens = int(body.get("max_tokens", 16))
            with state.lock:
                cache_key = body.get("prompt_cache_key", "")
                previous = state.prompts_by_key.get(cache_key, "") if state.keyed_cache else state.last_prompt
                cached_chars = common_prefix(previous, prompt) if state.cache else 0
                state.prompts_by_key[cache_key] = prompt
                state.last_prompt = prompt
            p_tok = ntok(prompt)
            cache_n = min(cached_chars // 4, max(p_tok - 1, 0))
            words = answer_for(prompt, last).split(" ")
            pieces = [w + " " for w in words][:max_tokens]
            time.sleep(0.003 + 0.000002 * (p_tok - cache_n))  # "prefill"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            nl = "\r\n" if state.sse_crlf else "\n"

            def send(obj) -> None:
                payload = obj if isinstance(obj, str) else json.dumps(obj)
                self.wfile.write(f"data: {payload}{nl}{nl}".encode())
                self.wfile.flush()

            chat = self.path != "/completion"
            if chat:
                send({"choices": [{"index": 0, "delta": {"role": "assistant"}}], "model": "fake-model"})
                if state.pinned_thinking:
                    send({"sonder": {"warnings": ["enable_thinking=off was overridden by the server pin (on) "
                                                 "that keeps the prompt prefix cacheable"]}, "choices": []})
                self.wfile.write(f": keep-alive comment{nl}{nl}".encode())
            k = state.chunk_tokens
            sent = 0
            for i in range(0, len(pieces), k):
                if state.fail_after_chunks is not None and sent >= state.fail_after_chunks:
                    send({"error": {"message": "backend exploded", "code": 500}})
                    return
                text = "".join(pieces[i:i + k])
                if chat:
                    field = "reasoning_content" if state.reasoning_only else "content"
                    send({"choices": [{"index": 0, "delta": {field: text}}], "model": "fake-model"})
                else:
                    send({"content": text, "stop": False})
                sent += 1
                time.sleep(state.delay)
            gen = len(pieces)
            with state.lock:
                state.accepted_per_position["0"] += 3
                state.accepted_per_position["1"] += 2
            timings = {"prompt_n": p_tok - cache_n, "prompt_ms": 1.0, "prompt_per_second": 1000.0 * (p_tok - cache_n),
                       "predicted_n": gen, "predicted_ms": 1.0, "predicted_per_second": 123.0}
            if state.cache_report == "timings":
                timings["cache_n"] = cache_n
            if state.draft:
                timings["draft_n"] = 10
                timings["draft_n_accepted"] = 7
            usage = {"prompt_tokens": p_tok, "completion_tokens": gen, "total_tokens": p_tok + gen}
            if state.cache_report == "usage":
                usage["prompt_tokens_details"] = {"cached_tokens": cache_n}
            if state.cache_report == "none":
                timings.pop("prompt_n")
                timings["prompt_n"] = p_tok  # Sonder serve style: prompt_n is the whole prompt
            if chat:
                final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "timings": timings}
                send(final)
                if (body.get("stream_options") or {}).get("include_usage"):
                    send({"choices": [], "usage": usage})
                send("[DONE]")
            else:
                send({"content": "", "stop": True, "timings": timings, "tokens_predicted": gen,
                      "tokens_evaluated": p_tok, "stop_type": "eos"})

    return H


class FakeServer:
    def __init__(self, **kw):
        self.state = FakeState(**kw)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()
