"""Real Python/C ABI/adapter interoperability against a local mock HTTP service."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sonder_inference as si


def test_python_chat_native_http_unicode_and_message_order(lib):
    received = []
    show = (Path(__file__).resolve().parents[3] /
            "src/backends/ollama/tests/fixtures/show.json").read_bytes()
    text = "héllo ✓"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body))
            if self.path == "/api/show":
                response = show
                content_type = "application/json"
            elif self.path == "/api/chat":
                records = [{"message": {"role": "assistant", "content": char}, "done": False}
                           for char in text]
                records.append({"message": {"role": "assistant", "content": ""}, "done": True,
                                "prompt_eval_count": 12, "eval_count": 8})
                response = b"".join((json.dumps(record, ensure_ascii=False) + "\n").encode()
                                    for record in records)
                content_type = "application/x-ndjson"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            # Awkward transport boundaries include the UTF-8 byte sequences.
            for byte in response:
                self.wfile.write(bytes([byte]))
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with si.Engine(telemetry_level=si.TelemetryLevel.OFF) as engine:
            engine.register_ollama_backend(f"http://127.0.0.1:{server.server_port}")
            with engine.load_model("ollama", "qwen3:8b") as model:
                with engine.create_session(model, si.SamplingConfig.greedy(24)) as session:
                    messages = [{"role": "system", "content": "Be concise"},
                                {"role": "user", "content": text}]
                    chunks = []
                    result = session.chat(messages, on_token=chunks.append)
        assert result.completed and result.text == "".join(chunks) == text
        assert result.chunks == len(text) and result.prompt_tokens == 12 and result.completion_tokens == 8
        chats = [body for path, body in received if path == "/api/chat"]
        assert len(chats) == 1 and chats[0]["messages"] == messages
        assert "prompt" not in chats[0] and not any(path == "/api/generate" for path, _ in received)
        assert chats[0]["options"]["num_predict"] == 24
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()
