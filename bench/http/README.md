# HTTP benchmark harness (`bench/http`)

`bench_http.py` benchmarks any OpenAI-compatible server over HTTP. That
includes raw `llama-server`, `sonder-infer serve` and a Sonder Runtime `/v1`
endpoint. It complements the in-process harness in [`bench/`](../README.md),
which runs through Engine sessions. This harness measures what a client sees
on the wire: cached prompt tokens, inter-token latency, concurrency and
long-context recall.

It needs Python 3.10+ and the standard library only (no pip installs). Tests
use pytest.

## What it measures

**Per request**

- Streaming `POST /v1/chat/completions` with `stream_options.include_usage`.
  For raw `llama-server` you can use `POST /completion` instead (`--api
  completion`).
- **TTFT** is measured from sending the request to the first chunk that
  carries generated text. Content, `reasoning_content` and tool-call deltas
  all count.
- **Inter-token latency** is one sample per token-bearing chunk: the gap to
  the previous one, divided by the tokens in that chunk. When a chunk does not
  say how many tokens it holds, completion tokens / chunks is used. Reported
  as p50/p95/p99.
- **End-to-end** time.
- **Prompt and completion tokens.**
- **Prefill and decode tok/s**, both client-side and from `timings` when the
  server sends it.
  - Client decode is (tokens - 1) / (end - TTFT).
  - Client prefill is uncached prompt tokens / TTFT. It includes network and
    queueing time, so it is a lower bound.
- **Prefix hit** is cached / prompt tokens. The cached count comes from
  `timings.cache_n` (llama-server) or
  `usage.prompt_tokens_details.cached_tokens`.
  - When the server reports neither, the hit is **unknown** (`null` in JSON).
    It is never shown as 0.
  - llama-server's `timings.prompt_n` counts only the tokens it processed,
    so prompt tokens = `prompt_n + cache_n`.
- **Draft acceptance** is `draft_n_accepted / draft_n` when present
  (speculative decoding).

**Scenarios** (built-in set; print it with `bench_http.py scenarios`):

| kind | what it does |
|---|---|
| `agent_session` | A multi-turn agent session. It has a large, byte-stable system prompt that includes tool schemas (embedded as text by default, because `sonder-infer serve` rejects `tools`; set `"tools_mode": "native"` to send `tools`). The history grows every turn with a tool-result block and the model's answer. With `"volatile_top": true`, a timestamp and nonce block goes at the very top of the system prompt and changes every turn, which shows how much prefix cache that costs. |
| `concurrency_sweep` | Sub-agent fan-out. One warm-up request primes a shared prefix, then short requests run at each concurrency level (default 1/2/4/8, `waves` × level requests per level). Records makespan, aggregate tok/s and requests/s. |
| `long_context` | One document per size (default 32k/64k/100k/130k) with planted facts at several depths (default 10/50/90 %), made of kvb.py-style `Record N: code …` filler. Each depth gets its own question, and the answer must contain the exact phrase. The first question per size is a cold prefill; the rest reuse the cached body. Sizes that do not fit the server's `n_ctx` (from `/props` or `--ctx-size`) are skipped and listed. |
| `repeat_prompt` | The same prompt N times. Repeats 2+ should be almost entirely cached. |

Sizes are in tokens. When the server has llama-server's `/tokenize`, prompts
are trimmed to size with it. Otherwise the harness estimates at 4 chars per
token, and the report says which method it used. The actual prompt tokens are
always recorded from the response.

**Non-vacuity checks.** Every run evaluates these. Each one is
`pass`/`warn`/`fail`/`skip`, and together they give the run's verdict:

- `request_count`: every expected request completed with `[DONE]` or a final
  event, and no error. Otherwise **fail**.
- `tokens_generated`: more than 0 completion tokens per scenario. Otherwise
  **fail**.
- `prefix_reuse`: on turn 2+ of the stable agent scenario, the prefix hit is
  greater than 0. It **warns** when the hit is 0 (a cache that stopped working
  looks like a slow model) or unknown.
- `repeat_reuse` and `volatile_cache_loss`: the cache behaves as expected (a
  repeat hit of 0.9 or more, and stable greater than volatile).
- `recall`: planted facts recalled exactly.
- `gpu_used`: with PDH sampling on, the server process holds dedicated GPU
  memory. A process holding 0 MiB **fails**.
- `vram_spill`: shared GPU memory rose more than 256 MiB above the clean
  baseline. This **warns**.
- `server_identity`: records `/v1/models`, `/props` (`build_info`,
  `model_path`, `n_ctx`, slots), `/health`, `/v1/sonder/health` and
  `/v1/sonder/identity`, whichever answer.

Exit code: 0 for pass or warn, 1 for fail (or for warn with `--strict`), 2
for a usage error.

## GPU memory (Windows)

`--pid N` or `--process-name llama-server` samples the per-process PDH
counters `\GPU Process Memory(pid_<PID>_*)\Dedicated Usage` and `Shared
Usage` once before the first scenario and before and after each scenario. This
is the same method as the KV-fit bench (`kvb.py`). The harness also records
`nvidia-smi` total used memory when that tool exists.

Spill is flagged when shared usage is more than 256 MiB above the clean
baseline. A clean load's shared usage rises about 1 MiB per 1024 ctx (for
example Q3_K_XL: 148 MiB at 49k), so the baseline can be:

- the first sample of the run (default). This is right when the server loaded
  cleanly.
- `--shared-baseline-mib X`.
- `--clean-line 148@49152`: the reference point plus (n_ctx - 49152) / 1024
  MiB, using the server's n_ctx.

On other platforms, or without `--pid`/`--process-name`, sampling is skipped
and the `gpu_used` check says so.

## Examples

Run these from the repository root. Results go to
`bench/http/out/<stamp>-<label>.json` and `.md`; that directory is
git-ignored. Use `--out PREFIX` to choose another location.

**Raw llama-server on 127.0.0.1:8080**

```
python bench/http/bench_http.py run --base-url http://127.0.0.1:8080 --label llama-q3 \
    --process-name llama-server --no-think
# only the long-context recall scenario, at sizes that fit a 64k server, via the raw /completion route
python bench/http/bench_http.py run --base-url http://127.0.0.1:8080 --only long-context \
    --long-sizes 16384,32768,60000 --api completion --label llama-q3-long
```

`--no-think` sends `chat_template_kwargs: {"enable_thinking": false}`. Without
it, a Qwen3-style model may spend a small `max_tokens` budget inside a
`<think>` block and then fail recall.

**`sonder-infer serve`** (default `--host 127.0.0.1 --port 11437`; see
[docs/SERVER.md](../../docs/SERVER.md))

```
sonder-infer serve --backend llamaserver --model local-model --llamaserver-config llamaserver.json
python bench/http/bench_http.py run --base-url http://127.0.0.1:11437/v1 --label sonder-serve \
    --process-name llama-server
# a server started with --token-file:
python bench/http/bench_http.py run --base-url http://127.0.0.1:11437 --token-file token.txt --label sonder-serve
```

`sonder-infer serve` reports `timings` but, depending on the backend, may not
report cached tokens. In that case the prefix hit shows as `unknown` and
`prefix_reuse` warns rather than passing. For a spawned `llamaserver` child,
`--process-name llama-server` samples the child process.

**Smaller or custom scenarios**

```
python bench/http/bench_http.py scenarios > my-scenarios.json    # edit, then:
python bench/http/bench_http.py run --base-url http://127.0.0.1:8080 --scenarios bench/http/scenarios.example.json
```

[`scenarios.example.json`](scenarios.example.json) fits a 32k-context server.
Per-scenario keys:

- `max_tokens`, `temperature`, `seed`, `api` (`chat`/`completion`) and
  `extra_body`.
- Per kind: `turns`, `system_tokens`, `tools`, `turn_tokens`, `volatile_top`,
  `tools_mode`, `levels`, `waves`, `shared_prefix_tokens`, `sizes`, `depths`,
  `prompt_tokens`, `repeats`.

**A/B compare**

```
python bench/http/bench_http.py run --base-url http://127.0.0.1:8080 --label A-q8q51 --out /tmp/A
python bench/http/bench_http.py run --base-url http://127.0.0.1:8080 --label B-q4q4 --out /tmp/B
python bench/http/bench_http.py compare /tmp/A.json /tmp/B.json --out /tmp/A-vs-B.md
```

`compare` matches scenario/group pairs and prints A, B, delta, delta % and
better/worse for each metric: TTFT and ITL percentiles, e2e, prefill/decode
tok/s, prefix hit, recall, and aggregate tok/s and makespan for concurrency
levels. `--json` prints the rows instead.

## Tests

```
python -m pytest -v bench/http/tests
```

The tests cover the metric math (percentiles, ITL division, prefix-hit
sources and the unknown-is-not-zero rule, the checks, PDH parsing) and the SSE
parser (split feeds, CRLF, comments, multi-line data, UTF-8). They also run
the client and full CLI runs against an in-process fake SSE server
(`tests/fake_server.py`, stdlib `http.server`). The fake has a single-slot
prefix cache and answers planted-fact questions from the prompt, so the
cache and recall checks are exercised for real.

They run in the `pytest` job of `.github/workflows/python.yml`. No GPU,
network or model weights are involved.

## Caveats

- Client-side timings include the HTTP stack and any server queueing. Use the
  `timings`-based columns for pure engine speed.
- Ollama's `/v1` endpoint reports neither `timings` nor cached tokens, so
  prefix hit is unknown there.
- The long-context and agent prompts are synthetic. Recall is an
  exact-substring check on planted phrases, not a general quality
  measurement.
