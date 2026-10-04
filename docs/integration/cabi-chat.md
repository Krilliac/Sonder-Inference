# Text chat through the C ABI and Python

`sonder_session_chat` is additive in ABI version 1. Existing records, status
values, exports and generation behavior retain their layout and semantics.
The actual implementation calls C++ `Session::chat`; it does not create another
scheduler, provider path, retry policy or queue. Native chat receives structured
messages; the existing role-labelled prompt fallback handles other backends.

## Records and bounds

```c
#define SONDER_MAX_CHAT_MESSAGES 1024u
typedef struct sonder_chat_message {
    uint32_t struct_size;
    const char* role;
    const char* content;
} sonder_chat_message;
sonder_status sonder_session_chat(sonder_session* session,
    const sonder_chat_message* const* messages, size_t message_count,
    sonder_token_callback callback, void* user_data,
    sonder_generation_stats* out_stats);
```

Initialize every record's `struct_size` to `sizeof(sonder_chat_message)`.
A pointer array allows differently sized records without an array-stride
compatibility break. The original prefix through `content` is required; larger
records are accepted with unknown tail fields ignored. All pointers must be
valid, non-null and remain unchanged during the synchronous call. Strings are
borrowed UTF-8 NUL-terminated inputs, copied before request submission.

Conversations require 1–1,024 records. Roles are system/user/assistant/tool and
the last message must be user or tool. Empty content is allowed. The limit
bounds record count, not total text bytes. Reasoning content, structured tool
calls, images, session metadata and telemetry callbacks are outside this slice.
Python validates shapes and NULs, rejects unsupported mapping fields, consumes
at most 1,025 iterator records, and retains native input buffers through return.

## Requests and compatibility

Callback, generation stats, errors, cancellation and session ownership follow
`sonder_session_generate`. Cancelled requests return `SONDER_OK` with outcome
CANCELLED. A session retains its model and engine after their caller handles
are released. C callers must not destroy a session while a request is running.
Python holds the handle until the native call returns; close from another
thread cancels and drains, while reentrant close in a callback is rejected.
The same incremental UTF-8 callback decoder is used by generate and chat.

Bindings detect the new export. Older ABI v1 libraries still load and generate;
chat alone raises UnsupportedError. No ABI major version bump is needed.

## Privacy and interoperability

Telemetry remains off by configuration or uses the existing bounded bus.
Requests keep kind=chat; producer identity, sequence/event IDs and drop behavior
are unchanged. Raw message text is absent from events. Token text is present
only with explicit capture_text consent, as for C++ chat. Engine batching,
producer cursor, effect recovery and rollback contracts remain upstream; this
bridge introduces no replacement mechanism.

## Qualification

Tests exercise the real C11/C++ ABI and real Python-loaded shared library:
message bounds, versioned record tails, native loopback Ollama HTTP routing,
UTF-8, stats, callback stop, cancellation, session reuse, engine/model lifetime,
concurrent close and privacy consent/cursors. Native HTTP uses an in-process
mock server with existing licensed test dependencies, not a live provider.

Local and hosted qualification receipts will be recorded on the reviewed
revision. Synthetic transport timings or request counts cannot establish
provider throughput, inference speed or model quality.

Reproduce request-cost and reuse stress after a shared-library build:

```sh
PYTHONPATH=bindings/python/src python scripts/stress_cabi_chat.py \
  --library build/shared/libsonder_inference.so \
  --workers 1 4 8 --requests-per-worker 1024 --cycles 8 --out chat-stress.json
```

An optional `--legacy-library /path/to/older/library` adds isolated baseline
generate scenarios. Each cell uses one process with a 120-second deadline and
the driver has a 600-second deadline. Every sixteenth request cancels from its
callback; sessions are reused and closed each cycle. The receipt records
library SHA-256, request latency distributions and Linux process peak RSS,
including interpreter/library memory. This is not a provider/model benchmark.
The normal CTest suite also retains all three bounded telemetry stress cases.
