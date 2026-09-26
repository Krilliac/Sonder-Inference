# Sonder repository boundaries

The ecosystem consists of three repositories:

| Repository | Responsibility |
| --- | --- |
| [Sonder Runtime](https://github.com/Krilliac/Sonder-runtime) | Agent/task orchestration, tool execution, routing, budgets, and host integration. |
| [Sonder Inference](https://github.com/Krilliac/Sonder-Inference) | Proposed inference engine: model/session lifecycle, batching, KV/context management, device policy, and backend execution. |
| [Sonder Observatory](https://github.com/Krilliac/Sonder-Observatory) | Proposed telemetry consumer: live visualization, recordings, replay, inspection, and diagnostics. |

Runtime coordinates tasks and requests inference. Inference owns execution and
its resource limits. Runtime and Inference emit versioned telemetry that
Observatory consumes and correlates. Observatory does not own orchestration or
inference state, and generation must work when it is absent or disconnected.

Producer instrumentation belongs with the runtime that owns the state. The
viewer remains a separate process; its renderer must not enter the inference
critical path. Preserve the existing Ollama compatibility path until the
inference roadmap's replacement gates pass.

Runtime reaches Inference over local HTTP served by `sonder-infer serve`
(ADR-020, [SERVER.md](SERVER.md)): an OpenAI-compatible chat subset plus
health, models and backend identity. Observatory connects to each producer
directly: Inference serves its own live telemetry (SSE/NDJSON, discovery at
`/.well-known/sonder-telemetry`), and Runtime does not relay it. Correlation
across producers uses the Runtime turn id, sent as `X-Sonder-Run-Id` and
`X-Sonder-Parent-Request-Id`.

These are architectural roles, not a claim that every proposed integration is
implemented. Protocol package ownership, compatibility policy, and deployment
details still require agreement. No Runtime code is extracted by this scaffold.
There is no separate Orchestrator dependency.

See Observatory's [architecture](https://github.com/Krilliac/Sonder-Observatory/blob/main/docs/ARCHITECTURE.md)
and [integration design](https://github.com/Krilliac/Sonder-Observatory/blob/main/docs/INTEGRATION.md).
