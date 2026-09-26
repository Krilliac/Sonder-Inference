# Proposed repository boundaries

Status: planning notes for the scaffold, not an approved extraction or API.

| Repository | Intended responsibility |
| --- | --- |
| Sonder Orchestrator | Goal/task planning, dependency coordination, bounded dispatch, budgets, and task-level recovery. |
| Sonder Runtime | Existing tool execution and host integration; any extraction or compatibility adapter is future work. |
| Sonder Inference | Model/session lifecycle, inference scheduling and batching, KV/context management, device policy, and backend execution. |
| Sonder Observatory | External observation and visualization of telemetry. |

Task scheduling and inference scheduling are separate concerns: Orchestrator
coordinates tasks, while Inference controls admission and token execution within
its own resource limits. Neither repository imports the other's private state.
Observatory must not become a dependency of inference's critical execution path.

The exact Runtime/Orchestrator split, calls between components, language choices,
transport, shared contracts, persistence, and deployment topology remain open.
No code or live configuration is migrated by this scaffold. Retain the existing
Ollama compatibility path until the inference roadmap's replacement gates pass.

References: [Sonder Runtime](https://github.com/Krilliac/Sonder-runtime),
[Sonder Inference](https://github.com/Krilliac/Sonder-Inference), and
[Sonder Observatory](https://github.com/Krilliac/Sonder-Observatory).
