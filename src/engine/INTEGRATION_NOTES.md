# Text chat C ABI integration

This bounded slice implements existing Python Session.chat through the existing
C++ Session::chat. The integrator must include the append-only public C header,
Python ctypes declarations/wrapper/tests, root C/C++ ABI tests, Ollama fake-HTTP
C ABI tests and their module registration, and integration/roadmap documentation.
No new dependency, ABI version change or reasoning/tool/image contract is added.
Existing scheduler, cancellation, telemetry consent/cursor and batching contracts
remain in the engine. See docs/integration/cabi-chat.md for qualification.
