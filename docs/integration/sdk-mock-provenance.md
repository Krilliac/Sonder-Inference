# SDK mock event provenance

A real Python SDK recording of 24 mock generate/chat/stream requests emitted
400 session events without `producer.synthetic`. Observatory displayed their
producer-reported provenance even though the backend was known to be mock.
Three regression cases reproduce this with an archived main shared library at
metrics, standard and deep telemetry levels. The HTTP server already labels
its single selected backend; embedding hosts can contain multiple backends.

Append a positive `TelemetryContext::synthetic_work` flag to the C++ context.
Known `mock` backend registration, model load/unload and session contexts set
it. Request, token and scheduler events carrying that session context retain
the flag. The bus writes the existing envelope `producer.synthetic = true`
when the context identifies synthetic work, even if the host-wide setting is
false. Other contexts inherit the host's optional setting. Unknown does not
become false, and model names such as `mock:tiny` do not classify a backend.
Engine/device/shared scheduler observations keep their existing host policy.
This needs no mutable registry or additional queue, lock, I/O or callback.

The C++ context layout changes and C++ consumers recompile; the stable C ABI's
layouts, symbols and major remain unchanged. C ABI/Python callers receive
correct mock labels without a new option. No schema, dependency, producer
instance, sequence, event ID, consent, raw-text capture, batching, effect
recovery or rollback contract changes. A custom synthetic backend with another
identity still needs its host's classification; the engine does not guess.

Native controls exercise host unknown/false/true precedence and a single engine
mixing a known mock backend with an unclassified deterministic test adapter.
They check model/session lifecycle, generate/chat, token events, cursor
uniqueness and private text exclusion. The adapter is a synthetic fixture,
not a real provider. Python controls exercise actual C ABI generate/chat/
stream calls at all three enabled levels. The existing bounded metadata stress
driver now requires synthetic provenance for every correlated mock event.

```sh
python -m pytest bindings/python/tests/test_mock_provenance.py -q
python scripts/stress_session_metadata.py --library build/shared/libsonder_inference.so --cycles 16 --requests-per-worker 32 --out /tmp/mock-provenance.json
```

Qualification uses loopback mock HTTP, fresh-process SDK stress and the actual
Observatory consumer. Timing measures synthetic SDK/engine/telemetry overhead,
and establishes no model/provider speed or quality. Exact-revision receipts
and platform results are recorded in the pull request; generated recordings
remain outside Git.

Root integration touches the shared C++ telemetry context/bus, engine model
lifecycle and sessions, native/Python controls, the existing stress driver and
these notes. No renderer or Runtime producer change is needed.

## Initial local qualification

Before commit, the strict native suite passes 928 tests and the actual shared
library passes 109 Python tests. Three sequential interleaved calibration
cohorts per revision exercise 10,752 mock requests / 336 sessions each, with
no concurrent builds. Baseline/candidate median SDK request time across those
cohorts is about 0.595/0.597 ms with one worker, 0.944/0.940 ms with two and
1.648/1.622 ms with four. The candidate checks 172,704 labelled session events.
These small local differences establish no statistical performance guarantee
or provider/model speedup. Exact committed-revision stress and consumer
qualification are recorded separately in the pull request.
