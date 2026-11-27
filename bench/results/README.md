# Benchmark results

Small, reviewed `sonder.inference.bench/1` snapshots only (no raw telemetry
dumps, no model files). Name files `<YYYY-MM-DD>-<backend>-<model>-<corpus>.json`
(the `sonder-bench` default stem). Each has a matching `.md` summary.
Results from the `mock` backend are harness checks, not performance data.

No measured native baseline is recorded here yet. Capture one with
`sonder-bench --model <path.gguf>` and record the hardware and build configuration.
