# Telemetry

Responsibility: Bounded inference events for external observers.

Implemented: `TelemetryBus` (`telemetry.cpp`) emitting Observatory envelope v1
JSONL through a bounded queue and background writer, with levels, drop
accounting, file/stream/memory sinks. See ../../docs/OBSERVATORY_CONTRACT.md.

See [source workspace](../README.md).
