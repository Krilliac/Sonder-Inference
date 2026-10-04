# Telemetry

Responsibility: Bounded inference events for external observers.

Implemented: `TelemetryBus` (`telemetry.cpp`) emitting Observatory envelope v1
JSONL through a bounded queue and background writer, with levels, drop
accounting, file/stream/memory sinks. See ../../docs/OBSERVATORY_CONTRACT.md.

Sink write/flush exceptions and built-in file/ostream fail bits retire the
failed sink without retry or exception
content logging. Healthy delivery, queue-drop accounting, and producer cursors
remain intact; `TelemetryBus::failed_sinks()` exposes a separate count. See
../../docs/TELEMETRY.md for drain behavior and blocking/silent-failure limits.

See [source workspace](../README.md).
