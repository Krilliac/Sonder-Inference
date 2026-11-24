"""Tests for GPU inventory, ADR-020 scheduling metadata, and live telemetry callbacks."""

from __future__ import annotations

import ctypes
from dataclasses import FrozenInstanceError
import json
import threading

import pytest

import sonder_inference as si
from sonder_inference import _structs as S
from sonder_inference._lib import Library


def test_gpu_and_request_metadata_struct_layouts():
    assert ctypes.sizeof(S.CGpuInfo) == S.GPU_INFO_SIZE == 496
    assert S.CGpuInfo.vendor_id.offset == 420
    assert S.CGpuInfo.device_id.offset == 424
    assert S.CGpuInfo.dedicated_total_bytes.offset == 432
    assert S.CGpuInfo.has_dedicated_total.offset == 472
    assert S.CGpuInfo.is_integrated_uma.offset == 492
    assert S.REQUEST_METADATA_V1_SIZE == 16
    assert S.REQUEST_METADATA_SIZE == 24
    assert S.CRequestMetadata.parent_request_id.offset == 8
    assert S.CRequestMetadata.workload_class.offset == 16
    assert S.CRequestMetadata.priority.offset == 20


def test_list_gpu_devices_matches_native_inventory(lib):
    devices = si.list_gpu_devices(lib)
    assert isinstance(devices, list)
    assert len(devices) == int(lib.cdll.sonder_gpu_device_count())
    for gpu in devices:
        assert isinstance(gpu, si.GpuInfo)
        assert gpu.id and gpu.vendor and gpu.source
        assert isinstance(gpu.name, str)
        assert isinstance(gpu.backend_hint, str)
        assert isinstance(gpu.vendor_id, int) and gpu.vendor_id >= 0
        assert isinstance(gpu.device_id, int) and gpu.device_id >= 0
        assert isinstance(gpu.is_integrated_uma, bool)
        for field_name in (
            "dedicated_total_bytes",
            "dedicated_free_bytes",
            "shared_total_bytes",
            "effective_vram_mib",
            "effective_free_vram_mib",
        ):
            val = getattr(gpu, field_name)
            assert val is None or (isinstance(val, int) and val >= 0)
        with pytest.raises(FrozenInstanceError):
            gpu.name = "mutated"  # type: ignore[misc]

    # Verify C ABI validation errors directly through the loaded library.
    raw = S.CGpuInfo()
    raw.struct_size = S.GPU_INFO_SIZE - 1
    assert lib.cdll.sonder_gpu_device_info(0, ctypes.byref(raw)) == int(si.Status.INVALID_ARGUMENT)
    raw.struct_size = S.GPU_INFO_SIZE
    assert lib.cdll.sonder_gpu_device_info(len(devices), ctypes.byref(raw)) == int(si.Status.NOT_FOUND)


def test_list_gpu_devices_missing_export_raises_unsupported(lib):
    class WithoutGpuExports:
        def __getattr__(self, name):
            if name in ("sonder_gpu_device_count", "sonder_gpu_device_info"):
                raise AttributeError(name)
            return getattr(lib.cdll, name)

    legacy = Library(WithoutGpuExports(), lib.path)
    with pytest.raises(si.UnsupportedError, match="sonder_gpu_device"):
        si.list_gpu_devices(legacy)


def test_engine_telemetry_callback_receives_events_replaces_and_stops_on_clear(lib):
    lock = threading.Lock()
    lines1: list[str] = []
    lines2: list[str] = []

    def cb1(line: str) -> None:
        with lock:
            lines1.append(line)

    def cb2(line: str) -> None:
        with lock:
            lines2.append(line)

    with si.Engine(telemetry_level=si.TelemetryLevel.STANDARD, telemetry_callback=cb1) as engine:
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(4)) as session:
                r1 = session.generate("telemetry callback first")
                assert r1.completed

                # Replacing flushes cb1 and routes subsequent events to cb2.
                engine.set_telemetry_callback(cb2)
                with lock:
                    snap1 = list(lines1)
                    count1_after_switch = len(lines1)

                assert snap1
                events1 = [json.loads(line) for line in snap1]
                assert all(e["schema"] == "sonder.observatory.event/1" for e in events1)
                types1 = {e["event_type"] for e in events1}
                assert {"request.queued", "request.started", "inference.token.generated", "request.completed"} <= types1

                r2 = session.generate("telemetry callback second")
                assert r2.completed

                # Clearing flushes cb2 and detaches it before further generations.
                engine.set_telemetry_callback(None)
                with lock:
                    snap2 = list(lines2)
                    count2_after_clear = len(lines2)

                assert snap2
                events2 = [json.loads(line) for line in snap2]
                assert all(e["schema"] == "sonder.observatory.event/1" for e in events2)
                assert any(e["event_type"] == "request.completed" for e in events2)

                r3 = session.generate("telemetry callback after clear")
                assert r3.completed

    with lock:
        assert len(lines1) == count1_after_switch
        assert len(lines2) == count2_after_clear


def test_telemetry_callback_contains_python_exceptions_and_validates_type(lib):
    with pytest.raises(TypeError, match="callable"):
        si.Engine(telemetry_level=si.TelemetryLevel.OFF, telemetry_callback="not-callable")  # type: ignore[arg-type]

    seen_errors = [0]
    after_lines: list[str] = []

    def boom(_line: str) -> None:
        seen_errors[0] += 1
        raise RuntimeError("callback failure must not escape")

    with si.Engine(telemetry_level=si.TelemetryLevel.METRICS) as engine:
        with pytest.raises(TypeError, match="callable"):
            engine.set_telemetry_callback(123)  # type: ignore[arg-type]

        engine.set_telemetry_callback(boom)
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(3)) as session:
                assert session.generate("survive callback exception").completed
                engine.set_telemetry_callback(after_lines.append)
                assert seen_errors[0] > 0
                assert session.generate("healthy after exception").completed
                engine.set_telemetry_callback(None)

    assert any(json.loads(line)["event_type"] == "request.completed" for line in after_lines)


def test_telemetry_callback_missing_export_raises_unsupported(lib):
    class WithoutTelemetryCallback:
        def __getattr__(self, name):
            if name == "sonder_engine_set_telemetry_callback":
                raise AttributeError(name)
            return getattr(lib.cdll, name)

    legacy = Library(WithoutTelemetryCallback(), lib.path)
    with pytest.raises(si.UnsupportedError, match="sonder_engine_set_telemetry_callback"):
        si.Engine(library=legacy, telemetry_level=si.TelemetryLevel.OFF, telemetry_callback=lambda _line: None)
    with si.Engine(library=legacy, telemetry_level=si.TelemetryLevel.OFF) as engine:
        with pytest.raises(si.UnsupportedError, match="sonder_engine_set_telemetry_callback"):
            engine.set_telemetry_callback(lambda _line: None)


def test_request_metadata_scheduling_fields_and_validation(lib):
    captured: list[str] = []
    with si.Engine(telemetry_level=si.TelemetryLevel.METRICS, telemetry_callback=captured.append) as engine:
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(3)) as session:
                bg = si.RequestMetadata(
                    parent_request_id="sched:bg",
                    workload_class=si.WorkloadClass.BACKGROUND_INDEXING,
                    priority=0,
                )
                assert session.generate("background work", metadata=bg).completed

                inter = si.RequestMetadata(
                    parent_request_id="sched:inter",
                    workload_class=si.WorkloadClass.INTERACTIVE_USER,
                    priority=16,
                )
                assert session.chat([si.ChatMessage("user", "interactive work")], metadata=inter).completed

                impl = si.RequestMetadata(
                    parent_request_id="sched:subagent",
                    workload_class=si.WorkloadClass.IMPLEMENTATION_WORKER,
                    priority=0,
                )
                assert session.generate("subagent work", metadata=impl).completed

                maint = si.RequestMetadata(workload_class=si.WorkloadClass.MAINTENANCE, priority=-16)
                assert session.generate("maintenance work", metadata=maint).completed

                assert int(si.WorkloadClass.DEFAULT) == 0
                assert int(si.WorkloadClass.INTERACTIVE_USER) == 1
                assert int(si.WorkloadClass.MAINTENANCE) == 7

                for bad_workload in (-2, -1, 8, 99, True, "fast"):
                    with pytest.raises(si.InvalidArgumentError, match="workload_class"):
                        session.generate(
                            "bad workload",
                            metadata=si.RequestMetadata(workload_class=bad_workload),  # type: ignore[arg-type]
                        )

                for bad_priority in (-17, 17, 100, True, 1.5):
                    with pytest.raises(si.InvalidArgumentError, match="priority"):
                        session.generate(
                            "bad priority",
                            metadata=si.RequestMetadata(priority=bad_priority),  # type: ignore[arg-type]
                        )

    events = [json.loads(line) for line in captured]
    queued = [e for e in events if e["event_type"] == "request.queued"]
    assert len(queued) == 4
    assert queued[0]["attributes"].get("priority_class") == "background"
    assert queued[1]["attributes"].get("priority_class") == "interactive"
    assert queued[2]["attributes"].get("priority_class") == "subagent"
    assert queued[3]["attributes"].get("priority_class") == "background"


def test_telemetry_callback_reentrant_close_rejected_and_concurrent_replace_safe(lib):
    reentrant_errors: list[BaseException] = []
    errors_lock = threading.Lock()

    with si.Engine(telemetry_level=si.TelemetryLevel.METRICS) as engine:
        def try_reentrant_close(_line: str) -> None:
            try:
                engine.close()
            except BaseException as exc:  # noqa: BLE001
                with errors_lock:
                    reentrant_errors.append(exc)

        engine.set_telemetry_callback(try_reentrant_close)
        engine.register_mock_backend()
        with engine.load_model("mock", "mock:tiny") as model:
            with engine.create_session(model, si.SamplingConfig.greedy(3)) as session:
                assert session.generate("reentrant close check").completed
                engine.set_telemetry_callback(None)
                assert not engine.closed
                with errors_lock:
                    assert reentrant_errors
                    assert all(isinstance(exc, si.InvalidStateError) for exc in reentrant_errors)

                # Concurrent set_telemetry_callback calls share one permanent trampoline.
                stop = threading.Event()
                worker_errors: list[BaseException] = []

                def churn(tid: int) -> None:
                    try:
                        for i in range(12):
                            if i % 3 == 2:
                                engine.set_telemetry_callback(None)
                            else:
                                engine.set_telemetry_callback(lambda _line, _t=tid: None)
                    except BaseException as exc:  # noqa: BLE001
                        worker_errors.append(exc)

                def generate_loop() -> None:
                    try:
                        while not stop.is_set():
                            session.generate("concurrent telemetry")
                    except BaseException as exc:  # noqa: BLE001
                        worker_errors.append(exc)

                gen_thread = threading.Thread(target=generate_loop)
                churn_threads = [threading.Thread(target=churn, args=(i,)) for i in range(3)]
                gen_thread.start()
                for t in churn_threads:
                    t.start()
                for t in churn_threads:
                    t.join(10)
                    assert not t.is_alive()
                stop.set()
                gen_thread.join(10)
                assert not gen_thread.is_alive()
                assert not worker_errors
                assert len(engine._telemetry_trampoline_box) == 1
                engine.set_telemetry_callback(None)
