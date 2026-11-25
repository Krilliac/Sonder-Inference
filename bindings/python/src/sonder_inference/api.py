"""Pythonic wrappers over the Sonder Inference C ABI."""

from __future__ import annotations

import codecs
import contextlib
import ctypes
import enum
import math
import os
import queue
import re
import threading
import weakref
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, TypeVar, Union

from . import _structs as S
from ._lib import Library, PathLike, get_library
from .errors import (
    InternalError,
    InvalidArgumentError,
    InvalidStateError,
    SonderError,
    Status,
    UnsupportedError,
    error_for_status,
)

TokenCallback = Callable[[str], Optional[bool]]
TelemetryCallback = Callable[[str], None]
LogitBiasInput = Union[Mapping[int, float], Sequence[tuple[int, float]]]

_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1
_UINT64_MAX = 2**64 - 1
_CORRELATION_ID_RE = re.compile(rf"[A-Za-z0-9._:-]{{1,{S.MAX_CORRELATION_ID_BYTES}}}", re.ASCII)
_H = TypeVar("_H", bound="_Handle")


class TelemetryLevel(enum.IntEnum):
    OFF = 0
    METRICS = 1
    STANDARD = 2
    DEEP = 3


class Outcome(enum.IntEnum):
    NONE = 0
    COMPLETED = 1
    CANCELLED = 2
    FAILED = 3


class WorkloadClass(enum.IntEnum):
    """Mirror of ``sonder_workload_class`` (ADR-020 scheduling classes)."""

    DEFAULT = 0
    INTERACTIVE_USER = 1
    OWNER_ORCHESTRATOR = 2
    CRITIC_VERIFICATION = 3
    IMPLEMENTATION_WORKER = 4
    RESEARCH_WORKER = 5
    BACKGROUND_INDEXING = 6
    MAINTENANCE = 7


def _check(lib: Library, rc: int) -> None:
    if rc != 0:
        detail = lib.last_error() or lib.status_string(rc) or f"sonder status {rc}"
        raise error_for_status(rc, detail)


def abi_version() -> int:
    """C ABI version reported by the loaded library."""
    return get_library().abi_version


def version() -> str:
    """Library version string."""
    return get_library().version


@dataclass(frozen=True)
class GpuInfo:
    """One GPU adapter reported by the native inventory.

    Memory fields are ``None`` when the platform could not determine them
    (distinct from a known zero reading).
    """

    id: str
    name: str
    vendor: str
    backend_hint: str
    source: str
    vendor_id: int
    device_id: int
    dedicated_total_bytes: Optional[int]
    dedicated_free_bytes: Optional[int]
    shared_total_bytes: Optional[int]
    effective_vram_mib: Optional[int]
    effective_free_vram_mib: Optional[int]
    is_integrated_uma: bool


def list_gpu_devices(library: Optional[Library] = None) -> list[GpuInfo]:
    """Return the live host GPU inventory reported by the native library."""
    lib = library or get_library()
    if not (lib.has_symbol("sonder_gpu_device_count") and lib.has_symbol("sonder_gpu_device_info")):
        raise UnsupportedError("this library does not export sonder_gpu_device_count/sonder_gpu_device_info")
    count = int(lib.cdll.sonder_gpu_device_count())
    devices: list[GpuInfo] = []
    for index in range(count):
        raw = S.CGpuInfo()
        raw.struct_size = ctypes.sizeof(S.CGpuInfo)
        rc = lib.cdll.sonder_gpu_device_info(index, ctypes.byref(raw))
        if rc == int(Status.NOT_FOUND):
            # Live inventory shrank between count and info calls; stop cleanly.
            break
        _check(lib, rc)
        devices.append(GpuInfo(
            id=raw.id.decode("utf-8", "replace"),
            name=raw.name.decode("utf-8", "replace"),
            vendor=raw.vendor.decode("utf-8", "replace"),
            backend_hint=raw.backend_hint.decode("utf-8", "replace"),
            source=raw.source.decode("utf-8", "replace"),
            vendor_id=int(raw.vendor_id),
            device_id=int(raw.device_id),
            dedicated_total_bytes=int(raw.dedicated_total_bytes) if raw.has_dedicated_total else None,
            dedicated_free_bytes=int(raw.dedicated_free_bytes) if raw.has_dedicated_free else None,
            shared_total_bytes=int(raw.shared_total_bytes) if raw.has_shared_total else None,
            effective_vram_mib=int(raw.effective_vram_mib) if raw.has_effective_vram else None,
            effective_free_vram_mib=int(raw.effective_free_vram_mib) if raw.has_effective_free_vram else None,
            is_integrated_uma=bool(raw.is_integrated_uma),
        ))
    return devices


# ---------------------------------------------------------------------------
# Sampling


def _int32(name: str, v: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"{name} must be an int, got {type(v).__name__}")
    if not _INT32_MIN <= v <= _INT32_MAX:
        raise InvalidArgumentError(f"{name} is outside the int32 range: {v}")
    return v


def _cstr(name: str, v: str) -> bytes:
    """Encode a str for a ``const char*`` parameter. The C side stops at the
    first NUL, so an embedded one would silently truncate the value."""
    if not isinstance(v, str):
        raise TypeError(f"{name} must be a str, got {type(v).__name__}")
    if "\0" in v:
        raise ValueError(f"{name} contains an embedded NUL character")
    return v.encode("utf-8")


def _cstr_path(name: str, v: PathLike) -> bytes:
    """Encode a ``str`` or ``os.PathLike[str]`` for a ``const char*`` path parameter."""
    if isinstance(v, os.PathLike):
        raw = os.fspath(v)
        if not isinstance(raw, str):
            raise TypeError(f"{name} must be a str or str PathLike, got {type(v).__name__}")
        return _cstr(name, raw)
    return _cstr(name, v)


def _float(name: str, v: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(v).__name__}")
    f = float(v)
    if math.isfinite(f) and math.isinf(ctypes.c_float(f).value):
        raise InvalidArgumentError(f"{name} overflows float32 range: {v}")
    return f


def _bias_items(bias: Optional[LogitBiasInput]) -> list[tuple[int, float]]:
    if bias is None:
        return []
    if isinstance(bias, (str, bytes, bytearray)):
        raise TypeError(f"logit_bias must be a mapping or sequence of (token, bias) pairs, got {type(bias).__name__}")
    items = bias.items() if isinstance(bias, Mapping) else bias
    out = []
    for i, entry in enumerate(items):
        try:
            token, value = entry
        except (TypeError, ValueError):
            raise TypeError(f"logit_bias[{i}] must be a (token, bias) pair") from None
        out.append((_int32(f"logit_bias[{i}].token", token), _float(f"logit_bias[{i}].bias", value)))
    return out


_EXTENDED_FIELDS = ("typical_p", "presence_penalty", "frequency_penalty", "repeat_last_n", "num_ctx")


@dataclass
class SamplingConfig:
    """Sampling policy. ``None`` means "use the library default".

    Ranges are validated by the library (see :meth:`validate`); the
    appended fields (``typical_p``, penalties, ``repeat_last_n``,
    ``num_ctx``, ``logit_bias``) need a library that supports them.
    ``logit_bias`` maps token id -> additive bias (``-math.inf`` bans).
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    typical_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repeat_last_n: Optional[int] = None
    num_ctx: Optional[int] = None
    logit_bias: Optional[LogitBiasInput] = None

    @classmethod
    def greedy(cls, max_tokens: int = 128, seed: int = 42) -> "SamplingConfig":
        """Deterministic preset (mirrors ``SamplingConfig::greedy`` in C++)."""
        return cls(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, repeat_penalty=1.0,
                   seed=seed, max_tokens=max_tokens)

    @classmethod
    def defaults(cls, library: Optional[Library] = None) -> "SamplingConfig":
        """The library's defaults, every supported field filled in.

        Against a library that predates the appended fields (see
        ``Library.supports_extended_sampling``) those fields stay ``None``:
        its init never wrote them, and setting them would make the config
        unusable with that library.
        """
        lib = library or get_library()
        c = S.CSamplingConfig()
        lib.init_sampling(c)
        cfg = cls(temperature=c.temperature, top_p=c.top_p, top_k=c.top_k, min_p=c.min_p,
                  repeat_penalty=c.repeat_penalty, seed=int(c.seed) if c.has_seed else None,
                  max_tokens=c.max_tokens)
        if lib.supports_extended_sampling:
            cfg.typical_p = c.typical_p
            cfg.presence_penalty = c.presence_penalty
            cfg.frequency_penalty = c.frequency_penalty
            cfg.repeat_last_n = c.repeat_last_n
            cfg.num_ctx = c.num_ctx
        return cfg

    def replace(self, **changes: Any) -> "SamplingConfig":
        return replace(self, **changes)

    def uses_extended_fields(self) -> bool:
        return any(getattr(self, n) is not None for n in _EXTENDED_FIELDS) or bool(self.logit_bias)

    def _to_c(self, lib: Library, struct_size: Optional[int] = None) -> "_CSampling":
        """Marshal into the C struct. Chooses struct_size by library support
        unless ``struct_size`` is given (tests use this to act as an old caller)."""
        c = S.CSamplingConfig()
        lib.init_sampling(c)
        if struct_size is None:
            if lib.supports_extended_sampling:
                struct_size = S.SAMPLING_CONFIG_SIZE
            elif self.uses_extended_fields():
                raise UnsupportedError(
                    f"{lib.path} predates typical_p/penalties/repeat_last_n/num_ctx/logit_bias; "
                    "rebuild the library or leave those fields unset")
            else:
                struct_size = S.SAMPLING_CONFIG_V1_SIZE
        c.struct_size = struct_size
        for name in ("temperature", "top_p", "min_p", "repeat_penalty", "typical_p",
                     "presence_penalty", "frequency_penalty"):
            v = getattr(self, name)
            if v is not None:
                setattr(c, name, _float(name, v))
        for name in ("top_k", "max_tokens", "repeat_last_n", "num_ctx"):
            v = getattr(self, name)
            if v is not None:
                setattr(c, name, _int32(name, v))
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= _UINT64_MAX:
                raise InvalidArgumentError(f"seed must be an int in [0, 2**64), got {self.seed!r}")
            c.has_seed = 1
            c.seed = self.seed
        items = _bias_items(self.logit_bias)
        array = None
        if items:
            array = (S.CLogitBias * len(items))(*[S.CLogitBias(t, b) for t, b in items])
            c.logit_bias = ctypes.cast(array, ctypes.POINTER(S.CLogitBias))
            c.logit_bias_count = len(items)
        return _CSampling(c, array)

    def validate(self, library: Optional[Library] = None) -> None:
        """Raise :class:`InvalidArgumentError` if the library rejects the config."""
        lib = library or get_library()
        m = self._to_c(lib)
        _check(lib, lib.cdll.sonder_sampling_config_validate(ctypes.byref(m.struct)))

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["logit_bias"] = dict(_bias_items(self.logit_bias)) if self.logit_bias else None
        return d


@dataclass
class _CSampling:
    struct: S.CSamplingConfig
    keepalive: Any = None  # logit_bias array; must outlive the C call


# ---------------------------------------------------------------------------
# Results


@dataclass(frozen=True)
class GenerationResult:
    text: str
    outcome: Outcome
    prompt_tokens: int
    completion_tokens: int
    chunks: int
    ttft_ms: Optional[float]  # None when no output was produced
    total_ms: float

    @property
    def completed(self) -> bool:
        return self.outcome == Outcome.COMPLETED

    @property
    def cancelled(self) -> bool:
        return self.outcome == Outcome.CANCELLED


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


# ---------------------------------------------------------------------------
# Handles


class _Handle:
    """Owns one C handle; close() is idempotent, thread-safe and also runs at GC.

    The C destroy functions free the handle immediately, so destroying it
    while another thread is inside a C call on it is a use-after-free.
    Every C call therefore runs inside :meth:`_use`, and :meth:`close`
    marks the handle closed (new calls raise :class:`InvalidStateError`),
    interrupts in-flight calls (:meth:`_interrupt`) and destroys only once
    they have all returned.
    """

    _destroy_fn = ""
    _INTERRUPT_INTERVAL_S = 0.05

    def __init__(self, lib: Library, ptr: int) -> None:
        if not ptr:
            raise InternalError(f"{type(self).__name__} received a null native handle")
        self._lib = lib
        self._ptr: Optional[int] = ptr
        self._cond = threading.Condition()
        self._closing = False
        self._active: dict[int, int] = {}  # thread ident -> nesting depth of in-flight calls
        self._destroy_lock = threading.Lock()
        self._finalizer: weakref.finalize[Any, Any] = weakref.finalize(
            self, getattr(lib.cdll, self._destroy_fn), ctypes.c_void_p(ptr))

    @property
    def closed(self) -> bool:
        return self._closing

    @contextlib.contextmanager
    def _use(self) -> Iterator[ctypes.c_void_p]:
        """Hold the handle open for the duration of one C call."""
        tid = threading.get_ident()
        with self._cond:
            if self._closing:
                raise InvalidStateError(f"{type(self).__name__} is closed")
            self._active[tid] = self._active.get(tid, 0) + 1
            ptr = self._ptr
        try:
            yield ctypes.c_void_p(ptr)
        finally:
            with self._cond:
                if self._active[tid] == 1:
                    del self._active[tid]
                else:
                    self._active[tid] -= 1
                self._cond.notify_all()

    def _in_own_call(self) -> bool:
        with self._cond:
            return threading.get_ident() in self._active

    def _interrupt(self, handle: ctypes.c_void_p) -> None:
        """Ask in-flight calls to return early (called while they still hold the handle)."""

    def _check_not_reentrant(self) -> None:
        if self._in_own_call():
            raise InvalidStateError(
                f"{type(self).__name__}.close() called from inside one of its own calls "
                "(e.g. a token callback); return False from the callback or call cancel() instead")

    def _begin_close(self) -> bool:
        """Refuse new calls. Returns False if the handle was already closing."""
        with self._cond:
            first = not self._closing
            self._closing = True
            return first

    def _drain(self) -> None:
        """Interrupt and wait until no call is in flight on this handle."""
        while True:
            # Concurrent closers share the destruction guard. Keep each
            # interrupt inside it, but release it before waiting for calls.
            with self._destroy_lock:
                with self._cond:
                    if not self._active:
                        return
                    handle = ctypes.c_void_p(self._ptr)
                self._interrupt(handle)
            with self._cond:
                if self._active:
                    self._cond.wait(self._INTERRUPT_INTERVAL_S)

    def _destroy(self) -> None:
        # Concurrent closers all block here until the one destroy completes.
        with self._destroy_lock:
            self._finalizer()
            with self._cond:
                self._ptr = None

    def close(self) -> None:
        self._check_not_reentrant()
        self._begin_close()
        self._drain()
        self._destroy()

    def __enter__(self: _H) -> _H:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        ptr = self._ptr
        state = "closed" if self._closing or ptr is None else hex(ptr)
        return f"<{type(self).__name__} {state}>"


@dataclass(frozen=True)
class SessionMetadata:
    """Caller correlation IDs. None keeps defaults; IDs follow the HTTP policy.

    Identifiers appear in telemetry envelopes independently of text capture.
    They must not contain prompts or secrets. Older ABI-v1 libraries can still
    create default sessions; explicit metadata requires the additive export.
    """

    session_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None

    def _to_c(self) -> "S.CSessionMetadata":
        record = S.CSessionMetadata()
        record.struct_size = ctypes.sizeof(record)
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and (not isinstance(value, str) or
                                      _CORRELATION_ID_RE.fullmatch(value) is None):
                raise InvalidArgumentError(f"{field.name} must match [A-Za-z0-9._:-]{{1,128}}")
            setattr(record, field.name, value.encode("ascii") if value is not None else None)
        return record


@dataclass(frozen=True)
class RequestMetadata:
    """Caller parent lineage and ADR-020 scheduling metadata for one request.

    None leaves the parent absent and workload_class at the session default.
    Identifiers must not contain prompts, secrets or personal data. Explicit
    metadata needs the additive export; session run IDs and engine-generated
    request IDs remain unchanged.
    """

    parent_request_id: Optional[str] = None
    workload_class: Optional[Union[WorkloadClass, int]] = None
    priority: int = 0

    def _to_c(self) -> "S.CRequestMetadata":
        value = self.parent_request_id
        if value is not None and (not isinstance(value, str) or
                                  _CORRELATION_ID_RE.fullmatch(value) is None):
            raise InvalidArgumentError("parent_request_id must match [A-Za-z0-9._:-]{1,128}")
        if self.workload_class is None:
            workload = int(WorkloadClass.DEFAULT)
        elif isinstance(self.workload_class, bool) or not isinstance(self.workload_class, int):
            raise InvalidArgumentError("workload_class must be WorkloadClass, int, or None")
        else:
            workload = int(self.workload_class)
            if workload < int(WorkloadClass.DEFAULT) or workload > int(WorkloadClass.MAINTENANCE):
                raise InvalidArgumentError("workload_class must be WorkloadClass.DEFAULT (0) or in [1, 7]")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise InvalidArgumentError("priority must be an int in [-16, 16]")
        if not -16 <= self.priority <= 16:
            raise InvalidArgumentError("priority must be an int in [-16, 16]")
        record = S.CRequestMetadata()
        record.struct_size = ctypes.sizeof(record)
        record.parent_request_id = value.encode("ascii") if value is not None else None
        record.workload_class = workload
        record.priority = self.priority
        return record


def _make_telemetry_trampoline(cb_lock: threading.RLock, state: dict[str, Any]) -> Any:
    """Create the single permanent ctypes callback trampoline for an Engine.

    Closes only over ``cb_lock`` and ``state`` (never ``Engine``) so the finalizer
    never introduces a reference cycle preventing garbage collection.
    """
    def trampoline(_user: Any, ptr: Optional[int], length: int) -> None:
        try:
            with cb_lock:
                cb = state["callback"]
                if cb is None:
                    return
                prev_tid = state["active_tid"]
                state["active_tid"] = threading.get_ident()
            try:
                line = ctypes.string_at(ptr, length).decode("utf-8", "replace") if length and ptr else ""
                cb(line)
            finally:
                with cb_lock:
                    state["active_tid"] = prev_tid
        except BaseException:  # noqa: BLE001 - never escape across ctypes/C ABI
            pass
    return S.TELEMETRY_CALLBACK(trampoline)


def _destroy_engine(destroy_fn: Any, ptr: ctypes.c_void_p,
                    trampoline_box: list[Any], state: dict[str, Any]) -> None:
    try:
        destroy_fn(ptr)
    finally:
        state["callback"] = None
        state["native_registered"] = False
        trampoline_box.clear()


class Engine(_Handle):
    """An inference engine. Closing it closes its models and sessions first."""

    _destroy_fn = "sonder_engine_destroy"

    def __init__(self, telemetry_level: Optional[Union[TelemetryLevel, int]] = None,
                 telemetry_jsonl_path: Optional[PathLike] = None, capture_text: bool = False,
                 library: Optional[Library] = None,
                 telemetry_callback: Optional[TelemetryCallback] = None) -> None:
        lib = library or get_library()
        if telemetry_callback is not None:
            if not callable(telemetry_callback):
                raise TypeError("telemetry_callback must be callable or None")
            if not lib.has_symbol("sonder_engine_set_telemetry_callback"):
                raise UnsupportedError("this library does not export sonder_engine_set_telemetry_callback")
        opts = S.EngineOptions()
        lib.cdll.sonder_engine_options_init(ctypes.byref(opts))
        if telemetry_level is not None:
            if isinstance(telemetry_level, bool) or not isinstance(telemetry_level, int):
                raise TypeError(
                    f"telemetry_level must be TelemetryLevel, int, or None, got {type(telemetry_level).__name__}")
            try:
                opts.telemetry_level = int(TelemetryLevel(telemetry_level))
            except ValueError:
                raise InvalidArgumentError(f"invalid telemetry_level: {telemetry_level!r}") from None
        path_bytes = (
            (_cstr_path("telemetry_jsonl_path", telemetry_jsonl_path) or None)
            if telemetry_jsonl_path is not None
            else None
        )
        opts.telemetry_jsonl_path = path_bytes
        opts.capture_text = 1 if capture_text else 0
        out = ctypes.c_void_p()
        _check(lib, lib.cdll.sonder_engine_create(ctypes.byref(opts), ctypes.byref(out)))
        if not out.value:
            raise InternalError("sonder_engine_create returned a null handle")
        super().__init__(lib, out.value)
        self._telemetry_cb_lock = threading.RLock()
        self._telemetry_set_lock = threading.RLock()
        self._telemetry_state: dict[str, Any] = {
            "callback": None,
            "active_tid": None,
            "native_registered": False,
        }
        self._telemetry_trampoline_box: list[Any] = []
        # Keep the permanent ctypes trampoline pinned until sonder_engine_destroy()
        # has flushed and cleared the native sink.
        self._finalizer.detach()
        self._finalizer = weakref.finalize(
            self, _destroy_engine, lib.cdll.sonder_engine_destroy,
            ctypes.c_void_p(out.value), self._telemetry_trampoline_box, self._telemetry_state)
        self._children: "weakref.WeakSet[_Handle]" = weakref.WeakSet()
        if telemetry_callback is not None:
            try:
                self.set_telemetry_callback(telemetry_callback)
            except BaseException:
                self.close()
                raise

    def _in_telemetry_callback(self) -> bool:
        cb_lock = getattr(self, "_telemetry_cb_lock", None)
        if cb_lock is None:
            return False
        with cb_lock:
            return bool(self._telemetry_state["active_tid"] == threading.get_ident())

    def _check_not_reentrant(self) -> None:
        super()._check_not_reentrant()
        if self._in_telemetry_callback():
            raise InvalidStateError(
                "Engine.close() called from inside its own telemetry callback; "
                "call set_telemetry_callback(None) or close the engine from another thread instead")

    @property
    def library(self) -> Library:
        return self._lib

    @property
    def device_count(self) -> int:
        with self._use() as h:
            return int(self._lib.cdll.sonder_engine_device_count(h))

    def set_telemetry_callback(self, callback: Optional[TelemetryCallback]) -> None:
        """Register, replace, or clear (``None``) a live Observatory JSONL sink.

        ``callback(json_line)`` receives one serialized
        ``sonder.observatory.event/1`` line (no trailing newline) on the
        engine's telemetry writer thread and must not block indefinitely.
        Calling :meth:`close` from inside the callback raises
        :class:`InvalidStateError`.
        """
        if callback is not None and not callable(callback):
            raise TypeError("telemetry_callback must be callable or None")
        if not self._lib.has_symbol("sonder_engine_set_telemetry_callback"):
            raise UnsupportedError("this library does not export sonder_engine_set_telemetry_callback")
        with self._use() as h:
            if self._in_telemetry_callback():
                if callback is None:
                    with self._telemetry_cb_lock:
                        self._telemetry_state["callback"] = None
                    _check(self._lib, self._lib.cdll.sonder_engine_set_telemetry_callback(
                        h, S.TELEMETRY_CALLBACK(), None))
                    with self._telemetry_cb_lock:
                        self._telemetry_state["native_registered"] = False
                    return
                with self._telemetry_cb_lock:
                    self._telemetry_state["callback"] = callback
                    need_register = not self._telemetry_state["native_registered"]
                    if need_register:
                        if not self._telemetry_trampoline_box:
                            self._telemetry_trampoline_box.append(
                                _make_telemetry_trampoline(self._telemetry_cb_lock, self._telemetry_state))
                        trampoline = self._telemetry_trampoline_box[0]
                    else:
                        trampoline = None
                if need_register:
                    try:
                        _check(self._lib, self._lib.cdll.sonder_engine_set_telemetry_callback(h, trampoline, None))
                    except BaseException:
                        with self._telemetry_cb_lock:
                            self._telemetry_state["callback"] = None
                        raise
                    with self._telemetry_cb_lock:
                        self._telemetry_state["native_registered"] = True
                return
            with self._telemetry_set_lock:
                if callback is None:
                    _check(self._lib, self._lib.cdll.sonder_engine_set_telemetry_callback(
                        h, S.TELEMETRY_CALLBACK(), None))
                    with self._telemetry_cb_lock:
                        self._telemetry_state["callback"] = None
                        self._telemetry_state["native_registered"] = False
                    return
                if not self._telemetry_trampoline_box:
                    self._telemetry_trampoline_box.append(
                        _make_telemetry_trampoline(self._telemetry_cb_lock, self._telemetry_state))
                trampoline = self._telemetry_trampoline_box[0]
                with self._telemetry_cb_lock:
                    was_registered = bool(self._telemetry_state["native_registered"])
                if was_registered:
                    # Detach first so queued events flush to the previous Python
                    # callback and any in-flight trampoline call finishes before swap.
                    _check(self._lib, self._lib.cdll.sonder_engine_set_telemetry_callback(
                        h, S.TELEMETRY_CALLBACK(), None))
                    with self._telemetry_cb_lock:
                        self._telemetry_state["native_registered"] = False
                with self._telemetry_cb_lock:
                    self._telemetry_state["callback"] = callback
                try:
                    _check(self._lib, self._lib.cdll.sonder_engine_set_telemetry_callback(h, trampoline, None))
                except BaseException:
                    with self._telemetry_cb_lock:
                        self._telemetry_state["callback"] = None
                    raise
                with self._telemetry_cb_lock:
                    self._telemetry_state["native_registered"] = True

    def register_mock_backend(self) -> None:
        """Register the deterministic MOCK backend ("mock"). Tests only."""
        with self._use() as h:
            _check(self._lib, self._lib.cdll.sonder_engine_register_mock_backend(h))

    def register_ollama_backend(self, base_url: Optional[str] = None) -> None:
        url = (_cstr("base_url", base_url) or None) if base_url is not None else None
        with self._use() as h:
            _check(self._lib, self._lib.cdll.sonder_engine_register_ollama_backend(h, url))

    def load_model(self, backend: str, model: str) -> "Model":
        backend_b, model_b = _cstr("backend", backend), _cstr("model", model)
        out = ctypes.c_void_p()
        with self._use() as h:
            _check(self._lib, self._lib.cdll.sonder_model_load(h, backend_b, model_b, ctypes.byref(out)))
            if not out.value:
                raise InternalError("sonder_model_load returned a null handle")
            # Registered before the call is released, so close() cannot miss it.
            m = Model(self, out.value)
            with self._cond:
                self._children.add(m)
        return m

    def create_session(self, model: "Model", sampling: Optional[SamplingConfig] = None,
                       *, metadata: Optional[SessionMetadata] = None) -> "Session":
        if not isinstance(model, Model):
            raise TypeError(f"model must be a Model, got {type(model).__name__}")
        if model.engine is not self:
            raise InvalidArgumentError("model belongs to a different engine")
        if sampling is not None and not isinstance(sampling, SamplingConfig):
            raise TypeError(f"sampling must be a SamplingConfig or None, got {type(sampling).__name__}")
        if metadata is not None:
            if not isinstance(metadata, SessionMetadata):
                raise InvalidArgumentError("metadata must be SessionMetadata or None")
            if not self._lib.has_symbol("sonder_session_create_with_metadata"):
                raise UnsupportedError("this library does not export sonder_session_create_with_metadata")
        c_metadata = metadata._to_c() if metadata is not None else None
        marshalled = sampling._to_c(self._lib) if sampling is not None else None
        out = ctypes.c_void_p()
        with self._use() as h, model._use() as mh:
            sampling_ptr = ctypes.byref(marshalled.struct) if marshalled else None
            if c_metadata is None:
                code = self._lib.cdll.sonder_session_create(h, mh, sampling_ptr, ctypes.byref(out))
            else:
                code = self._lib.cdll.sonder_session_create_with_metadata(
                    h, mh, sampling_ptr, ctypes.byref(c_metadata), ctypes.byref(out))
            _check(self._lib, code)
            if not out.value:
                raise InternalError("sonder_session_create returned a null handle")
            s = Session(self, model, out.value, sampling)
            with self._cond:
                self._children.add(s)
        return s

    def close(self) -> None:
        with self._cond:
            children = list(self._children)
        self._check_not_reentrant()
        for child in children:
            child._check_not_reentrant()
        self._begin_close()
        # No new engine calls can start; wait for load_model/create_session in
        # flight so every child they create is registered before we snapshot.
        self._drain()
        with self._cond:
            children = list(self._children)
        # Sessions (cancelling and waiting for their in-flight calls) before
        # models before the engine.
        for child in children:
            if isinstance(child, Session):
                child.close()
        for child in children:
            child.close()
        self._destroy()


class Model(_Handle):
    """A loaded model. Sessions keep the underlying model alive, so closing
    this handle while sessions exist is safe."""

    _destroy_fn = "sonder_model_release"

    def __init__(self, engine: Engine, ptr: int) -> None:
        super().__init__(engine._lib, ptr)
        self.engine = engine


class Session(_Handle):
    """A generation session. One generate or chat call at a time."""

    _destroy_fn = "sonder_session_destroy"

    def __init__(self, engine: Engine, model: Model, ptr: int, sampling: Optional[SamplingConfig]) -> None:
        super().__init__(engine._lib, ptr)
        self.engine = engine  # keeps the engine alive while the session is
        self.sampling = sampling
        self._stream_cancel_epoch = 0

    def generate(self, prompt: str, on_token: Optional[TokenCallback] = None, *,
                 metadata: Optional[RequestMetadata] = None) -> GenerationResult:
        """Run a request to completion.

        ``on_token(text)`` is called for each streamed chunk (UTF-8 decoded
        incrementally, so multi-byte characters are never split). Return
        ``False`` from it to stop early; an exception raised in it stops the
        generation and is re-raised here.
        """
        prompt_b = _cstr("prompt", prompt)
        function, args = self._request_args("sonder_session_generate", (prompt_b,), metadata)
        return self._request(function, args, on_token)

    def _request_args(self, name: str, args: tuple[Any, ...],
                      metadata: Optional[RequestMetadata]) -> tuple[Any, tuple[Any, ...]]:
        if metadata is None:
            return getattr(self._lib.cdll, name), args
        if not isinstance(metadata, RequestMetadata):
            raise InvalidArgumentError("metadata must be RequestMetadata or None")
        record = metadata._to_c()  # prepare owned bytes before any worker starts
        export = name + "_with_metadata"
        if not self._lib.has_symbol(export):
            raise UnsupportedError(f"this library does not export {export}")
        return getattr(self._lib.cdll, export), (*args, ctypes.pointer(record))

    def _request(self, function: Any, args: tuple[Any, ...],
                 on_token: Optional[TokenCallback]) -> GenerationResult:
        if on_token is not None and not callable(on_token):
            raise TypeError("on_token must be callable or None")
        lib = self._lib
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        parts: list[str] = []
        error: list[BaseException] = []
        stopped = [False]

        def deliver(text: str) -> int:
            parts.append(text)
            if on_token is not None and on_token(text) is False:
                stopped[0] = True
                return 1
            return 0

        def trampoline(_user: Any, ptr: Optional[int], length: int) -> int:
            if error or stopped[0]:
                return 1
            try:
                text = decoder.decode(ctypes.string_at(ptr, length)) if length and ptr else ""
                return deliver(text) if text else 0
            except BaseException as e:  # noqa: BLE001 - re-raised after the C call returns
                error.append(e)
                return 1

        c_callback = S.TOKEN_CALLBACK(trampoline)
        stats = S.CGenerationStats()
        stats.struct_size = ctypes.sizeof(S.CGenerationStats)
        # close() on another thread cancels this call and destroys the C
        # session only after it returns (see _Handle).
        with self._use() as handle:
            rc = function(handle, *args, c_callback, None, ctypes.byref(stats))
        if error:
            raise error[0]
        _check(lib, rc)
        tail = decoder.decode(b"", final=True)
        if tail and not stopped[0]:
            deliver(tail)
        ttft_ms = None if stats.ttft_ms < 0 or math.isnan(stats.ttft_ms) else stats.ttft_ms
        return GenerationResult(
            text="".join(parts), outcome=Outcome(stats.outcome), prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens, chunks=stats.chunks,
            ttft_ms=ttft_ms, total_ms=stats.total_ms)

    def stream(self, prompt: str, *, metadata: Optional[RequestMetadata] = None) -> "TokenStream":
        """Iterate over chunks as they are produced (generation runs on a
        worker thread). At most 64 chunks wait for the consumer; a full
        buffer pauses delivery without dropping text. Closing the stream
        early cancels the request. The final result still retains its text."""
        prompt_b = _cstr("prompt", prompt)  # fail here, not later on the worker thread
        function, args = self._request_args("sonder_session_generate", (prompt_b,), metadata)
        if self.closed:
            raise InvalidStateError("Session is closed")
        if metadata is None:
            return TokenStream(self, prompt)
        return TokenStream._from_request(self, function, args)

    def cancel(self) -> None:
        """Cancel the in-flight request (thread-safe; no-op when idle)."""
        with self._use() as h:
            # Wake a stream callback waiting for consumer capacity. Native
            # cancellation alone cannot interrupt Python while inside it.
            with self._cond:
                self._stream_cancel_epoch += 1
            _check(self._lib, self._lib.cdll.sonder_session_cancel(h))

    def _interrupt(self, handle: ctypes.c_void_p) -> None:
        # Thread-safe in the C ABI; repeated by _drain() until the in-flight
        # call returns, which also covers a call that had not yet started
        # its request when the first cancel arrived.
        self._lib.cdll.sonder_session_cancel(handle)

    def close(self) -> None:
        """Close the session. A generate()/chat()/stream() on another thread
        is cancelled, and the C session is destroyed only after it returns.
        Idempotent and thread-safe; later calls raise InvalidStateError."""
        super().close()

    def chat(self, messages: Iterable[Union[ChatMessage, Mapping[str, str]]],
             on_token: Optional[TokenCallback] = None, *,
             metadata: Optional[RequestMetadata] = None) -> GenerationResult:
        """Run text chat with 1 to 1024 messages, ending in user or tool.

        Accepts ChatMessage or mappings containing only role/content. Native
        backend chat is used when available, otherwise the engine formats a
        role-labelled prompt. Callback and cancellation match generate().
        Older ABI v1 libraries without the chat export raise UnsupportedError.
        """
        args = self._chat_args(messages)
        function, args = self._request_args("sonder_session_chat", args, metadata)
        return self._request(function, args, on_token)

    def chat_stream(self, messages: Iterable[Union[ChatMessage, Mapping[str, str]]], *,
                    metadata: Optional[RequestMetadata] = None) -> "TokenStream":
        """Iterate over structured chat chunks with the same 64-chunk bound
        and cancellation/lifetime behavior as stream(). Native input buffers
        are prepared before starting the worker. Close an abandoned iterator.
        """
        args = self._chat_args(messages)
        function, args = self._request_args("sonder_session_chat", args, metadata)
        if self.closed:
            raise InvalidStateError("Session is closed")
        return TokenStream._from_request(self, function, args)

    def _chat_args(self, messages: Iterable[Union[ChatMessage, Mapping[str, str]]]) -> tuple[Any, int]:
        if not self._lib.has_symbol("sonder_session_chat"):
            raise UnsupportedError("this library does not export sonder_session_chat")
        if isinstance(messages, (str, bytes, bytearray, Mapping, ChatMessage)):
            raise TypeError("messages must be an iterable of ChatMessage or role/content mappings")
        records: list[S.CChatMessage] = []
        for message in messages:
            if len(records) == S.MAX_CHAT_MESSAGES:
                raise InvalidArgumentError("chat accepts at most 1024 messages")
            if isinstance(message, ChatMessage):
                role, content = message.role, message.content
            elif isinstance(message, Mapping):
                if set(message) - {"role", "content"}:
                    raise UnsupportedError("chat mappings support only role and content")
                if "role" not in message or "content" not in message:
                    raise InvalidArgumentError("chat messages require role and content")
                role, content = message["role"], message["content"]
            else:
                raise TypeError("chat messages must be ChatMessage or role/content mappings")
            # ctypes records retain the encoded bytes; the pointer array keeps
            # the records alive until the synchronous native call returns.
            records.append(S.CChatMessage(ctypes.sizeof(S.CChatMessage),
                                          _cstr("role", role), _cstr("content", content)))
        if not records:
            raise InvalidArgumentError("chat requires at least one message")
        pointers = (ctypes.POINTER(S.CChatMessage) * len(records))(
            *(ctypes.pointer(record) for record in records))
        return pointers, len(records)


class _StreamState:
    """Worker-side state for :class:`TokenStream`.

    Holds no strong reference to the ``TokenStream`` instance so an abandoned
    iterator can be garbage-collected and cancelled instead of leaking its
    worker thread and wedging the session.
    """

    __slots__ = (
        "session",
        "queue",
        "closed",
        "done",
        "cancel_epoch",
        "exc",
        "result",
        "thread",
        "_weak_stream",
        "_call",
    )

    def __init__(self, stream: "TokenStream", session: Session,
                 make_call: Callable[[TokenCallback], Callable[[], GenerationResult]]) -> None:
        self.session = session
        self.queue: "queue.Queue[Union[str, object]]" = queue.Queue(maxsize=TokenStream._MAX_BUFFERED_CHUNKS)
        self.closed = threading.Event()
        self.done = threading.Event()
        with session._cond:
            self.cancel_epoch = session._stream_cancel_epoch
        self.exc: Optional[BaseException] = None
        self.result: Optional[GenerationResult] = None
        self._weak_stream = weakref.ref(stream)
        self._call: Optional[Callable[[], GenerationResult]] = make_call(self._dispatch_push)
        self.thread = threading.Thread(target=self.run, name="sonder-stream", daemon=True)

    def _dispatch_push(self, text: str) -> Optional[bool]:
        stream = self._weak_stream()
        if stream is not None and (
            type(stream)._push is not _DEFAULT_TOKEN_STREAM_PUSH or "_push" in stream.__dict__
        ):
            return stream._push(text)
        # Drop the strong reference before potentially blocking on queue.put so
        # GC can collect and cancel an abandoned stream while backpressured.
        del stream
        return self.push(text)

    def push(self, text: str) -> Optional[bool]:
        while not self.closed.is_set():
            with self.session._cond:
                cancelled = (self.session._closing or
                             self.session._stream_cancel_epoch != self.cancel_epoch)
            if cancelled:
                break
            try:
                self.queue.put(text, timeout=TokenStream._POLL_INTERVAL_S)
                return None
            except queue.Full:
                continue
        # Cancellation may have arrived before the native request started.
        # Reassert it while this callback is inside that request so its
        # terminal outcome is cancelled, rather than callback-stop completed.
        try:
            self.session.cancel()
        except SonderError:
            pass  # session close has already cancelled the native handle
        return False

    def run(self) -> None:
        call = self._call
        self._call = None
        try:
            assert call is not None
            self.result = call()
        except BaseException as e:  # noqa: BLE001 - surfaced to the consumer
            self.exc = e
        finally:
            # The native call has ended; retain its result, not its prepared
            # input buffers.
            del call
            # Completion must never wait for a consumer: close() joins this
            # worker even when the buffer is full. The event covers that case.
            try:
                self.queue.put_nowait(TokenStream._DONE)
            except queue.Full:
                pass
            self.done.set()


def _abandon_stream(state: _StreamState) -> None:
    """Finalizer for an unclosed TokenStream: signal stop and cancel native work."""
    state.closed.set()
    if not state.done.is_set():
        try:
            state.session.cancel()
        except BaseException:  # noqa: BLE001 - never escape a finalizer
            pass


class TokenStream:
    """Iterator over generated chunks; see :meth:`Session.stream` and :meth:`Session.chat_stream`."""

    _DONE = object()
    _MAX_BUFFERED_CHUNKS = 64
    _POLL_INTERVAL_S = 0.025

    def __init__(self, session: Session, prompt: str) -> None:
        _cstr("prompt", prompt)
        if session.closed:
            raise InvalidStateError("Session is closed")
        self._prompt = prompt
        self._start(session, lambda on_token: (lambda: session.generate(prompt, on_token=on_token)))

    @classmethod
    def _from_chat(cls, session: Session, args: tuple[Any, int]) -> "TokenStream":
        return cls._from_request(session, session._lib.cdll.sonder_session_chat, args)

    @classmethod
    def _from_request(cls, session: Session, function: Any, args: tuple[Any, ...]) -> "TokenStream":
        stream = cls.__new__(cls)
        stream._start(session, lambda on_token: (lambda: session._request(function, args, on_token)))
        return stream

    def _start(self, session: Session,
               make_call: Callable[[TokenCallback], Callable[[], GenerationResult]]) -> None:
        state = _StreamState(self, session, make_call)
        self._state = state
        self._session = session
        self._queue = state.queue
        self._closed = state.closed
        self._done = state.done
        self._cancel_epoch = state.cancel_epoch
        self._finished = False
        self._thread = state.thread
        self._finalizer: weakref.finalize[Any, Any] = weakref.finalize(self, _abandon_stream, state)
        state.thread.start()

    @property
    def result(self) -> Optional[GenerationResult]:
        return self._state.result

    def _push(self, text: str) -> Optional[bool]:
        return self._state.push(text)

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        if self._finished:
            raise StopIteration
        while True:
            if self._closed.is_set():
                raise StopIteration
            if self._done.is_set() and self._queue.empty():
                item: Union[str, object] = self._DONE
                break
            try:
                item = self._queue.get(timeout=self._POLL_INTERVAL_S)
                break
            except queue.Empty:
                continue
        if item is self._DONE:
            self._finalizer.detach()
            if self._thread is not threading.current_thread():
                self._thread.join()
            self._finished = True
            if self._state.exc is not None:
                raise self._state.exc
            raise StopIteration
        if self._closed.is_set():
            raise StopIteration
        assert isinstance(item, str)
        return item

    def close(self) -> None:
        """Stop early: cancels the request and waits for the worker."""
        if not self._finished:
            self._closed.set()
            self._finalizer.detach()
            if self._thread is not threading.current_thread():
                while self._thread.is_alive():
                    if not self._done.is_set():
                        try:
                            self._session.cancel()
                        except SonderError:
                            pass
                    self._thread.join(self._POLL_INTERVAL_S)
            self._finished = True

    def __enter__(self) -> "TokenStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


_DEFAULT_TOKEN_STREAM_PUSH = TokenStream._push


__all__ = [
    "ChatMessage", "Engine", "GenerationResult", "GpuInfo", "Model", "Outcome", "SamplingConfig", "Session",
    "RequestMetadata", "SessionMetadata", "TelemetryLevel", "TokenStream", "WorkloadClass",
    "abi_version", "list_gpu_devices", "version",
]
