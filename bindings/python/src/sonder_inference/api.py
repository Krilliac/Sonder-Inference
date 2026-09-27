"""Pythonic wrappers over the Sonder Inference C ABI."""

from __future__ import annotations

import codecs
import ctypes
import enum
import queue
import threading
import weakref
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, Union

from . import _structs as S
from ._lib import Library, get_library
from .errors import InvalidArgumentError, InvalidStateError, SonderError, UnsupportedError, error_for_status

TokenCallback = Callable[[str], Optional[bool]]
LogitBiasInput = Union[Mapping[int, float], Sequence[tuple[int, float]]]

_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1
_UINT64_MAX = 2**64 - 1


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


def _check(lib: Library, rc: int) -> None:
    if rc != 0:
        raise error_for_status(rc, lib.last_error() or f"sonder status {rc}")


def abi_version() -> int:
    """C ABI version reported by the loaded library."""
    return get_library().abi_version


def version() -> str:
    """Library version string."""
    return get_library().version


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


def _float(name: str, v: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(v).__name__}")
    return float(v)


def _bias_items(bias: Optional[LogitBiasInput]) -> list[tuple[int, float]]:
    if not bias:
        return []
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
        lib.cdll.sonder_sampling_config_init(ctypes.byref(c))
        cfg = cls(temperature=c.temperature, top_p=c.top_p, top_k=c.top_k, min_p=c.min_p,
                  repeat_penalty=c.repeat_penalty, seed=None, max_tokens=c.max_tokens)
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
        lib.cdll.sonder_sampling_config_init(ctypes.byref(c))
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
    """Owns one C handle; close() is idempotent and also runs at GC."""

    _destroy_fn = ""

    def __init__(self, lib: Library, ptr: int) -> None:
        self._lib = lib
        self._ptr: Optional[int] = ptr
        self._finalizer = weakref.finalize(self, getattr(lib.cdll, self._destroy_fn), ctypes.c_void_p(ptr))

    @property
    def closed(self) -> bool:
        return self._ptr is None

    def _handle(self) -> ctypes.c_void_p:
        if self._ptr is None:
            raise InvalidStateError(f"{type(self).__name__} is closed")
        return ctypes.c_void_p(self._ptr)

    def close(self) -> None:
        if self._ptr is not None:
            self._ptr = None
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._ptr is None else hex(self._ptr)
        return f"<{type(self).__name__} {state}>"


class Engine(_Handle):
    """An inference engine. Closing it closes its models and sessions first."""

    _destroy_fn = "sonder_engine_destroy"

    def __init__(self, telemetry_level: Optional[TelemetryLevel] = None,
                 telemetry_jsonl_path: Optional[str] = None, capture_text: bool = False,
                 library: Optional[Library] = None) -> None:
        lib = library or get_library()
        opts = S.EngineOptions()
        lib.cdll.sonder_engine_options_init(ctypes.byref(opts))
        if telemetry_level is not None:
            opts.telemetry_level = int(TelemetryLevel(telemetry_level))
        path_bytes = _cstr("telemetry_jsonl_path", telemetry_jsonl_path) if telemetry_jsonl_path else None
        opts.telemetry_jsonl_path = path_bytes
        opts.capture_text = 1 if capture_text else 0
        out = ctypes.c_void_p()
        _check(lib, lib.cdll.sonder_engine_create(ctypes.byref(opts), ctypes.byref(out)))
        super().__init__(lib, out.value)  # type: ignore[arg-type]
        self._children: "weakref.WeakSet[_Handle]" = weakref.WeakSet()

    @property
    def library(self) -> Library:
        return self._lib

    @property
    def device_count(self) -> int:
        return int(self._lib.cdll.sonder_engine_device_count(self._handle()))

    def register_mock_backend(self) -> None:
        """Register the deterministic MOCK backend ("mock"). Tests only."""
        _check(self._lib, self._lib.cdll.sonder_engine_register_mock_backend(self._handle()))

    def register_ollama_backend(self, base_url: Optional[str] = None) -> None:
        url = _cstr("base_url", base_url) if base_url else None
        _check(self._lib, self._lib.cdll.sonder_engine_register_ollama_backend(self._handle(), url))

    def load_model(self, backend: str, model: str) -> "Model":
        backend_b, model_b = _cstr("backend", backend), _cstr("model", model)
        out = ctypes.c_void_p()
        _check(self._lib, self._lib.cdll.sonder_model_load(
            self._handle(), backend_b, model_b, ctypes.byref(out)))
        m = Model(self, out.value)  # type: ignore[arg-type]
        self._children.add(m)
        return m

    def create_session(self, model: "Model", sampling: Optional[SamplingConfig] = None) -> "Session":
        if model.engine is not self:
            raise InvalidArgumentError("model belongs to a different engine")
        marshalled = sampling._to_c(self._lib) if sampling is not None else None
        out = ctypes.c_void_p()
        _check(self._lib, self._lib.cdll.sonder_session_create(
            self._handle(), model._handle(),
            ctypes.byref(marshalled.struct) if marshalled else None, ctypes.byref(out)))
        s = Session(self, model, out.value, sampling)  # type: ignore[arg-type]
        self._children.add(s)
        return s

    def close(self) -> None:
        if self._ptr is None:
            return
        # Sessions before models before the engine.
        children = list(self._children)
        for child in children:
            if isinstance(child, Session):
                child.close()
        for child in children:
            child.close()
        super().close()


class Model(_Handle):
    """A loaded model. Sessions keep the underlying model alive, so closing
    this handle while sessions exist is safe."""

    _destroy_fn = "sonder_model_release"

    def __init__(self, engine: Engine, ptr: int) -> None:
        super().__init__(engine._lib, ptr)
        self.engine = engine


class Session(_Handle):
    """A generation session. One generate call at a time."""

    _destroy_fn = "sonder_session_destroy"

    def __init__(self, engine: Engine, model: Model, ptr: int, sampling: Optional[SamplingConfig]) -> None:
        super().__init__(engine._lib, ptr)
        self.engine = engine  # keeps the engine alive while the session is
        self.sampling = sampling

    def generate(self, prompt: str, on_token: Optional[TokenCallback] = None) -> GenerationResult:
        """Run a request to completion.

        ``on_token(text)`` is called for each streamed chunk (UTF-8 decoded
        incrementally, so multi-byte characters are never split). Return
        ``False`` from it to stop early; an exception raised in it stops the
        generation and is re-raised here.
        """
        lib = self._lib
        prompt_b = _cstr("prompt", prompt)
        handle = self._handle()
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
            try:
                text = decoder.decode(ctypes.string_at(ptr, length)) if length and ptr else ""
                return deliver(text) if text else 0
            except BaseException as e:  # noqa: BLE001 - re-raised after the C call returns
                error.append(e)
                return 1

        c_callback = S.TOKEN_CALLBACK(trampoline)
        stats = S.CGenerationStats()
        stats.struct_size = ctypes.sizeof(S.CGenerationStats)
        rc = lib.cdll.sonder_session_generate(handle, prompt_b, c_callback, None,
                                              ctypes.byref(stats))
        if error:
            raise error[0]
        _check(lib, rc)
        tail = decoder.decode(b"", final=True)
        if tail and not stopped[0]:
            deliver(tail)
        return GenerationResult(
            text="".join(parts), outcome=Outcome(stats.outcome), prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens, chunks=stats.chunks,
            ttft_ms=None if stats.ttft_ms < 0 else stats.ttft_ms, total_ms=stats.total_ms)

    def stream(self, prompt: str) -> "TokenStream":
        """Iterate over chunks as they are produced (generation runs on a
        worker thread). Closing the stream early cancels the request."""
        _cstr("prompt", prompt)  # fail here, not later on the worker thread
        return TokenStream(self, prompt)

    def cancel(self) -> None:
        """Cancel the in-flight request (thread-safe; no-op when idle)."""
        _check(self._lib, self._lib.cdll.sonder_session_cancel(self._handle()))

    def chat(self, messages: Iterable[Union[ChatMessage, Mapping[str, str]]],
             on_token: Optional[TokenCallback] = None) -> GenerationResult:
        """Chat entry point: STUB until the C ABI exports one.

        The engine has Backend::chat, but include/sonder_inference.h has no chat
        function yet, so this always raises :class:`UnsupportedError`. See
        bindings/python/README.md for the proposed signature.
        """
        del messages, on_token
        raise UnsupportedError(
            "chat is not exported by the Sonder C ABI yet (no sonder_session_chat); "
            "use generate() with a formatted prompt")


class TokenStream:
    """Iterator over generated chunks; see :meth:`Session.stream`."""

    _DONE = object()

    def __init__(self, session: Session, prompt: str) -> None:
        self._session = session
        self._prompt = prompt
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._closed = threading.Event()
        self._exc: Optional[BaseException] = None
        self._finished = False
        self.result: Optional[GenerationResult] = None
        self._thread = threading.Thread(target=self._run, name="sonder-stream", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.result = self._session.generate(self._prompt, on_token=self._push)
        except BaseException as e:  # noqa: BLE001 - surfaced to the consumer
            self._exc = e
        finally:
            self._queue.put(self._DONE)

    def _push(self, text: str) -> Optional[bool]:
        if self._closed.is_set():
            return False
        self._queue.put(text)
        return None

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        if self._finished:
            raise StopIteration
        item = self._queue.get()
        if item is self._DONE:
            self._finished = True
            self._thread.join()
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return item

    def close(self) -> None:
        """Stop early: cancels the request and waits for the worker."""
        if not self._finished:
            self._closed.set()
            if self._thread.is_alive():
                try:
                    self._session.cancel()
                except SonderError:
                    pass
            self._thread.join()
            self._finished = True

    def __enter__(self) -> "TokenStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = [
    "ChatMessage", "Engine", "GenerationResult", "Model", "Outcome", "SamplingConfig", "Session",
    "TelemetryLevel", "TokenStream", "abi_version", "version",
]
