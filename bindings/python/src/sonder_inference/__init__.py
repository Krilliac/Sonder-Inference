"""Python bindings for the Sonder Inference C ABI (ctypes, no compiler needed).

Point the bindings at the shared library with ``SONDER_INFERENCE_LIBRARY``
(file) or ``SONDER_INFERENCE_LIB_DIR`` (directory), or call
:func:`load_library` with a path before creating an :class:`Engine`.
"""

from ._lib import (
    ENV_LIB_DIR,
    ENV_LIBRARY,
    SUPPORTED_ABI_VERSION,
    Library,
    LibraryNotFoundError,
    get_library,
    load_library,
)
from .api import (
    ChatMessage,
    Engine,
    GenerationResult,
    GpuInfo,
    Model,
    Outcome,
    RequestMetadata,
    SamplingConfig,
    Session,
    SessionMetadata,
    TelemetryLevel,
    TokenStream,
    WorkloadClass,
    abi_version,
    list_gpu_devices,
    version,
)
from .errors import (
    BackendError,
    CancelledError,
    InternalError,
    InvalidArgumentError,
    InvalidStateError,
    NotFoundError,
    ProtocolError,
    SonderError,
    SonderIOError,
    SonderTimeoutError,
    Status,
    UnavailableError,
    UnsupportedError,
    error_for_status,
)

__version__ = "0.1.0"

__all__ = [
    "BackendError", "CancelledError", "ChatMessage", "ENV_LIBRARY", "ENV_LIB_DIR", "Engine",
    "GenerationResult", "GpuInfo", "InternalError", "InvalidArgumentError", "InvalidStateError", "Library",
    "LibraryNotFoundError", "Model", "NotFoundError", "Outcome", "ProtocolError", "RequestMetadata",
    "SUPPORTED_ABI_VERSION", "SamplingConfig", "Session", "SessionMetadata", "SonderError", "SonderIOError",
    "SonderTimeoutError", "Status", "TelemetryLevel", "TokenStream", "UnavailableError",
    "UnsupportedError", "WorkloadClass", "abi_version", "error_for_status", "get_library",
    "list_gpu_devices", "load_library", "version",
]
