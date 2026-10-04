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
    Model,
    Outcome,
    SamplingConfig,
    Session,
    SessionMetadata,
    TelemetryLevel,
    TokenStream,
    abi_version,
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
    "GenerationResult", "InternalError", "InvalidArgumentError", "InvalidStateError", "Library",
    "LibraryNotFoundError", "Model", "NotFoundError", "Outcome", "ProtocolError",
    "SUPPORTED_ABI_VERSION", "SamplingConfig", "Session", "SessionMetadata", "SonderError", "SonderIOError",
    "SonderTimeoutError", "Status", "TelemetryLevel", "TokenStream", "UnavailableError",
    "UnsupportedError", "abi_version", "error_for_status", "get_library", "load_library", "version",
]
