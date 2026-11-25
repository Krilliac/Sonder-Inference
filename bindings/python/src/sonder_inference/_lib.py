"""Locating and loading the Sonder Inference shared library (ctypes).

Search order used by :func:`load_library` when no explicit path is given:

1. ``SONDER_INFERENCE_LIBRARY``: full path to the shared library file.
2. ``SONDER_INFERENCE_LIB_DIR``: a directory that contains it.
3. The package directory (for wheels that bundle the library).
4. The platform loader search path (``ctypes.util.find_library``).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional, Union

from . import _structs as S

#: C ABI major version these bindings were written against.
SUPPORTED_ABI_VERSION = 1

ENV_LIBRARY = "SONDER_INFERENCE_LIBRARY"
ENV_LIB_DIR = "SONDER_INFERENCE_LIB_DIR"

PathLike = Union[str, "os.PathLike[str]"]


def library_file_names() -> list[str]:
    """Platform-specific file names the library may have."""
    if sys.platform == "win32":
        return ["sonder_inference.dll", "libsonder_inference.dll"]
    if sys.platform == "darwin":
        return ["libsonder_inference.dylib"]
    return ["libsonder_inference.so"]


class LibraryNotFoundError(OSError):
    """The shared library could not be located or loaded."""


def _candidates(path: Optional[PathLike]) -> list[Path]:
    if path is not None:
        p = Path(path)
        return [p / n for n in library_file_names()] if p.is_dir() else [p]
    out: list[Path] = []
    env_file = os.environ.get(ENV_LIBRARY)
    if env_file:
        out.append(Path(env_file))
    env_dir = os.environ.get(ENV_LIB_DIR)
    if env_dir:
        out.extend(Path(env_dir) / n for n in library_file_names())
    here = Path(__file__).resolve().parent
    out.extend(here / n for n in library_file_names())
    return out


def _declare_abi_version(lib: ctypes.CDLL, path: str) -> int:
    try:
        abi_fn = getattr(lib, "sonder_abi_version")
    except AttributeError as e:
        raise LibraryNotFoundError(
            f"{path}: missing required C ABI symbol 'sonder_abi_version'") from e
    abi_fn.restype = ctypes.c_uint32
    abi_fn.argtypes = []
    return int(abi_fn())


def _declare(lib: ctypes.CDLL, path: str) -> None:
    c = ctypes
    status = c.c_int

    def fn(name: str, restype: Any, *argtypes: Any) -> None:
        try:
            f = getattr(lib, name)
        except AttributeError as e:
            raise LibraryNotFoundError(
                f"{path}: missing required C ABI symbol '{name}'") from e
        f.restype = restype
        f.argtypes = list(argtypes)

    fn("sonder_abi_version", c.c_uint32)
    fn("sonder_version_string", c.c_char_p)
    fn("sonder_status_string", c.c_char_p, status)
    fn("sonder_last_error_message", c.c_char_p)

    fn("sonder_engine_options_init", None, c.POINTER(S.EngineOptions))
    fn("sonder_engine_create", status, c.POINTER(S.EngineOptions), c.POINTER(c.c_void_p))
    fn("sonder_engine_destroy", None, c.c_void_p)
    fn("sonder_engine_device_count", c.c_size_t, c.c_void_p)
    fn("sonder_engine_register_mock_backend", status, c.c_void_p)
    fn("sonder_engine_register_ollama_backend", status, c.c_void_p, c.c_char_p)

    fn("sonder_model_load", status, c.c_void_p, c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p))
    fn("sonder_model_release", None, c.c_void_p)

    fn("sonder_sampling_config_init", None, c.POINTER(S.CSamplingConfig))
    if hasattr(lib, "sonder_sampling_config_init_sized"):
        fn("sonder_sampling_config_init_sized", None, c.POINTER(S.CSamplingConfig), c.c_size_t)
    fn("sonder_sampling_config_validate", status, c.POINTER(S.CSamplingConfig))

    fn("sonder_session_create", status, c.c_void_p, c.c_void_p, c.POINTER(S.CSamplingConfig),
       c.POINTER(c.c_void_p))
    if hasattr(lib, "sonder_session_create_with_metadata"):
        fn("sonder_session_create_with_metadata", status, c.c_void_p, c.c_void_p,
           c.POINTER(S.CSamplingConfig), c.POINTER(S.CSessionMetadata), c.POINTER(c.c_void_p))
    fn("sonder_session_destroy", None, c.c_void_p)
    fn("sonder_session_generate", status, c.c_void_p, c.c_char_p, S.TOKEN_CALLBACK, c.c_void_p,
       c.POINTER(S.CGenerationStats))
    # Additive ABI-v1 operations remain optional in older libraries.
    if hasattr(lib, "sonder_session_chat"):
        fn("sonder_session_chat", status, c.c_void_p, c.POINTER(c.POINTER(S.CChatMessage)),
           c.c_size_t, S.TOKEN_CALLBACK, c.c_void_p, c.POINTER(S.CGenerationStats))
    fn("sonder_session_cancel", status, c.c_void_p)
    if hasattr(lib, "sonder_session_generate_with_metadata"):
        fn("sonder_session_generate_with_metadata", status, c.c_void_p, c.c_char_p,
           c.POINTER(S.CRequestMetadata), S.TOKEN_CALLBACK, c.c_void_p, c.POINTER(S.CGenerationStats))
    if hasattr(lib, "sonder_session_chat_with_metadata"):
        fn("sonder_session_chat_with_metadata", status, c.c_void_p, c.POINTER(c.POINTER(S.CChatMessage)),
           c.c_size_t, c.POINTER(S.CRequestMetadata), S.TOKEN_CALLBACK, c.c_void_p,
           c.POINTER(S.CGenerationStats))
    if hasattr(lib, "sonder_gpu_device_count"):
        fn("sonder_gpu_device_count", c.c_size_t)
    if hasattr(lib, "sonder_gpu_device_info"):
        fn("sonder_gpu_device_info", status, c.c_size_t, c.POINTER(S.CGpuInfo))
    if hasattr(lib, "sonder_engine_set_telemetry_callback"):
        fn("sonder_engine_set_telemetry_callback", status, c.c_void_p, S.TELEMETRY_CALLBACK, c.c_void_p)


class Library:
    """A loaded, ABI-checked Sonder Inference shared library."""

    def __init__(self, cdll: ctypes.CDLL, path: str) -> None:
        self.cdll = cdll
        self.path = path
        # Verify ABI version before declaring symbols so an incompatible ABI
        # raises a clear version-mismatch LibraryNotFoundError rather than an
        # AttributeError on a renamed/removed symbol.
        self.abi_version = _declare_abi_version(cdll, path)
        if self.abi_version != SUPPORTED_ABI_VERSION:
            raise LibraryNotFoundError(
                f"{path}: C ABI version {self.abi_version} is not supported "
                f"(these bindings target version {SUPPORTED_ABI_VERSION})")
        _declare(cdll, path)
        self._extended_sampling: Optional[bool] = None
        self._probe_lock = threading.Lock()

    @property
    def version(self) -> str:
        return (self.cdll.sonder_version_string() or b"").decode("utf-8", "replace")

    def init_sampling(self, cfg: "S.CSamplingConfig") -> None:
        """Library defaults into ``cfg``. Uses the sized initializer when the
        library has it: the original ``sonder_sampling_config_init`` symbol
        only initialises the original layout (it cannot know the caller's
        allocation size), so the appended fields would stay zero."""
        if self.has_symbol("sonder_sampling_config_init_sized"):
            self.cdll.sonder_sampling_config_init_sized(ctypes.byref(cfg), ctypes.sizeof(cfg))
        else:
            self.cdll.sonder_sampling_config_init(ctypes.byref(cfg))

    def has_symbol(self, name: str) -> bool:
        try:
            getattr(self.cdll, name)
        except AttributeError:
            return False
        return True

    @property
    def supports_extended_sampling(self) -> bool:
        """Whether the library reads the appended sampling fields.

        Libraries built before typical_p/penalties/logit_bias/num_ctx were
        appended accept the larger struct but ignore the tail. Probe once: an
        invalid ``typical_p`` is rejected only by a library that reads it.
        """
        if self._extended_sampling is None:
            with self._probe_lock:
                if self._extended_sampling is None:
                    cfg = S.CSamplingConfig()
                    self.init_sampling(cfg)
                    cfg.struct_size = ctypes.sizeof(S.CSamplingConfig)
                    cfg.typical_p = 0.0  # invalid when read
                    rc = self.cdll.sonder_sampling_config_validate(ctypes.byref(cfg))
                    if rc != 0:
                        # Clear the thread-local error written by the intentional probe failure.
                        clean = S.CSamplingConfig()
                        self.init_sampling(clean)
                        self.cdll.sonder_sampling_config_validate(ctypes.byref(clean))
                    self._extended_sampling = rc != 0
        return self._extended_sampling

    def status_string(self, status: int) -> str:
        return (self.cdll.sonder_status_string(int(status)) or b"").decode("utf-8", "replace")

    def last_error(self) -> str:
        return (self.cdll.sonder_last_error_message() or b"").decode("utf-8", "replace")


_lock = threading.Lock()
_library: Optional[Library] = None
_dll_dir_handles: list[Any] = []


def _same_path(loaded: str, candidate: Path) -> bool:
    try:
        return os.path.samefile(loaded, candidate)
    except OSError:
        return os.path.normcase(str(Path(loaded).resolve())) == os.path.normcase(str(candidate.resolve()))


def load_library(path: Optional[PathLike] = None) -> Library:
    """Load (or return the already loaded) library.

    ``path`` may be a file or a directory. Without it the environment
    variables and default locations documented in this module are tried.
    Loading a different path after the first successful load raises.
    """
    global _library
    with _lock:
        if _library is not None:
            if path is not None:
                if not any(_same_path(_library.path, c) for c in _candidates(path) if c.is_file()):
                    raise LibraryNotFoundError(
                        f"library already loaded from {_library.path}; cannot switch to {path}")
            return _library
        tried: list[str] = []
        for cand in _candidates(path):
            tried.append(str(cand))
            if cand.is_file():
                _library = Library(_open(cand), str(cand.resolve()))
                return _library
        if path is None:
            found = ctypes.util.find_library("sonder_inference")
            if found:
                tried.append(found)
                try:
                    cdll = ctypes.CDLL(found)
                except OSError:
                    cdll = None
                if cdll is not None:
                    # Do not swallow LibraryNotFoundError (e.g. ABI mismatch).
                    _library = Library(cdll, found)
                    return _library
        raise LibraryNotFoundError(
            "could not find the Sonder Inference shared library. Set "
            f"{ENV_LIBRARY} to the file or {ENV_LIB_DIR} to its directory. Tried: "
            + ", ".join(tried))


def _open(path: Path) -> ctypes.CDLL:
    resolved = path.resolve()
    dll_dir: Any = None
    if sys.platform == "win32":
        # Let the DLL's own directory satisfy its dependencies.
        dll_dir = os.add_dll_directory(str(resolved.parent))
    try:
        cdll = ctypes.CDLL(str(resolved))
    except OSError as e:
        if dll_dir is not None:
            dll_dir.close()
        raise LibraryNotFoundError(f"failed to load {path}: {e}") from e
    if dll_dir is not None:
        _dll_dir_handles.append(dll_dir)
    return cdll


def get_library() -> Library:
    """The loaded library, loading it with the default search if needed."""
    return _library if _library is not None else load_library()
