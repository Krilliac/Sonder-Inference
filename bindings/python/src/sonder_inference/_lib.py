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
from typing import Optional, Union

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


def _declare(lib: ctypes.CDLL) -> None:
    c = ctypes
    status = c.c_int

    def fn(name: str, restype, *argtypes) -> None:
        f = getattr(lib, name)
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
    fn("sonder_sampling_config_validate", status, c.POINTER(S.CSamplingConfig))

    fn("sonder_session_create", status, c.c_void_p, c.c_void_p, c.POINTER(S.CSamplingConfig),
       c.POINTER(c.c_void_p))
    fn("sonder_session_destroy", None, c.c_void_p)
    fn("sonder_session_generate", status, c.c_void_p, c.c_char_p, S.TOKEN_CALLBACK, c.c_void_p,
       c.POINTER(S.CGenerationStats))
    fn("sonder_session_cancel", status, c.c_void_p)


class Library:
    """A loaded, ABI-checked Sonder Inference shared library."""

    def __init__(self, cdll: ctypes.CDLL, path: str) -> None:
        self.cdll = cdll
        self.path = path
        _declare(cdll)
        self.abi_version = int(cdll.sonder_abi_version())
        if self.abi_version != SUPPORTED_ABI_VERSION:
            raise LibraryNotFoundError(
                f"{path}: C ABI version {self.abi_version} is not supported "
                f"(these bindings target version {SUPPORTED_ABI_VERSION})")
        self._extended_sampling: Optional[bool] = None

    @property
    def version(self) -> str:
        return (self.cdll.sonder_version_string() or b"").decode("utf-8", "replace")

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
            cfg = S.CSamplingConfig()
            self.cdll.sonder_sampling_config_init(ctypes.byref(cfg))
            cfg.struct_size = ctypes.sizeof(S.CSamplingConfig)
            cfg.typical_p = 0.0  # invalid when read
            rc = self.cdll.sonder_sampling_config_validate(ctypes.byref(cfg))
            self._extended_sampling = rc != 0
        return self._extended_sampling

    def last_error(self) -> str:
        return (self.cdll.sonder_last_error_message() or b"").decode("utf-8", "replace")


_lock = threading.Lock()
_library: Optional[Library] = None


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
                wanted = {str(c.resolve()) for c in _candidates(path) if c.is_file()}
                if _library.path not in wanted:
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
                    _library = Library(ctypes.CDLL(found), found)
                    return _library
                except OSError:
                    pass
        raise LibraryNotFoundError(
            "could not find the Sonder Inference shared library. Set "
            f"{ENV_LIBRARY} to the file or {ENV_LIB_DIR} to its directory. Tried: "
            + ", ".join(tried))


def _open(path: Path) -> ctypes.CDLL:
    if sys.platform == "win32":
        # Let the DLL's own directory satisfy its dependencies.
        os.add_dll_directory(str(path.resolve().parent))
    try:
        return ctypes.CDLL(str(path.resolve()))
    except OSError as e:
        raise LibraryNotFoundError(f"failed to load {path}: {e}") from e


def get_library() -> Library:
    """The loaded library, loading it with the default search if needed."""
    return _library if _library is not None else load_library()
