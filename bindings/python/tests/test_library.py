import ctypes

from pathlib import Path

import pytest

import sonder_inference as si
from sonder_inference import _structs as S


def test_abi_and_version(lib):
    assert si.abi_version() == si.SUPPORTED_ABI_VERSION == 1
    assert si.version()
    assert Path(lib.path).is_file()


def test_load_library_is_idempotent_and_accepts_same_path(lib):
    assert si.load_library() is lib
    assert si.load_library(lib.path) is lib
    assert si.load_library(Path(lib.path).parent) is lib


def test_load_library_refuses_to_switch(lib, tmp_path):
    other = tmp_path / si._lib.library_file_names()[0]
    other.write_bytes(b"not a library")
    with pytest.raises(si.LibraryNotFoundError, match="already loaded"):
        si.load_library(other)


def test_env_var_candidates(monkeypatch, tmp_path):
    f = tmp_path / "custom.so"
    monkeypatch.setenv(si.ENV_LIBRARY, str(f))
    monkeypatch.setenv(si.ENV_LIB_DIR, str(tmp_path))
    cands = si._lib._candidates(None)
    assert cands[0] == f
    assert cands[1] == tmp_path / si._lib.library_file_names()[0]


def test_struct_layout_matches_header():
    # Offsets of the C header layout on the supported 64-bit platforms.
    assert S.CSamplingConfig.seed.offset == 32
    assert S.CSamplingConfig.max_tokens.offset == 40
    assert S.CSamplingConfig.typical_p.offset == 44
    assert S.CSamplingConfig.logit_bias.offset == 64
    assert S.SAMPLING_CONFIG_V1_SIZE == 48
    assert S.SAMPLING_CONFIG_SIZE == 80
    assert ctypes.sizeof(S.CLogitBias) == 8
    assert ctypes.sizeof(S.CGenerationStats) == 48


def test_library_supports_extended_sampling(lib):
    assert lib.supports_extended_sampling is True


def test_status_strings(lib):
    assert lib.cdll.sonder_status_string(int(si.Status.CANCELLED)) == b"cancelled"


def test_missing_library_message(monkeypatch, tmp_path):
    monkeypatch.setattr(si._lib, "_library", None)
    monkeypatch.setenv(si.ENV_LIBRARY, str(tmp_path / "nope.so"))
    monkeypatch.delenv(si.ENV_LIB_DIR, raising=False)
    monkeypatch.setattr(si._lib.ctypes.util, "find_library", lambda name: None)
    with pytest.raises(si.LibraryNotFoundError, match=si.ENV_LIBRARY):
        si.load_library()
    assert isinstance(si.LibraryNotFoundError("x"), OSError)
