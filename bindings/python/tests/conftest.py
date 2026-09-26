"""Tests run against the deterministic MOCK backend of a real shared library.

Set SONDER_INFERENCE_LIBRARY (file) or SONDER_INFERENCE_LIB_DIR (directory).
Without either, the tests fail rather than skip, so CI cannot pass vacuously.
"""

from __future__ import annotations

import pytest

import sonder_inference as si


@pytest.fixture(scope="session")
def lib() -> si.Library:
    return si.load_library()


@pytest.fixture()
def engine(lib: si.Library):
    with si.Engine(telemetry_level=si.TelemetryLevel.OFF) as e:
        e.register_mock_backend()
        yield e


@pytest.fixture()
def model(engine: si.Engine):
    with engine.load_model("mock", "mock:tiny") as m:
        yield m
