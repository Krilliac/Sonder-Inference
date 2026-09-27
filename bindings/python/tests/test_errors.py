import pytest

import sonder_inference as si


@pytest.mark.parametrize(
    "status,cls,builtin",
    [
        (si.Status.INVALID_ARGUMENT, si.InvalidArgumentError, ValueError),
        (si.Status.INVALID_STATE, si.InvalidStateError, RuntimeError),
        (si.Status.NOT_FOUND, si.NotFoundError, LookupError),
        (si.Status.UNAVAILABLE, si.UnavailableError, ConnectionError),
        (si.Status.CANCELLED, si.CancelledError, si.SonderError),
        (si.Status.TIMEOUT, si.SonderTimeoutError, TimeoutError),
        (si.Status.BACKEND, si.BackendError, RuntimeError),
        (si.Status.PROTOCOL, si.ProtocolError, RuntimeError),
        (si.Status.IO, si.SonderIOError, OSError),
        (si.Status.UNSUPPORTED, si.UnsupportedError, NotImplementedError),
        (si.Status.INTERNAL, si.InternalError, RuntimeError),
    ],
)
def test_error_for_status(status, cls, builtin):
    e = si.error_for_status(int(status), "boom")
    assert type(e) is cls
    assert isinstance(e, si.SonderError) and isinstance(e, builtin)
    assert e.status == status and e.code == int(status) and str(e) == "boom"


def test_unknown_status_maps_to_base():
    e = si.error_for_status(99, "future code")
    assert type(e) is si.SonderError
    assert e.code == 99 and e.status is None


def test_status_enum_matches_library(lib):
    for s in si.Status:
        assert lib.cdll.sonder_status_string(int(s))


def test_real_errors_carry_library_message(engine):
    with pytest.raises(si.NotFoundError, match="mock"):
        engine.load_model("mock", "not-a-mock")
    with pytest.raises(si.InvalidStateError):
        engine.register_mock_backend()  # already registered
    with pytest.raises(si.NotFoundError):
        engine.load_model("no-such-backend", "x")
