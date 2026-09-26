"""Exceptions mapped from sonder_status codes."""

from __future__ import annotations

import enum
from typing import Optional


class Status(enum.IntEnum):
    """Mirror of ``sonder_status`` (append-only)."""

    OK = 0
    INVALID_ARGUMENT = 1
    INVALID_STATE = 2
    NOT_FOUND = 3
    UNAVAILABLE = 4
    CANCELLED = 5
    TIMEOUT = 6
    BACKEND = 7
    PROTOCOL = 8
    IO = 9
    UNSUPPORTED = 10
    INTERNAL = 11


class SonderError(Exception):
    """Base class for errors reported by the C ABI."""

    status: Optional[Status] = None

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        if status is not None:
            try:
                self.status = Status(status)
            except ValueError:
                self.status = None
        self.code = int(status) if status is not None else (int(self.status) if self.status else None)


class InvalidArgumentError(SonderError, ValueError):
    status = Status.INVALID_ARGUMENT


class InvalidStateError(SonderError, RuntimeError):
    status = Status.INVALID_STATE


class NotFoundError(SonderError, LookupError):
    status = Status.NOT_FOUND


class UnavailableError(SonderError, ConnectionError):
    status = Status.UNAVAILABLE


class CancelledError(SonderError):
    status = Status.CANCELLED


class SonderTimeoutError(SonderError, TimeoutError):
    status = Status.TIMEOUT


class BackendError(SonderError, RuntimeError):
    status = Status.BACKEND


class ProtocolError(SonderError, RuntimeError):
    status = Status.PROTOCOL


class SonderIOError(SonderError, OSError):
    status = Status.IO


class UnsupportedError(SonderError, NotImplementedError):
    status = Status.UNSUPPORTED


class InternalError(SonderError, RuntimeError):
    status = Status.INTERNAL


_BY_STATUS: dict[int, type[SonderError]] = {
    cls.status.value: cls  # type: ignore[union-attr]
    for cls in (InvalidArgumentError, InvalidStateError, NotFoundError, UnavailableError,
                CancelledError, SonderTimeoutError, BackendError, ProtocolError, SonderIOError,
                UnsupportedError, InternalError)
}


def error_for_status(status: int, message: str) -> SonderError:
    """Build the exception for a non-OK status (unknown codes -> SonderError)."""
    cls = _BY_STATUS.get(int(status), SonderError)
    return cls(message, status)
