"""Incremental Server-Sent Events parser (WHATWG event-stream subset).

Feed it raw bytes as they arrive; it returns complete events. Handles LF, CR
and CRLF line endings, multi-line ``data:`` fields, comments, the optional
single space after the colon, and input split at arbitrary byte boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SSEEvent:
    data: str
    event: str = "message"
    id: str | None = None

    @property
    def is_done(self) -> bool:
        """OpenAI-style terminator ``data: [DONE]``."""
        return self.data.strip() == "[DONE]"


class SSEParser:
    def __init__(self) -> None:
        self._buf = b""
        self._data: list[str] = []
        self._event = ""
        self._id: str | None = None
        self._pending_cr = False

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        """Consume bytes, return the events completed by them."""
        if self._pending_cr and chunk.startswith(b"\n"):
            chunk = chunk[1:]  # second half of a CRLF split across feeds
        self._pending_cr = False
        self._buf += chunk
        events: list[SSEEvent] = []
        while True:
            idx_n = self._buf.find(b"\n")
            idx_r = self._buf.find(b"\r")
            if idx_n < 0 and idx_r < 0:
                break
            if idx_r >= 0 and (idx_n < 0 or idx_r < idx_n):
                line = self._buf[:idx_r]
                rest = self._buf[idx_r + 1:]
                if rest.startswith(b"\n"):
                    rest = rest[1:]
                elif not rest:
                    self._pending_cr = True
            else:
                line = self._buf[:idx_n]
                rest = self._buf[idx_n + 1:]
            self._buf = rest
            ev = self._line(line.decode("utf-8", errors="replace"))
            if ev is not None:
                events.append(ev)
        return events

    def flush(self) -> list[SSEEvent]:
        """End of stream: a trailing line without newline and any pending event."""
        events: list[SSEEvent] = []
        if self._buf:
            ev = self._line(self._buf.decode("utf-8", errors="replace"))
            self._buf = b""
            if ev is not None:
                events.append(ev)
        ev = self._line("")
        if ev is not None:
            events.append(ev)
        return events

    def _line(self, line: str) -> SSEEvent | None:
        if line == "":
            if not self._data:
                self._event = ""
                return None
            ev = SSEEvent(data="\n".join(self._data), event=self._event or "message", id=self._id)
            self._data = []
            self._event = ""
            return ev
        if line.startswith(":"):
            return None  # comment / keep-alive
        field, sep, value = line.partition(":")
        if not sep:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value
        elif field == "id":
            self._id = value
        return None
