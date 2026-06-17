from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator


_CONFIRMED_WRITE_SCOPE = ContextVar("confirmed_write_scope", default=False)


def confirmed_writes_enabled() -> bool:
    """Return whether irreversible write tools may run in this execution scope."""
    return _CONFIRMED_WRITE_SCOPE.get()


@contextmanager
def allow_confirmed_writes() -> Iterator[None]:
    """Temporarily allow irreversible writes after the UI approval gate passed."""
    token = _CONFIRMED_WRITE_SCOPE.set(True)
    try:
        yield
    finally:
        _CONFIRMED_WRITE_SCOPE.reset(token)
