from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


_ACTIVE_PROFILE_SLUG: ContextVar[str | None] = ContextVar("active_grade_manager_profile_slug", default=None)
_ACTIVE_THREAD_ID: ContextVar[str | None] = ContextVar("active_chat_thread_id", default=None)


def get_active_profile_slug() -> str | None:
    """Return the profile slug selected for the current crew run, if any."""
    return _ACTIVE_PROFILE_SLUG.get()


def get_active_thread_id() -> str | None:
    """Return the chat thread ID selected for the current crew run, if any."""
    return _ACTIVE_THREAD_ID.get()


@contextmanager
def use_grade_manager_profile(profile_slug: str | None) -> Iterator[None]:
    """Scope Grade Manager tools to a Streamlit-selected profile.

    CLI and test runs can omit this context and will keep the existing primary
    profile fallback behavior.
    """
    cleaned = str(profile_slug or "").strip() or None
    token = _ACTIVE_PROFILE_SLUG.set(cleaned)
    try:
        yield
    finally:
        _ACTIVE_PROFILE_SLUG.reset(token)


@contextmanager
def use_chat_thread_id(thread_id: str | None) -> Iterator[None]:
    """Scope chat persistence to a specific thread ID during a crew run."""
    cleaned = str(thread_id or "").strip() or None
    token = _ACTIVE_THREAD_ID.set(cleaned)
    try:
        yield
    finally:
        _ACTIVE_THREAD_ID.reset(token)

