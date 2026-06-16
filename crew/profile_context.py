from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


_ACTIVE_PROFILE_SLUG: ContextVar[str | None] = ContextVar("active_grade_manager_profile_slug", default=None)


def get_active_profile_slug() -> str | None:
    """Return the profile slug selected for the current crew run, if any."""
    return _ACTIVE_PROFILE_SLUG.get()


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
