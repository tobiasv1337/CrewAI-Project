from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
import threading
import time
from math import ceil
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterator

from crewai import LLM
from dotenv import load_dotenv


DEFAULT_GWDG_API_BASE = "https://chat-ai.hpc.gwdg.de/v1"
DEFAULT_STUDY_ASSISTANT_MODEL = "qwen3.5-122b-a10b"
DEFAULT_STUDY_ASSISTANT_OBSERVER_MODEL = "qwen3-30b-a3b-instruct-2507"
DEFAULT_TEMPERATURE = 0.2
# A failed or starved upstream request must become visible to the crew within a
# useful interaction window.  Individual deployments may still override this
# through STUDY_ASSISTANT_LLM_TIMEOUT_SECONDS.
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_OBSERVER_TIMEOUT_SECONDS = 120

# GWDG's OpenAI-compatible endpoint can leave one of two simultaneous
# tool-calling requests waiting without a response.  CrewAI delegates sibling
# specialists concurrently, so use one in-process lane per endpoint/model.
# This is deliberately keyed below the agent layer: every agent using the same
# deployed model receives the same protection, while a distinct observer model
# remains independent.
_LLM_CALL_LOCKS: dict[tuple[str, str, str], threading.Lock] = {}
_LLM_CALL_LOCKS_GUARD = threading.Lock()
_RATE_LIMIT_WAIT_NOTIFIER: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "rate_limit_wait_notifier",
    default=None,
)


class LLMConfigurationError(RuntimeError):
    """Raised when the study assistant cannot build an LLM configuration."""


@contextmanager
def report_rate_limit_waits(
    notifier: Callable[[dict[str, Any]], None] | None,
) -> Iterator[None]:
    """Publish bounded LLM rate-limit waits to the active runtime trace."""
    token = _RATE_LIMIT_WAIT_NOTIFIER.set(notifier)
    try:
        yield
    finally:
        _RATE_LIMIT_WAIT_NOTIFIER.reset(token)


@dataclass(frozen=True)
class LLMSettings:
    model: str
    api_key: str
    base_url: str
    temperature: float
    top_p: float | None = None
    provider: str = "openai"
    timeout: int = DEFAULT_TIMEOUT_SECONDS


def normalize_openai_model_name(model: str) -> str:
    """Accept handoff-style openai/<model> values while sending raw GWDG IDs."""
    value = str(model or "").strip()
    if value.startswith("openai/"):
        return value.split("/", 1)[1]
    return value


def resolve_study_assistant_model(model: str | None = None) -> str:
    """Resolve an optional runtime override against the configured .env default."""
    load_dotenv()
    return normalize_openai_model_name(
        model or os.getenv("STUDY_ASSISTANT_MODEL", DEFAULT_STUDY_ASSISTANT_MODEL)
    )


def resolve_study_assistant_manager_model(
    *,
    manager_model: str | None = None,
    specialist_model: str | None = None,
) -> str:
    """Resolve the manager override, then its .env default, then the specialist model."""
    load_dotenv()
    return normalize_openai_model_name(
        manager_model
        or os.getenv("STUDY_ASSISTANT_MANAGER_MODEL")
        or specialist_model
        or resolve_study_assistant_model()
    )


def resolve_study_assistant_observer_model(
    *,
    observer_model: str | None = None,
    manager_model: str | None = None,
    specialist_model: str | None = None,
) -> str:
    """Resolve the dedicated lightweight routing and runtime-observer model.

    Manager and specialist arguments remain accepted for runtime-config compatibility;
    the observer intentionally has an independent, inexpensive default.
    """
    del manager_model, specialist_model
    load_dotenv()
    return normalize_openai_model_name(
        observer_model
        or os.getenv("STUDY_ASSISTANT_OBSERVER_MODEL")
        or DEFAULT_STUDY_ASSISTANT_OBSERVER_MODEL
    )


def resolve_study_assistant_observer_timeout() -> int:
    """Resolve the observer timeout independently from long-running crew calls."""
    load_dotenv()
    return _positive_timeout(
        os.getenv("STUDY_ASSISTANT_OBSERVER_TIMEOUT_SECONDS"),
        default=DEFAULT_OBSERVER_TIMEOUT_SECONDS,
        setting="STUDY_ASSISTANT_OBSERVER_TIMEOUT_SECONDS",
    )


def _positive_timeout(value: str | int | None, *, default: int, setting: str) -> int:
    try:
        timeout = default if value in (None, "") else int(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError(f"{setting} must be a positive integer.") from exc
    if timeout <= 0:
        raise LLMConfigurationError(f"{setting} must be a positive integer.")
    return timeout


def _retry_after_seconds(error: Exception) -> int | None:
    """Extract a provider-supplied Retry-After value from an OpenAI client error."""
    response = getattr(error, "response", None)
    status_code = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    if status_code != 429:
        return None
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        seconds = ceil(float(value))
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _notify_rate_limit_wait(event: dict[str, Any], notifier: Callable[[dict[str, Any]], None] | None = None) -> None:
    if notifier is None:
        notifier = _RATE_LIMIT_WAIT_NOTIFIER.get()
    if notifier is None:
        return
    try:
        notifier(event)
    except Exception:
        # User-facing trace callbacks must never alter a model request.
        return


def resolve_llm_settings(
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str | None = None,
    timeout: int | None = None,
) -> LLMSettings:
    """Resolve explicit overrides plus .env values into a validated LLM config."""
    load_dotenv()
    resolved_model = resolve_study_assistant_model(model)
    resolved_key = api_key if api_key is not None else os.getenv("GWDG_API_KEY", "")
    resolved_base_url = base_url or os.getenv("GWDG_API_BASE", DEFAULT_GWDG_API_BASE)
    resolved_provider = provider or os.getenv("STUDY_ASSISTANT_PROVIDER", "openai")
    resolved_temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    resolved_timeout = _positive_timeout(
        timeout if timeout is not None else os.getenv("STUDY_ASSISTANT_LLM_TIMEOUT_SECONDS"),
        default=DEFAULT_TIMEOUT_SECONDS,
        setting="STUDY_ASSISTANT_LLM_TIMEOUT_SECONDS",
    )

    if not resolved_model:
        raise LLMConfigurationError("STUDY_ASSISTANT_MODEL is empty.")
    if not resolved_key:
        raise LLMConfigurationError(
            "GWDG_API_KEY is not configured. Add it to .env or pass api_key explicitly."
        )

    return LLMSettings(
        model=resolved_model,
        api_key=resolved_key,
        base_url=resolved_base_url,
        temperature=resolved_temperature,
        top_p=top_p,
        provider=resolved_provider,
        timeout=resolved_timeout,
    )


def _serialize_endpoint_calls(llm: Any, settings: LLMSettings) -> Any:
    """Serialize synchronous calls sharing one deployed upstream model.

    The hierarchical manager can launch multiple specialist calls at once.
    On the configured GWDG endpoint this has repeatedly starved one request
    (it emitted ``llm_started`` but never reached a tool).  Applying the lock
    to each newly-created LLM instance lets CrewAI keep its agent concurrency
    while preventing simultaneous requests to the same model deployment.
    """
    original_call = getattr(llm, "call", None)
    if not callable(original_call):
        return llm

    key = (settings.provider, settings.base_url.rstrip("/"), settings.model)
    with _LLM_CALL_LOCKS_GUARD:
        call_lock = _LLM_CALL_LOCKS.setdefault(key, threading.Lock())

    captured_notifier = _RATE_LIMIT_WAIT_NOTIFIER.get()

    @wraps(original_call)
    def serialized_call(*args: Any, **kwargs: Any) -> Any:
        with call_lock:
            attempt = 0
            while True:
                try:
                    return original_call(*args, **kwargs)
                except Exception as error:
                    retry_after = _retry_after_seconds(error)
                    if retry_after is None:
                        raise
                    attempt += 1
                    from crew.tracing import agent_label_for_role
                    agent = kwargs.get("from_agent")
                    agent_role = getattr(agent, "role", None)
                    agent_label = agent_label_for_role(agent_role) if agent_role else "Orchestrator"
                    active_notifier = _RATE_LIMIT_WAIT_NOTIFIER.get() or captured_notifier
                    _notify_rate_limit_wait(
                        {
                            "event": "llm_rate_limit_wait",
                            "agent_role": agent_role,
                            "agent_label": agent_label,
                            "model": settings.model,
                            "phase": "rate_limit",
                            "status": "waiting",
                            "retry_after_seconds": retry_after,
                            "retry_at_unix": time.time() + retry_after,
                            "attempt": attempt,
                            "activity": (
                                f"API rate limit reached; waiting {retry_after} seconds before retry {attempt}."
                            ),
                        },
                        notifier=active_notifier,
                    )
                    time.sleep(retry_after)
                    _notify_rate_limit_wait(
                        {
                            "event": "llm_rate_limit_retry_started",
                            "agent_role": agent_role,
                            "agent_label": agent_label,
                            "model": settings.model,
                            "phase": "rate_limit",
                            "status": "running",
                            "attempt": attempt,
                            "activity": "API rate-limit wait ended; retrying the LLM request.",
                        },
                        notifier=active_notifier,
                    )

    try:
        setattr(llm, "call", serialized_call)
    except (AttributeError, TypeError):
        # Keep compatibility with custom LLM implementations that disallow
        # method replacement; their native concurrency behaviour is retained.
        pass
    return llm


def get_default_llm(
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str | None = None,
    timeout: int | None = None,
) -> LLM:
    """Build a CrewAI LLM for GWDG's OpenAI-compatible endpoint."""
    settings = resolve_llm_settings(
        model=model,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        timeout=timeout,
    )
    kwargs = {
        "model": settings.model,
        "provider": settings.provider,
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "temperature": settings.temperature,
        "timeout": settings.timeout,
    }
    if settings.top_p is not None:
        kwargs["top_p"] = settings.top_p
    return _serialize_endpoint_calls(LLM(**kwargs), settings)
