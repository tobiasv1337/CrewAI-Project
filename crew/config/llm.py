from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import threading
import time
from math import ceil
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterator

from crewai import LLM
from dotenv import load_dotenv
from pydantic import BaseModel


DEFAULT_GWDG_API_BASE = "https://chat-ai.hpc.gwdg.de/v1"
DEFAULT_STUDY_ASSISTANT_MODEL = "qwen3.5-122b-a10b"
DEFAULT_STUDY_ASSISTANT_OBSERVER_MODEL = "qwen3-30b-a3b-instruct-2507"
DEFAULT_TEMPERATURE = 0.2
# A failed or starved upstream request must become visible to the crew within a
# useful interaction window.  Individual deployments may still override this
# through STUDY_ASSISTANT_LLM_TIMEOUT_SECONDS.
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_OBSERVER_TIMEOUT_SECONDS = 120
DEFAULT_CLASSIFIER_MAX_TOKENS = 4096
DEFAULT_THINKING_MAX_TOKENS = 16384

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


def structured_output_instructions(response_model: type[BaseModel]) -> str:
    """Expose the schema to the model as well as the provider's JSON decoder.

    Gemma can otherwise emit a field followed by repeating whitespace under
    strict decoding. Keep the existing provider schema and Pydantic validation.
    """
    schema = json.dumps(response_model.model_json_schema(), separators=(",", ":"))
    return (
        "\n\nReturn one complete compact JSON object on a single line. "
        "Do not use Markdown, indentation, blank lines, or trailing whitespace. "
        f"Follow this JSON schema: {schema}"
    )


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
    """Resolve the lightweight proposal-decision and runtime-observer model.

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
    """Extract a provider-supplied Retry-After value from an OpenAI client error, or return a default fallback."""
    response = getattr(error, "response", None)
    status_code = getattr(error, "status_code", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    is_rate_limit = False
    if status_code == 429:
        is_rate_limit = True
    else:
        err_str = str(error).lower()
        if "429" in err_str or "rate limit" in err_str:
            is_rate_limit = True

    if not is_rate_limit:
        return None

    # Check for retry-after header
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is not None:
        try:
            seconds = ceil(float(value))
            if seconds > 0:
                return seconds
        except (TypeError, ValueError):
            pass

    # No valid retry-after header found, but it is a rate limit error.
    # Fall back to a default cooldown (e.g. 30 seconds, or via env variable).
    load_dotenv()
    env_cooldown = os.getenv("STUDY_ASSISTANT_RATE_LIMIT_COOLDOWN")
    if env_cooldown:
        try:
            seconds = int(env_cooldown)
            if seconds > 0:
                return seconds
        except ValueError:
            pass
    return 30


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


def _serialize_endpoint_calls(
    llm: Any, settings: LLMSettings, *, max_retries: int | None = None,
) -> Any:
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
        if not call_lock.acquire(timeout=settings.timeout):
            raise TimeoutError(
                f"Timed out waiting for another request to model {settings.model}."
            )
        try:
            attempt = 0
            while True:
                try:
                    return original_call(*args, **kwargs)
                except Exception as error:
                    retry_after = _retry_after_seconds(error)
                    if retry_after is None or (max_retries is not None and attempt >= max_retries):
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
        finally:
            call_lock.release()

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
    max_tokens: int | None = None,
    max_retries: int | None = None,
    extra_body: dict[str, Any] | None = None,
    thinking: bool | None = None,
) -> LLM:
    """Build a GWDG LLM; explicit retry limits cover SDK and rate-limit retries."""
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
    if thinking is True and max_tokens is None:
        max_tokens = _positive_timeout(
            os.getenv("STUDY_ASSISTANT_THINKING_MAX_TOKENS"),
            default=DEFAULT_THINKING_MAX_TOKENS,
            setting="STUDY_ASSISTANT_THINKING_MAX_TOKENS",
        )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if thinking is not None:
        extra_body = {**(extra_body or {}), **_thinking_request_body(settings.model, thinking)}
    if extra_body:
        kwargs["extra_body"] = extra_body
    return _serialize_endpoint_calls(LLM(**kwargs), settings, max_retries=max_retries)


def get_classifier_llm(*, model: str | None = None, top_p: float | None = None) -> LLM:
    """Use the manager model with room for reasoning before classification JSON."""
    load_dotenv()
    return get_default_llm(
        model=resolve_study_assistant_manager_model(manager_model=model),
        temperature=0.0,
        top_p=top_p,
        timeout=_positive_timeout(
            os.getenv("STUDY_ASSISTANT_CLASSIFIER_TIMEOUT_SECONDS"),
            default=DEFAULT_TIMEOUT_SECONDS,
            setting="STUDY_ASSISTANT_CLASSIFIER_TIMEOUT_SECONDS",
        ),
        max_tokens=_positive_timeout(
            os.getenv("STUDY_ASSISTANT_CLASSIFIER_MAX_TOKENS"),
            default=DEFAULT_CLASSIFIER_MAX_TOKENS,
            setting="STUDY_ASSISTANT_CLASSIFIER_MAX_TOKENS",
        ),
        max_retries=0,
        thinking=True,
    )


def get_observer_llm(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    max_tokens: int = 1024,
) -> LLM:
    """Build a bounded non-thinking LLM for proposal decisions and status JSON.

    Thinking is always disabled for these short calls: reasoning shares their
    completion-token budget with the JSON. GWDG's model backends use different
    switches. Classification has a separate reasoning-enabled configuration.
    """
    resolved_model = resolve_study_assistant_observer_model(observer_model=model)
    return get_default_llm(
        model=resolved_model,
        temperature=temperature,
        top_p=top_p,
        timeout=resolve_study_assistant_observer_timeout(),
        max_tokens=max_tokens,
        max_retries=0,
        thinking=False,
    )


def _thinking_request_body(model: str, enabled: bool) -> dict[str, Any]:
    """Set thinking with the configured GWDG model backend's API controls."""
    name = normalize_openai_model_name(model).casefold()
    if "gpt-oss" in name or name.startswith("deepseek-r1"):
        if enabled:
            return {}  # These models already reason and cannot switch it off.
        raise LLMConfigurationError(
            f"Model {model} has no supported non-thinking mode. "
            "Choose an instruct model or a model that supports disabling thinking."
        )
    # Mistral tokenizers reject chat_template_kwargs entirely. Send this via
    # extra_body because CrewAI 1.15.4 otherwise omits reasoning_effort for
    # non-OpenAI model IDs.
    if name.startswith(("mistral", "devstral")):
        return {"reasoning_effort": "high" if enabled else "none"}
    body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": enabled}}
    if name.startswith("deepseek"):
        # Cover the API toggle and DeepSeek's native tokenizer parameter.
        body["thinking"] = {"type": "enabled" if enabled else "disabled"}
        body["chat_template_kwargs"]["thinking"] = enabled
    return body
