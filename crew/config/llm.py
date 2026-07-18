from __future__ import annotations

import os
from dataclasses import dataclass

from crewai import LLM
from dotenv import load_dotenv


DEFAULT_GWDG_API_BASE = "https://chat-ai.hpc.gwdg.de/v1"
DEFAULT_STUDY_ASSISTANT_MODEL = "qwen3.5-122b-a10b"
DEFAULT_STUDY_ASSISTANT_OBSERVER_MODEL = "qwen3-30b-a3b-instruct-2507"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_OBSERVER_TIMEOUT_SECONDS = 120


class LLMConfigurationError(RuntimeError):
    """Raised when the study assistant cannot build an LLM configuration."""


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
    return LLM(**kwargs)
