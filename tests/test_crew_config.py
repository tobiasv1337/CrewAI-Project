from __future__ import annotations

import pytest

import crew.config.llm as llm_config


def test_resolve_llm_settings_uses_env_and_strips_openai_prefix(monkeypatch):
    monkeypatch.setenv("GWDG_API_KEY", "test-key")
    monkeypatch.setenv("GWDG_API_BASE", "https://gwdg.example.test/v1")
    monkeypatch.setenv("STUDY_ASSISTANT_MODEL", "openai/devstral-2-123b-instruct-2512")

    settings = llm_config.resolve_llm_settings(temperature=0.3, top_p=0.8)

    assert settings.model == "devstral-2-123b-instruct-2512"
    assert settings.api_key == "test-key"
    assert settings.base_url == "https://gwdg.example.test/v1"
    assert settings.temperature == 0.3
    assert settings.top_p == 0.8
    assert settings.provider == "openai"


def test_resolve_llm_settings_requires_api_key():
    with pytest.raises(llm_config.LLMConfigurationError, match="GWDG_API_KEY"):
        llm_config.resolve_llm_settings(api_key="", model="devstral")


def test_get_default_llm_builds_openai_compatible_llm(monkeypatch):
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(llm_config, "LLM", FakeLLM)

    llm = llm_config.get_default_llm(
        model="openai/devstral-2-123b-instruct-2512",
        api_key="test-key",
        base_url="https://gwdg.example.test/v1",
        temperature=0.1,
        top_p=0.9,
    )

    assert isinstance(llm, FakeLLM)
    assert calls == [
        {
            "model": "devstral-2-123b-instruct-2512",
            "provider": "openai",
            "api_key": "test-key",
            "base_url": "https://gwdg.example.test/v1",
            "temperature": 0.1,
            "timeout": llm_config.DEFAULT_TIMEOUT_SECONDS,
            "top_p": 0.9,
        }
    ]
