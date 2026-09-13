from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import crew.config.llm as llm_config


def test_resolve_study_assistant_model_uses_env_default_and_normalizes_it(monkeypatch):
    monkeypatch.setenv("STUDY_ASSISTANT_MODEL", "openai/devstral-2-123b-instruct-2512")

    assert llm_config.resolve_study_assistant_model() == "devstral-2-123b-instruct-2512"


def test_resolve_manager_model_uses_env_default_before_specialist(monkeypatch):
    monkeypatch.setenv("STUDY_ASSISTANT_MANAGER_MODEL", "openai/mistral-medium-3.5-128b")

    assert (
        llm_config.resolve_study_assistant_manager_model(specialist_model="deepseek-v4-flash")
        == "mistral-medium-3.5-128b"
    )


def test_resolve_manager_model_allows_explicit_override(monkeypatch):
    monkeypatch.setenv("STUDY_ASSISTANT_MANAGER_MODEL", "mistral-medium-3.5-128b")

    assert (
        llm_config.resolve_study_assistant_manager_model(
            manager_model="openai/qwen3.5-122b-a10b",
            specialist_model="deepseek-v4-flash",
        )
        == "qwen3.5-122b-a10b"
    )


def test_resolve_observer_model_prefers_lightweight_override(monkeypatch):
    monkeypatch.setenv("STUDY_ASSISTANT_OBSERVER_MODEL", "openai/meta-llama-3.1-8b-instruct")

    assert (
        llm_config.resolve_study_assistant_observer_model(
            manager_model="qwen3.5-122b-a10b",
            specialist_model="devstral",
        )
        == "meta-llama-3.1-8b-instruct"
    )


def test_resolve_observer_model_uses_dedicated_lightweight_default(monkeypatch):
    monkeypatch.delenv("STUDY_ASSISTANT_OBSERVER_MODEL", raising=False)
    monkeypatch.setattr(llm_config, "load_dotenv", lambda: None)

    assert (
        llm_config.resolve_study_assistant_observer_model(
            manager_model="openai/qwen3.6-35b-a3b",
            specialist_model="devstral",
        )
        == "qwen3-30b-a3b-instruct-2507"
    )


def test_resolve_observer_timeout_is_independently_configurable(monkeypatch):
    monkeypatch.setenv("STUDY_ASSISTANT_OBSERVER_TIMEOUT_SECONDS", "145")

    assert llm_config.resolve_study_assistant_observer_timeout() == 145


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


def test_resolve_llm_settings_uses_configurable_main_timeout(monkeypatch):
    monkeypatch.setenv("GWDG_API_KEY", "test-key")
    monkeypatch.setenv("STUDY_ASSISTANT_LLM_TIMEOUT_SECONDS", "360")

    settings = llm_config.resolve_llm_settings(model="devstral")

    assert settings.timeout == 360


def test_resolve_llm_settings_rejects_invalid_timeout(monkeypatch):
    monkeypatch.setenv("GWDG_API_KEY", "test-key")
    monkeypatch.setenv("STUDY_ASSISTANT_LLM_TIMEOUT_SECONDS", "never")

    with pytest.raises(llm_config.LLMConfigurationError, match="positive integer"):
        llm_config.resolve_llm_settings(model="devstral")


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


def test_llm_calls_to_same_endpoint_and_model_are_serialized(monkeypatch):
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    class FakeLLM:
        instances = 0

        def __init__(self, **kwargs):
            del kwargs
            self.index = FakeLLM.instances
            FakeLLM.instances += 1

        def call(self, prompt):
            del prompt
            if self.index == 0:
                first_entered.set()
                assert release_first.wait(timeout=1)
            else:
                second_entered.set()
            return self.index

    monkeypatch.setattr(llm_config, "LLM", FakeLLM)
    first = llm_config.get_default_llm(
        model="shared-model",
        api_key="test-key",
        base_url="https://gwdg.example.test/v1",
    )
    second = llm_config.get_default_llm(
        model="shared-model",
        api_key="test-key",
        base_url="https://gwdg.example.test/v1",
    )

    first_thread = threading.Thread(target=lambda: first.call("first"))
    second_thread = threading.Thread(target=lambda: second.call("second"))
    first_thread.start()
    assert first_entered.wait(timeout=1)
    second_thread.start()
    assert not second_entered.wait(timeout=0.1)

    release_first.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)
    assert second_entered.is_set()


@pytest.mark.parametrize(("model", "body"), [
    ("openai/qwen3.8-27b", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("qwen3.5-122b-a10b", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("gemma-4-31b-it", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("qwen3-30b-a3b-instruct-2507", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("glm-4.7", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("deepseek-v4-flash", {"chat_template_kwargs": {"enable_thinking": False, "thinking": False},
                           "thinking": {"type": "disabled"}}),
    ("mistral-medium-3.5-128b", {"reasoning_effort": "none"}),
    ("devstral-2-123b-instruct-2512", {"reasoning_effort": "none"}),
])
def test_observer_and_specialists_disable_thinking_with_backend_controls(monkeypatch, model, body):
    monkeypatch.setenv("GWDG_API_KEY", "test-key")
    monkeypatch.setenv("GWDG_API_BASE", "https://observer.example.test/v1")
    monkeypatch.setenv("STUDY_ASSISTANT_OBSERVER_TIMEOUT_SECONDS", "23")
    # Real CrewAI/OpenAI clients, without sending a request.
    main = llm_config.get_default_llm(model=model)
    observer = llm_config.get_observer_llm(model=model, max_tokens=1024)
    specialist = llm_config.get_default_llm(model=model, thinking=False)

    assert "extra_body" not in main.additional_params
    assert main._get_sync_client().max_retries == 2
    assert observer._get_sync_client().max_retries == 0
    assert observer.timeout == 23
    assert observer.max_tokens == 1024
    assert observer.additional_params["extra_body"] == body
    assert specialist.additional_params["extra_body"] == body


@pytest.mark.parametrize("model", ["openai-gpt-oss-120b", "deepseek-r1-distill-llama-70b"])
def test_observer_rejects_reasoning_only_models_before_creating_client(monkeypatch, model):
    monkeypatch.setenv("GWDG_API_KEY", "test-key")
    monkeypatch.setattr(llm_config, "LLM", lambda **kwargs: pytest.fail("Client created"))
    with pytest.raises(llm_config.LLMConfigurationError, match="no supported non-thinking mode"):
        llm_config.get_observer_llm(model=model)


def test_classifier_and_manager_thinking_budgets_are_independent(monkeypatch):
    monkeypatch.setenv("GWDG_API_KEY", "test-key")
    monkeypatch.setenv("GWDG_API_BASE", "https://thinking.example.test/v1")
    monkeypatch.setenv("STUDY_ASSISTANT_MANAGER_MODEL", "qwen3.8-27b")
    monkeypatch.setenv("STUDY_ASSISTANT_OBSERVER_MODEL", "qwen3-30b-a3b-instruct-2507")
    monkeypatch.setenv("STUDY_ASSISTANT_CLASSIFIER_MAX_TOKENS", "8192")
    monkeypatch.setenv("STUDY_ASSISTANT_THINKING_MAX_TOKENS", "32768")

    classifier = llm_config.get_classifier_llm()
    manager = llm_config.get_default_llm(model="qwen3.8-27b", thinking=True)
    observer = llm_config.get_observer_llm()

    assert classifier.model == manager.model == "qwen3.8-27b"
    assert classifier.max_tokens == 8192
    assert manager.max_tokens == 32768
    assert observer.max_tokens == 1024
    assert classifier.additional_params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert manager.additional_params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert observer.additional_params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("setting", ["STUDY_ASSISTANT_CLASSIFIER_MAX_TOKENS", "STUDY_ASSISTANT_THINKING_MAX_TOKENS"])
def test_thinking_token_budgets_must_be_positive(monkeypatch, setting):
    monkeypatch.setenv("GWDG_API_KEY", "test-key")
    monkeypatch.setenv(setting, "0")
    with pytest.raises(llm_config.LLMConfigurationError, match="positive integer"):
        if setting == "STUDY_ASSISTANT_CLASSIFIER_MAX_TOKENS":
            llm_config.get_classifier_llm(model="qwen3.8-27b")
        else:
            llm_config.get_default_llm(model="qwen3.8-27b", thinking=True)


def test_waiting_for_busy_model_times_out_without_releasing_its_lock(monkeypatch):
    lock = threading.Lock()
    lock.acquire()
    key = ("openai", "https://busy.example.test/v1", "busy-model")
    monkeypatch.setitem(llm_config._LLM_CALL_LOCKS, key, lock)
    llm = SimpleNamespace(call=lambda *args, **kwargs: pytest.fail("Busy model was called"))
    # A fractional timeout keeps this concurrency regression fast.
    settings = llm_config.LLMSettings(
        model=key[2], api_key="test-key", base_url=key[1], temperature=0, timeout=0.01,
    )
    llm_config._serialize_endpoint_calls(llm, settings)

    try:
        with pytest.raises(TimeoutError, match="waiting for another request"):
            llm.call("classify")
        assert lock.locked()
    finally:
        lock.release()


def test_explicit_rate_limit_retry_budget_releases_model_lane(monkeypatch):
    class RateLimited(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "1"})

    calls = []
    sleeps = []

    def limited_call(*args, **kwargs):
        calls.append(1)
        raise RateLimited()

    llm = SimpleNamespace(call=limited_call)
    settings = llm_config.LLMSettings(
        model="limited-model", api_key="test-key", base_url="https://gwdg.example.test/v1", temperature=0,
    )
    monkeypatch.setattr(llm_config.time, "sleep", sleeps.append)
    llm_config._serialize_endpoint_calls(llm, settings, max_retries=1)

    with pytest.raises(RateLimited):
        llm.call("classify")

    assert len(calls) == 2
    assert sleeps == [1]
    key = (settings.provider, settings.base_url, settings.model)
    assert not llm_config._LLM_CALL_LOCKS[key].locked()


def test_llm_retries_provider_retry_after_and_emits_wait_event(monkeypatch):
    sleeps = []
    events = []

    class FakeRateLimitError(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "75"})

    class FakeLLM:
        def __init__(self, **kwargs):
            del kwargs
            self.calls = 0

        def call(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            if self.calls == 1:
                raise FakeRateLimitError()
            return "recovered"

    monkeypatch.setattr(llm_config, "LLM", FakeLLM)
    monkeypatch.setattr(llm_config.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_config.time, "time", lambda: 1_000.0)
    llm = llm_config.get_default_llm(
        model="shared-model",
        api_key="test-key",
        base_url="https://gwdg.example.test/v1",
    )

    with llm_config.report_rate_limit_waits(events.append):
        assert llm.call("retry this") == "recovered"

    assert sleeps == [75]
    assert events[0]["event"] == "llm_rate_limit_wait"
    assert events[0]["retry_after_seconds"] == 75
    assert events[0]["retry_at_unix"] == 1_075.0
    assert events[1]["event"] == "llm_rate_limit_retry_started"


def test_llm_retries_emits_wait_event_on_worker_thread(monkeypatch):
    import threading

    sleeps = []
    events = []

    class FakeRateLimitError(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "5"})

    class FakeLLM:
        def __init__(self, **kwargs):
            del kwargs
            self.calls = 0

        def call(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            if self.calls == 1:
                raise FakeRateLimitError()
            return "recovered"

    monkeypatch.setattr(llm_config, "LLM", FakeLLM)
    monkeypatch.setattr(llm_config.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_config.time, "time", lambda: 1_000.0)

    # Instantiate LLM while report_rate_limit_waits context is active
    with llm_config.report_rate_limit_waits(events.append):
        llm = llm_config.get_default_llm(
            model="shared-model",
            api_key="test-key",
            base_url="https://gwdg.example.test/v1",
        )

    # Execute on a worker thread (simulating CrewAI thread dispatch)
    result = []
    def worker():
        try:
            res = llm.call("retry this")
            result.append(res)
        except Exception as e:
            result.append(e)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert result == ["recovered"]
    assert sleeps == [5]
    assert len(events) == 2
    assert events[0]["event"] == "llm_rate_limit_wait"
    assert events[0]["retry_after_seconds"] == 5
    assert events[0]["retry_at_unix"] == 1_005.0
    assert events[1]["event"] == "llm_rate_limit_retry_started"


def test_llm_retries_falls_back_when_headers_missing(monkeypatch):
    sleeps = []
    events = []

    class FakeRateLimitError(Exception):
        status_code = 429
        response = None  # No response / headers

    class FakeLLM:
        def __init__(self, **kwargs):
            del kwargs
            self.calls = 0

        def call(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            if self.calls == 1:
                raise FakeRateLimitError("Rate limit exceeded")
            return "recovered"

    monkeypatch.setattr(llm_config, "LLM", FakeLLM)
    monkeypatch.setattr(llm_config.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_config.time, "time", lambda: 1_000.0)

    llm = llm_config.get_default_llm(
        model="shared-model",
        api_key="test-key",
        base_url="https://gwdg.example.test/v1",
    )

    with llm_config.report_rate_limit_waits(events.append):
        assert llm.call("retry this") == "recovered"

    # Default fallback cooldown is 30 seconds
    assert sleeps == [30]
    assert len(events) == 2
    assert events[0]["event"] == "llm_rate_limit_wait"
    assert events[0]["retry_after_seconds"] == 30
    assert events[0]["retry_at_unix"] == 1_030.0


def test_llm_retries_honors_env_cooldown_override(monkeypatch):
    sleeps = []
    events = []

    class FakeRateLimitError(Exception):
        status_code = 429

    class FakeLLM:
        def __init__(self, **kwargs):
            del kwargs
            self.calls = 0

        def call(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            if self.calls == 1:
                raise FakeRateLimitError()
            return "recovered"

    monkeypatch.setattr(llm_config, "LLM", FakeLLM)
    monkeypatch.setattr(llm_config.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_config.time, "time", lambda: 1_000.0)
    monkeypatch.setenv("STUDY_ASSISTANT_RATE_LIMIT_COOLDOWN", "15")

    llm = llm_config.get_default_llm(
        model="shared-model",
        api_key="test-key",
        base_url="https://gwdg.example.test/v1",
    )

    with llm_config.report_rate_limit_waits(events.append):
        assert llm.call("retry this") == "recovered"

    assert sleeps == [15]
    assert events[0]["retry_after_seconds"] == 15


def test_llm_retries_detects_rate_limit_from_exception_string(monkeypatch):
    sleeps = []
    events = []

    class FakeStringError(Exception):
        # No status_code attribute at all
        pass

    class FakeLLM:
        def __init__(self, **kwargs):
            del kwargs
            self.calls = 0

        def call(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            if self.calls == 1:
                raise FakeStringError("Error: 429 rate limit exceeded")
            return "recovered"

    monkeypatch.setattr(llm_config, "LLM", FakeLLM)
    monkeypatch.setattr(llm_config.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm_config.time, "time", lambda: 1_000.0)

    llm = llm_config.get_default_llm(
        model="shared-model",
        api_key="test-key",
        base_url="https://gwdg.example.test/v1",
    )

    with llm_config.report_rate_limit_waits(events.append):
        assert llm.call("retry this") == "recovered"

    assert sleeps == [30]
    assert events[0]["retry_after_seconds"] == 30
