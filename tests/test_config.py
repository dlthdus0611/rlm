from rlm.config import Settings, get_settings


def test_defaults_when_env_unset(monkeypatch):
    for var in ("OPENROUTER_BASE_URL", "RLM_ROOT_MODEL", "RLM_SUB_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = get_settings()
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert s.rlm_root_model == "openai/gpt-5.6-sol"
    assert s.rlm_sub_model == "openai/gpt-5.6-luna"


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("RLM_ROOT_MODEL", "anthropic/claude-3")
    monkeypatch.setenv("RLM_SUB_MODEL", "openai/gpt-4o")
    s = get_settings()
    assert s.openrouter_base_url == "https://example.test/v1"
    assert s.rlm_root_model == "anthropic/claude-3"
    assert s.rlm_sub_model == "openai/gpt-4o"


def test_get_settings_reads_env_at_call_time(monkeypatch):
    # 호출 시점에 환경변수를 읽으므로, 같은 프로세스에서 값이 바뀌면 반영된다.
    monkeypatch.setenv("RLM_ROOT_MODEL", "first")
    assert get_settings().rlm_root_model == "first"
    monkeypatch.setenv("RLM_ROOT_MODEL", "second")
    assert get_settings().rlm_root_model == "second"


def test_make_llm_uses_env_base_url(monkeypatch):
    # base_url 은 호출 시점에 Settings 를 거쳐 읽혀야 한다(import 시점 고정이 아니라).
    from rlm.llm import make_llm

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/v1")
    llm = make_llm("openai/gpt-4o-mini")
    assert str(llm.openai_api_base) == "https://example.test/v1"
