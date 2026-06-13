import pytest

from rlm.llm import make_llm


def test_make_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_llm("openai/gpt-4o-mini")
