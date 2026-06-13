import os

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ROOT_MODEL = "openai/gpt-4o-mini"
DEFAULT_SUB_MODEL = "openai/gpt-4o-mini"


def make_llm(model: str) -> ChatOpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY 환경변수가 필요합니다.")
    return ChatOpenAI(model=model, base_url=OPENROUTER_BASE_URL, api_key=key)
