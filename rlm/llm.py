from langchain_openai import ChatOpenAI

from .config import get_settings


def make_llm(model: str) -> ChatOpenAI:
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY 환경변수가 필요합니다.")
    return ChatOpenAI(
        model=model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )
