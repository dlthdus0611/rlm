"""RAG 검색 임베딩 팩토리 — OpenAI 임베딩을 생성한다(실제 OpenAI 호출).

llm.py가 ChatOpenAI를 격리하듯, 임베딩 구체 생성을 여기로 격리한다. 그래프/리트리버는
.embed_documents()/.embed_query()를 가진 객체를 주입받으므로 테스트는 FakeEmbedder로 대체한다.
"""
from typing import Optional

from langchain_openai import OpenAIEmbeddings

from rlm.config import get_settings


def make_embeddings(model: Optional[str] = None) -> OpenAIEmbeddings:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다(RAG 임베딩).")
    return OpenAIEmbeddings(
        model=model or settings.rag_embed_model,
        api_key=settings.openai_api_key,
    )
