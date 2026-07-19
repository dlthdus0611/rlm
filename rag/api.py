"""RAG 공개 진입점 — make_llm/make_embeddings/build_index/Retriever/answer를 묶는다.

rlm/api.run과 대칭. LLM·임베딩을 주입하면 네트워크 없이 도므로 테스트가 이를 관통한다.
"""
from typing import Optional

from rlm.config import get_settings
from rlm.llm import make_llm

from .embeddings import make_embeddings
from .index import build_index
from .pipeline import answer as _answer
from .retriever import Retriever


def build_retriever(context, sub_llm, embeddings, settings, *,
                    cache_dir=None, use_hyde=None, use_rerank=None) -> Retriever:
    store, _info = build_index(
        context, embeddings,
        chunk_size=settings.rag_chunk_size, overlap=settings.rag_chunk_overlap,
        cache_dir=cache_dir or settings.rag_cache_dir,
    )
    return Retriever(
        store, sub_llm,
        top_k=settings.rag_top_k, top_n=settings.rag_top_n,
        use_hyde=settings.rag_use_hyde if use_hyde is None else use_hyde,
        use_rerank=settings.rag_use_rerank if use_rerank is None else use_rerank,
    )


def run(question: str, context: str, *,
        root_model: Optional[str] = None, sub_model: Optional[str] = None,
        root_llm=None, sub_llm=None, embeddings=None,
        cache_dir=None, use_hyde=None, use_rerank=None) -> Optional[str]:
    settings = get_settings()
    root = root_llm or make_llm(root_model or settings.rlm_root_model)
    sub = sub_llm or make_llm(sub_model or settings.rlm_sub_model)
    emb = embeddings or make_embeddings()
    retriever = build_retriever(context, sub, emb, settings,
                                cache_dir=cache_dir, use_hyde=use_hyde, use_rerank=use_rerank)
    return _answer(question, retriever, root).answer
