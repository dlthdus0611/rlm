"""문서를 passage로 재청킹해 FAISS 벡터 인덱스를 만든다(디스크 캐시).

원본 repl.py가 전체 문서를 통째로 다루는 것과 대비되는 RAG의 출발점: 문서를 작은 조각으로
쪼개 임베딩해 두고 질문마다 유사한 조각만 꺼낸다. 임베딩은 주입되므로 테스트는 FakeEmbedder로 돈다.
반복 평가에서 재임베딩을 피하려 내용+파라미터 해시를 키로 디스크에 캐시한다.
"""
import hashlib
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """길이 기준으로 겹침(overlap)을 두고 자른다. 표준 RAG 청킹의 단순판."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size는 overlap보다 커야 합니다.")
    step = chunk_size - overlap
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += step
    return [c for c in chunks if c.strip()]


def _cache_key(context: str, chunk_size: int, overlap: int) -> str:
    h = hashlib.sha256()
    h.update(f"{chunk_size}:{overlap}:".encode())
    h.update(context.encode("utf-8"))
    return h.hexdigest()[:16]


def build_index(context: str, embeddings, *, chunk_size: int, overlap: int,
                cache_dir: str) -> tuple[FAISS, dict]:
    key = _cache_key(context, chunk_size, overlap)
    path = os.path.join(cache_dir, key)
    if os.path.isdir(path):
        store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        return store, {"cache_hit": True, "n_chunks": store.index.ntotal, "cache_key": key}
    chunks = split_text(context, chunk_size, overlap)
    docs = [Document(page_content=c, metadata={"i": i}) for i, c in enumerate(chunks)]
    store = FAISS.from_documents(docs, embeddings)
    os.makedirs(cache_dir, exist_ok=True)
    store.save_local(path)
    return store, {"cache_hit": False, "n_chunks": len(chunks), "cache_key": key}
