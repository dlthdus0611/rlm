import math

from langchain_core.embeddings import Embeddings


class FakeEmbedder(Embeddings):
    """문자 코드 합을 seed로 결정적 벡터를 내는 가짜 임베더(네트워크 없음).

    FAISS가 Embeddings 인스턴스로 인식하도록 langchain_core.embeddings.Embeddings를 상속한다.
    """
    def __init__(self, dim=16):
        self.dim = dim

    def _vec(self, text):
        v = [0.0] * self.dim
        for i, ch in enumerate(text):
            v[i % self.dim] += (ord(ch) % 17) / 17.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_split_text_overlap():
    from rag.index import split_text
    text = "가나다라마바사아자차카타파하" * 20
    chunks = split_text(text, chunk_size=50, overlap=10)
    assert len(chunks) >= 2
    assert all(len(c) <= 50 for c in chunks)
    # 인접 청크는 overlap만큼 겹친다
    assert chunks[0][-10:] == chunks[1][:10]


def test_build_index_and_search(tmp_path):
    from rag.index import build_index
    ctx = "삼성전자는 1969년 설립되었다. " * 30 + "종속기업은 232개다. " * 30
    store, info = build_index(ctx, FakeEmbedder(), chunk_size=60, overlap=10,
                              cache_dir=str(tmp_path))
    assert info["cache_hit"] is False and info["n_chunks"] >= 2
    hits = store.similarity_search("종속기업 몇 개", k=3)
    assert len(hits) == 3


def test_build_index_disk_cache(tmp_path):
    from rag.index import build_index
    ctx = "반복 텍스트 " * 100
    _, info1 = build_index(ctx, FakeEmbedder(), chunk_size=40, overlap=5,
                           cache_dir=str(tmp_path))
    _, info2 = build_index(ctx, FakeEmbedder(), chunk_size=40, overlap=5,
                           cache_dir=str(tmp_path))
    assert info1["cache_hit"] is False
    assert info2["cache_hit"] is True
    assert info1["cache_key"] == info2["cache_key"]


def test_build_index_key_changes_with_params(tmp_path):
    from rag.index import build_index
    ctx = "반복 텍스트 " * 100
    _, a = build_index(ctx, FakeEmbedder(), chunk_size=40, overlap=5, cache_dir=str(tmp_path))
    _, b = build_index(ctx, FakeEmbedder(), chunk_size=80, overlap=5, cache_dir=str(tmp_path))
    assert a["cache_key"] != b["cache_key"]
