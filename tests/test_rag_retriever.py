from langchain_core.documents import Document


class FakeStore:
    """유사도 검색을 흉내 내는 가짜 벡터스토어 — 질문 토큰 포함 수로 정렬."""
    def __init__(self, texts):
        self.docs = [Document(page_content=t, metadata={"i": i}) for i, t in enumerate(texts)]

    def _rank(self, query):
        toks = [w for w in query.split() if w]
        return sorted(self.docs, key=lambda d: -sum(t in d.page_content for t in toks))

    def similarity_search(self, query, k):
        return self._rank(query)[:k]

    def max_marginal_relevance_search(self, query, k, fetch_k):
        return self._rank(query)[:k]


class FakeScorer:
    """리랭킹용 sub_llm — passage 안 '점수N'을 관련도로 되돌려주는 결정적 가짜."""
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        import re
        from langchain_core.messages import AIMessage
        self.calls += 1
        text = prompt if isinstance(prompt, str) else str(prompt)
        m = re.search(r"점수(\d+)", text)
        return AIMessage(content=m.group(1) if m else "5")


def test_retrieve_plain_topk_no_toggles():
    from rag.retriever import Retriever
    store = FakeStore(["종속기업 232개", "계열사 63개", "잡음 텍스트", "또 잡음"])
    r = Retriever(store, FakeScorer(), top_k=2, top_n=4, use_hyde=False, use_rerank=False)
    out = r.retrieve("종속기업 232개")
    assert len(out) == 2
    assert out[0].text == "종속기업 232개"


def test_retrieve_rerank_orders_by_score():
    from rag.retriever import Retriever
    store = FakeStore(["A 점수3", "B 점수9", "C 점수1"])
    scorer = FakeScorer()
    r = Retriever(store, scorer, top_k=2, top_n=3, use_hyde=False, use_rerank=True)
    out = r.retrieve("무엇")
    assert [p.text for p in out] == ["B 점수9", "A 점수3"]
    assert scorer.calls == 3   # 후보 3개 각각 채점


def test_retrieve_hyde_calls_sub_llm():
    from rag.retriever import Retriever
    store = FakeStore(["가", "나"])
    scorer = FakeScorer()
    r = Retriever(store, scorer, top_k=1, top_n=2, use_hyde=True, use_rerank=False)
    r.retrieve("질문")
    assert scorer.calls >= 1   # HyDE 가설 생성 호출
