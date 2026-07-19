"""강화 RAG 리트리버 — HyDE 쿼리 확장 → 벡터 후보(MMR) → LLM 리랭킹 → top-k.

각 단계는 토글이며 전부 끄면 벡터 top-k만 남는(표준 RAG) 파생 구조. sub_llm은 주입받아
HyDE/리랭킹에 쓰며, 그 토큰도 상위 계측(callbacks)에 잡히도록 solver가 콜백을 전파한다.
"""
import re
from dataclasses import dataclass

from .prompts import build_hyde_prompt, build_rerank_prompt


@dataclass
class Passage:
    text: str
    i: int
    score: float = 0.0


def _content(msg) -> str:
    c = getattr(msg, "content", msg)
    return c if isinstance(c, str) else str(c)


class Retriever:
    def __init__(self, store, sub_llm, *, top_k, top_n, use_hyde, use_rerank):
        self.store = store
        self.sub_llm = sub_llm
        self.top_k = top_k
        self.top_n = top_n
        self.use_hyde = use_hyde
        self.use_rerank = use_rerank

    def retrieve(self, question: str) -> list[Passage]:
        query = question
        if self.use_hyde:
            hypo = _content(self.sub_llm.invoke(build_hyde_prompt(question)))
            query = f"{question}\n{hypo}"
        # 리랭킹이 있으면 top_n 후보를, 없으면 바로 top_k를 뽑는다.
        want = self.top_n if self.use_rerank else self.top_k
        docs = self.store.max_marginal_relevance_search(query, k=want, fetch_k=want * 2)
        cands = [Passage(d.page_content, d.metadata.get("i", -1)) for d in docs]
        if not self.use_rerank:
            return cands[:self.top_k]
        for p in cands:
            raw = _content(self.sub_llm.invoke(build_rerank_prompt(question, p.text)))
            m = re.search(r"\d+", raw)
            p.score = float(m.group(0)) if m else 0.0
        cands.sort(key=lambda p: -p.score)
        return cands[:self.top_k]
