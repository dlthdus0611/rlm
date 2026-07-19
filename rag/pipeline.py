"""검색 → 단일 생성. RLM의 반복 루프와 대비되는 선형 파이프라인.

gen_llm은 주입(=비교 공정성상 RLM의 root_model과 동일 모델). callbacks를 config로 흘려
생성 토큰이 상위 usage 계측에 잡히게 한다.
"""
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage

from .prompts import RAG_SYSTEM, build_answer_prompt
from .retriever import Passage


@dataclass
class RagResult:
    answer: str
    passages: list = field(default_factory=list)


def _content(msg) -> str:
    c = getattr(msg, "content", "")
    return c if isinstance(c, str) else str(c)


def answer(question: str, retriever, gen_llm, callbacks=None) -> RagResult:
    passages: list[Passage] = retriever.retrieve(question)
    messages = [
        SystemMessage(content=RAG_SYSTEM),
        HumanMessage(content=build_answer_prompt(question, [p.text for p in passages])),
    ]
    config = {"callbacks": callbacks} if callbacks is not None else None
    out = _content(gen_llm.invoke(messages, config=config))
    return RagResult(answer=out.strip(), passages=passages)
