"""RLM 평가 하니스 — 테스트셋으로 RLM 정답 정확도를 측정한다.

streamlit/CLI에 의존하지 않는 순수 로직. LLM은 .invoke()를 가진 Runnable로
주입하므로(graph.py의 root_llm/sub_llm 패턴) 네트워크 없이 FakeChat으로 테스트 가능.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QAItem:
    id: str
    difficulty: str
    question: str
    answer: str
    page: object = None
    section: str = ""                              # 단일섹션 테스트셋
    sections: list = field(default_factory=list)   # 교차 종합형 테스트셋
    question_textbook: str = ""


def load_testset(path: str) -> list[QAItem]:
    """qa_testset.json / qa_crosssection.json 을 QAItem 리스트로 읽는다."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for d in data:
        items.append(QAItem(
            id=d["id"],
            difficulty=d["difficulty"],
            question=d.get("question", ""),
            answer=str(d.get("answer", "")),
            page=d.get("page"),
            section=d.get("section", ""),
            sections=d.get("sections", []),
            question_textbook=d.get("question_textbook", ""),
        ))
    return items


def select_items(items: list[QAItem], n: Optional[int] = None,
                 seed: int = 42, difficulties: Optional[list] = None) -> list[QAItem]:
    """난이도 필터 후 n개를 결정적으로(seed 고정) 샘플링한다. n이 None이면 전체."""
    pool = [it for it in items if not difficulties or it.difficulty in difficulties]
    if n is not None and n < len(pool):
        pool = random.Random(seed).sample(pool, n)
    return pool
