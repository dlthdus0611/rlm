"""RLM 평가 하니스 — 테스트셋으로 RLM 정답 정확도를 측정한다.

streamlit/CLI에 의존하지 않는 순수 로직. LLM은 .invoke()를 가진 Runnable로
주입하므로(graph.py의 root_llm/sub_llm 패턴) 네트워크 없이 FakeChat으로 테스트 가능.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from rlm.graph import build_rlm_graph


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


@dataclass
class Verdict:
    label: str           # "correct" | "partial" | "incorrect"
    reason: str = ""


@dataclass
class EvalResult:
    item: QAItem
    model_answer: Optional[str]
    turns: int
    verdict: Verdict
    error: Optional[str] = None


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


JUDGE_SYSTEM = (
    "너는 QA 채점자다. 주어진 '정답'과 '모델답변'이 핵심적으로 일치하는지 판정한다.\n"
    "- correct: 정답의 핵심 사실과 수치가 모두 일치한다(표현·어순 차이는 무방하나, "
    "숫자는 반올림·단위까지 일치해야 한다).\n"
    "- partial: 여러 부분 중 일부만 맞거나, 핵심 수치는 맞지만 요구된 부가정보가 빠졌다.\n"
    "- incorrect: 틀리거나 핵심이 누락되었거나 답이 없다.\n"
    '반드시 JSON 한 줄로만 답하라: {"label": "correct|partial|incorrect", "reason": "간단한 이유"}'
)

_VALID_LABELS = {"correct", "partial", "incorrect"}


def _build_judge_prompt(question: str, gold: str, candidate: str) -> str:
    return (
        f"[질문]\n{question}\n\n"
        f"[정답]\n{gold}\n\n"
        f"[모델답변]\n{candidate}\n\n"
        "위 모델답변을 정답 기준으로 채점하라."
    )


def _parse_verdict(text: str) -> Optional[Verdict]:
    # 응답에 산문이나 여분 객체가 섞여도, 파싱 가능하고 유효 label을 가진 첫 객체를 택한다.
    # 먼저 전체를 시도(reason에 중괄호가 있어도 처리), 이어서 개별 {...} 후보를 훑는다.
    candidates = [text.strip()]
    candidates += [m.group(0) for m in re.finditer(r"\{.*?\}", text, re.S)]
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        label = str(obj.get("label", "")).strip().lower()
        if label in _VALID_LABELS:
            return Verdict(label=label, reason=str(obj.get("reason", "")))
    return None


def judge(question: str, gold: str, candidate: Optional[str], judge_llm) -> Verdict:
    """LLM-as-judge로 모델답변을 정답과 대조해 Verdict를 낸다.

    candidate가 비었으면 모델 호출 없이 incorrect. 파싱 실패 시 1회 재시도 후 incorrect.
    """
    if candidate is None or not str(candidate).strip():
        return Verdict("incorrect", "모델이 답을 제출하지 않음")
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=_build_judge_prompt(question, gold, str(candidate))),
    ]
    text = ""
    for _ in range(2):
        raw = judge_llm.invoke(messages).content
        text = raw if isinstance(raw, str) else str(raw)
        verdict = _parse_verdict(text)
        if verdict is not None:
            return verdict
    return Verdict("incorrect", f"judge 파싱 실패: {text[:200]}")


def run_one(item: QAItem, context: str, root_llm, sub_llm, judge_llm,
            max_depth: int = 1, max_iterations: int = 10,
            question_field: str = "question") -> EvalResult:
    """한 문항에 대해 RLM을 실행하고 채점해 EvalResult를 낸다.

    question_field로 '대충질문'(question) vs '정석'(question_textbook)을 선택.
    실행 예외는 잡아 EvalResult.error에 담는다(배치 계속).
    """
    question = getattr(item, question_field, "") or item.question
    try:
        graph = build_rlm_graph(root_llm, sub_llm, max_depth, max_iterations)
        state = graph.invoke(
            {"question": question, "context": context, "depth": 0},
            config={"recursion_limit": 2 * max_iterations + 10},
        )
    except Exception as exc:  # noqa: BLE001 - 문항 단위 실패는 기록하고 계속
        return EvalResult(item, None, 0, Verdict("incorrect", "실행 오류"), error=str(exc))
    answer = state.get("final_answer")
    turns = state.get("iteration", 0)
    verdict = judge(question, item.answer, answer, judge_llm)
    return EvalResult(item, answer, turns, verdict)


def _bucket(results: list[EvalResult]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r.verdict.label == "correct")
    partial = sum(1 for r in results if r.verdict.label == "partial")
    incorrect = sum(1 for r in results if r.verdict.label == "incorrect")
    errors = sum(1 for r in results if r.error)
    turns = [r.turns for r in results if not r.error]
    score = (correct + 0.5 * partial) / total if total else 0.0
    return {
        "total": total, "correct": correct, "partial": partial,
        "incorrect": incorrect, "errors": errors, "score": round(score, 3),
        "avg_turns": round(sum(turns) / len(turns), 2) if turns else 0.0,
    }


def aggregate(results: list[EvalResult]) -> dict:
    """전체 + 난이도별 집계. score = (correct + 0.5*partial)/total."""
    by_difficulty = {}
    for diff in ("low", "medium", "high", "expert"):
        subset = [r for r in results if r.item.difficulty == diff]
        if subset:
            by_difficulty[diff] = _bucket(subset)
    return {"overall": _bucket(results), "by_difficulty": by_difficulty}
