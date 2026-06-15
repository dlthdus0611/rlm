# RLM 평가 하니스 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 삼성 사업보고서 QA 테스트셋으로 RLM의 정답 정확도를 측정하는 평가 하니스(순수 로직 모듈 + CLI)를 만든다.

**Architecture:** `rlm/eval.py`에 streamlit/CLI 비의존 순수 로직(로드·필터·LLM-as-judge 채점·RLM 실행·집계)을 두고 LLM은 Runnable로 주입해 FakeChat으로 오프라인 테스트한다. `eval_run.py`는 이를 묶는 얇은 CLI다.

**Tech Stack:** Python 3.10, LangGraph(기존 `build_rlm_graph`), langchain-core 메시지, pytest, argparse.

---

## File Structure

- Create: `rlm/eval.py` — 평가 핵심 로직(데이터클래스, `load_testset`, `select_items`, `judge`, `run_one`, `aggregate`)
- Create: `eval_run.py` — 레포 루트 CLI 러너
- Create: `tests/test_eval.py` — FakeChat 기반 오프라인 단위 테스트
- Modify: `rlm/__init__.py` — eval 공개 심볼 export(선택)

`run_one`은 `graph.invoke`가 반환하는 state의 `final_answer`·`iteration`을 그대로 읽어 턴 수를 얻는다(스트리밍 불필요, `app_trace` 미사용).

---

## Task 1: 데이터클래스 + load_testset

**Files:**
- Create: `rlm/eval.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_eval.py`:

```python
import json

from rlm.eval import QAItem, load_testset


def test_load_testset_single(tmp_path):
    p = tmp_path / "qa.json"
    p.write_text(json.dumps([
        {"id": "Q001", "difficulty": "low", "question": "삼성 언제 생겼어?",
         "answer": "1969년 1월 13일", "page": 5, "section": "01_회사개요_연혁",
         "question_textbook": "삼성전자 설립일은?"}
    ], ensure_ascii=False), encoding="utf-8")

    items = load_testset(str(p))

    assert len(items) == 1
    it = items[0]
    assert isinstance(it, QAItem)
    assert it.id == "Q001"
    assert it.difficulty == "low"
    assert it.answer == "1969년 1월 13일"
    assert it.section == "01_회사개요_연혁"


def test_load_testset_cross_uses_sections(tmp_path):
    p = tmp_path / "cross.json"
    p.write_text(json.dumps([
        {"id": "C001", "difficulty": "high", "question": "q", "answer": "a",
         "page": 6, "sections": ["01_회사개요_연혁", "14_계열회사"]}
    ], ensure_ascii=False), encoding="utf-8")

    items = load_testset(str(p))

    assert items[0].sections == ["01_회사개요_연혁", "14_계열회사"]
    assert items[0].section == ""
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rlm.eval'`

- [ ] **Step 3: 최소 구현**

`rlm/eval.py`:

```python
"""RLM 평가 하니스 — 테스트셋으로 RLM 정답 정확도를 측정한다.

streamlit/CLI에 의존하지 않는 순수 로직. LLM은 .invoke()를 가진 Runnable로
주입하므로(graph.py의 root_llm/sub_llm 패턴) 네트워크 없이 FakeChat으로 테스트 가능.
"""
from __future__ import annotations

import json
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_eval.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add rlm/eval.py tests/test_eval.py
git commit -m "평가 하니스: QAItem 데이터클래스와 load_testset 추가"
```

---

## Task 2: select_items (필터 + 샘플링)

**Files:**
- Modify: `rlm/eval.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/test_eval.py`에 추가)

```python
from rlm.eval import select_items


def _items():
    return [
        QAItem(id=f"Q{i}", difficulty=("low" if i % 2 == 0 else "high"),
               question="q", answer="a")
        for i in range(10)
    ]


def test_select_items_filters_by_difficulty():
    out = select_items(_items(), difficulties=["high"])
    assert len(out) == 5
    assert all(it.difficulty == "high" for it in out)


def test_select_items_samples_n_deterministically():
    a = select_items(_items(), n=3, seed=42)
    b = select_items(_items(), n=3, seed=42)
    assert len(a) == 3
    assert [it.id for it in a] == [it.id for it in b]  # seed 고정 → 동일


def test_select_items_n_larger_than_pool_returns_all():
    out = select_items(_items(), n=999)
    assert len(out) == 10
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_eval.py -v -k select`
Expected: FAIL — `ImportError: cannot import name 'select_items'`

- [ ] **Step 3: 최소 구현** (`rlm/eval.py`에 추가, 상단 import에 `import random` 추가)

```python
import random


def select_items(items: list[QAItem], n: Optional[int] = None,
                 seed: int = 42, difficulties: Optional[list] = None) -> list[QAItem]:
    """난이도 필터 후 n개를 결정적으로(seed 고정) 샘플링한다. n이 None이면 전체."""
    pool = [it for it in items if not difficulties or it.difficulty in difficulties]
    if n is not None and n < len(pool):
        pool = random.Random(seed).sample(pool, n)
    return pool
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_eval.py -v -k select`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add rlm/eval.py tests/test_eval.py
git commit -m "평가 하니스: select_items 필터·샘플링 추가"
```

---

## Task 3: judge (LLM-as-judge)

**Files:**
- Modify: `rlm/eval.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/test_eval.py`에 추가)

기존 `tests/test_graph.py`의 FakeChat과 동일한 가짜 모델을 이 파일 상단에도 둔다.

```python
from langchain_core.messages import AIMessage

from rlm.eval import Verdict, judge


class FakeJudge:
    """정해진 문자열을 순서대로 반환하는 가짜 judge 모델."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0

    def invoke(self, messages):
        idx = min(self.i, len(self.responses) - 1)
        self.i += 1
        return AIMessage(content=self.responses[idx])


def test_judge_parses_correct():
    j = FakeJudge(['{"label": "correct", "reason": "수치 일치"}'])
    v = judge("q", "1969년", "1969년 1월", j)
    assert isinstance(v, Verdict)
    assert v.label == "correct"
    assert v.reason == "수치 일치"


def test_judge_parses_partial_with_surrounding_text():
    j = FakeJudge(['판정: {"label":"partial","reason":"일부만"} 입니다'])
    v = judge("q", "a", "b", j)
    assert v.label == "partial"


def test_judge_empty_answer_is_incorrect_without_calling_llm():
    j = FakeJudge(['{"label":"correct","reason":"x"}'])
    v = judge("q", "a", "   ", j)
    assert v.label == "incorrect"
    assert j.i == 0  # judge 모델 호출 안 함


def test_judge_unparseable_retries_then_incorrect():
    j = FakeJudge(["헛소리1", "헛소리2"])
    v = judge("q", "a", "b", j)
    assert v.label == "incorrect"
    assert j.i == 2  # 2회 시도
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_eval.py -v -k judge`
Expected: FAIL — `ImportError: cannot import name 'judge'`

- [ ] **Step 3: 최소 구현** (`rlm/eval.py`에 추가)

상단 import에 추가:

```python
import re

from langchain_core.messages import SystemMessage, HumanMessage
```

`Verdict` 데이터클래스 추가(`QAItem` 아래):

```python
@dataclass
class Verdict:
    label: str           # "correct" | "partial" | "incorrect"
    reason: str = ""
```

judge 로직:

```python
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
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    label = str(obj.get("label", "")).strip().lower()
    if label not in _VALID_LABELS:
        return None
    return Verdict(label=label, reason=str(obj.get("reason", "")))


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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_eval.py -v -k judge`
Expected: PASS (4 passed)

- [ ] **Step 5: 커밋**

```bash
git add rlm/eval.py tests/test_eval.py
git commit -m "평가 하니스: LLM-as-judge 채점(judge) 추가"
```

---

## Task 4: run_one (RLM 실행 + 채점)

**Files:**
- Modify: `rlm/eval.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/test_eval.py`에 추가)

`tests/test_graph.py`의 FakeChat/FakeSub를 이 파일에도 둔다(RLM 루트/서브 가짜 모델).

```python
from rlm.eval import EvalResult, run_one


class FakeChat:
    """RLM 루트 모델 대역 — 정해진 응답을 순서대로 반환."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0

    def invoke(self, messages):
        idx = min(self.i, len(self.responses) - 1)
        self.i += 1
        return AIMessage(content=self.responses[idx])

    def batch(self, prompts, config=None):
        return [AIMessage(content="x") for _ in prompts]


class FakeSub:
    def invoke(self, prompt):
        return AIMessage(content="고정")

    def batch(self, prompts, config=None):
        return [AIMessage(content="고정") for _ in prompts]


def test_run_one_full_path_offline():
    item = QAItem(id="Q1", difficulty="low", question="q", answer="정답 42")
    root = FakeChat([
        '```repl\nanswer["content"] = "정답 42"\nanswer["ready"] = True\n```'
    ])
    jdg = FakeJudge(['{"label":"correct","reason":"일치"}'])

    res = run_one(item, "ctx", root, FakeSub(), jdg, max_iterations=5)

    assert isinstance(res, EvalResult)
    assert res.model_answer == "정답 42"
    assert res.turns == 1
    assert res.verdict.label == "correct"
    assert res.error is None


def test_run_one_no_answer_is_incorrect():
    item = QAItem(id="Q2", difficulty="low", question="q", answer="a")
    root = FakeChat(['```repl\nprint("탐색만")\n```'])  # answer 미설정
    jdg = FakeJudge(['{"label":"correct","reason":"호출되면 안됨"}'])

    res = run_one(item, "ctx", root, FakeSub(), jdg, max_iterations=3)

    assert res.model_answer is None
    assert res.verdict.label == "incorrect"
    assert jdg.i == 0  # 미제출이면 judge 호출 안 함


def test_run_one_uses_question_textbook_field():
    item = QAItem(id="Q3", difficulty="low", question="대충질문",
                  answer="a", question_textbook="정석질문")
    captured = {}

    class CapRoot(FakeChat):
        def invoke(self, messages):
            # 첫 호출 메시지에 질문이 포함됨(메타데이터 메시지)
            captured["text"] = " ".join(
                m.content for m in messages if isinstance(m.content, str)
            )
            return super().invoke(messages)

    root = CapRoot(['```repl\nanswer["content"]="ok"\nanswer["ready"]=True\n```'])
    jdg = FakeJudge(['{"label":"correct","reason":"x"}'])

    run_one(item, "ctx", root, FakeSub(), jdg, max_iterations=5,
            question_field="question_textbook")

    assert "정석질문" in captured["text"]
    assert "대충질문" not in captured["text"]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_eval.py -v -k run_one`
Expected: FAIL — `ImportError: cannot import name 'EvalResult'` 등

- [ ] **Step 3: 최소 구현** (`rlm/eval.py`에 추가)

상단 import에 추가:

```python
from .graph import build_rlm_graph
```

`EvalResult` 데이터클래스 추가(`Verdict` 아래):

```python
@dataclass
class EvalResult:
    item: QAItem
    model_answer: Optional[str]
    turns: int
    verdict: Verdict
    error: Optional[str] = None
```

run_one 로직:

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_eval.py -v -k run_one`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add rlm/eval.py tests/test_eval.py
git commit -m "평가 하니스: run_one(RLM 실행+채점) 추가"
```

---

## Task 5: aggregate (집계)

**Files:**
- Modify: `rlm/eval.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/test_eval.py`에 추가)

```python
from rlm.eval import aggregate


def _result(diff, label, turns=2, error=None):
    return EvalResult(
        QAItem(id="x", difficulty=diff, question="q", answer="a"),
        model_answer="m", turns=turns, verdict=Verdict(label), error=error,
    )


def test_aggregate_overall_score():
    results = [
        _result("low", "correct"),
        _result("low", "partial"),
        _result("high", "incorrect"),
        _result("high", "correct"),
    ]
    agg = aggregate(results)
    # score = (correct 2 + 0.5*partial 1) / 4 = 0.625
    assert agg["overall"]["total"] == 4
    assert agg["overall"]["correct"] == 2
    assert agg["overall"]["partial"] == 1
    assert agg["overall"]["score"] == 0.625


def test_aggregate_by_difficulty_and_errors():
    results = [
        _result("low", "correct", turns=3),
        _result("low", "incorrect", error="boom"),
    ]
    agg = aggregate(results)
    assert set(agg["by_difficulty"].keys()) == {"low"}
    low = agg["by_difficulty"]["low"]
    assert low["total"] == 2
    assert low["errors"] == 1
    assert low["avg_turns"] == 3.0  # 에러 문항은 평균 턴에서 제외
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_eval.py -v -k aggregate`
Expected: FAIL — `ImportError: cannot import name 'aggregate'`

- [ ] **Step 3: 최소 구현** (`rlm/eval.py`에 추가)

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_eval.py -v -k aggregate`
Expected: PASS (2 passed)

- [ ] **Step 5: 전체 테스트 실행**

Run: `pytest -v`
Expected: 기존 그래프 테스트 + 신규 eval 테스트 모두 PASS

- [ ] **Step 6: 커밋**

```bash
git add rlm/eval.py tests/test_eval.py
git commit -m "평가 하니스: aggregate 집계 추가"
```

---

## Task 6: eval_run.py CLI 러너

**Files:**
- Create: `eval_run.py`
- Modify: `rlm/__init__.py`

CLI는 얇은 wrapper라 단위 테스트 대신 수동 스모크로 검증한다(`--help`와 가짜 키 없이 인자 파싱).

- [ ] **Step 1: `rlm/__init__.py`에 eval 심볼 export**

`rlm/__init__.py`를 다음으로 교체:

```python
from .api import run
from .graph import build_rlm_graph
from .llm import make_llm
from .repl import REPL
from .parsing import parse_code_blocks
from .eval import (
    QAItem, Verdict, EvalResult,
    load_testset, select_items, judge, run_one, aggregate,
)

__all__ = [
    "run", "build_rlm_graph", "make_llm", "REPL", "parse_code_blocks",
    "QAItem", "Verdict", "EvalResult",
    "load_testset", "select_items", "judge", "run_one", "aggregate",
]
```

- [ ] **Step 2: `eval_run.py` 작성**

```python
"""RLM 평가 CLI — 삼성 사업보고서 QA 테스트셋으로 RLM 정답 정확도를 측정한다.

실행 예:
  python eval_run.py --set single --n 10 --difficulty low --difficulty medium
  python eval_run.py --set cross --n 5 --question-field question_textbook

⚠️ 실제 OpenRouter 호출. .env 또는 환경변수에 OPENROUTER_API_KEY 필요.
모델이 생성한 코드를 in-process exec()로 실행한다(샌드박스 없음).
"""
import argparse
import json

from dotenv import load_dotenv

from rlm.config import get_settings
from rlm.eval import aggregate, load_testset, run_one, select_items
from rlm.llm import make_llm

load_dotenv()

SETS = {
    "single": ["data/qa_testset.json"],
    "cross": ["data/qa_crosssection.json"],
    "both": ["data/qa_testset.json", "data/qa_crosssection.json"],
}
CONTEXT_PATH = "data/samsung_2023.txt"


def _print_summary(agg: dict) -> None:
    o = agg["overall"]
    print("\n===== 요약 =====")
    print(f"전체 {o['total']}문항 | score={o['score']} | "
          f"correct={o['correct']} partial={o['partial']} "
          f"incorrect={o['incorrect']} errors={o['errors']} | avg_turns={o['avg_turns']}")
    for diff, b in agg["by_difficulty"].items():
        print(f"  [{diff:<7}] {b['total']:>3}문항 | score={b['score']:<5} "
              f"c={b['correct']} p={b['partial']} i={b['incorrect']} e={b['errors']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RLM 평가 하니스")
    parser.add_argument("--set", choices=SETS, default="single", help="테스트셋")
    parser.add_argument("--n", type=int, default=None, help="샘플 수(미지정=전체)")
    parser.add_argument("--seed", type=int, default=42, help="샘플링 seed")
    parser.add_argument("--difficulty", action="append", default=None,
                        choices=["low", "medium", "high", "expert"],
                        help="난이도 필터(반복 지정 가능)")
    parser.add_argument("--question-field", choices=["question", "question_textbook"],
                        default="question", help="대충질문 vs 정석")
    parser.add_argument("--judge-model", default=None, help="채점 모델(기본 root)")
    parser.add_argument("--root-model", default=None)
    parser.add_argument("--sub-model", default=None)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--out", default="data/eval_results.json")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY 없음 — .env 또는 환경변수에 설정하세요.")

    root_model = args.root_model or settings.rlm_root_model
    sub_model = args.sub_model or settings.rlm_sub_model
    judge_model = args.judge_model or root_model

    with open(CONTEXT_PATH, encoding="utf-8") as f:
        context = f.read()

    items = []
    for path in SETS[args.set]:
        items += load_testset(path)
    items = select_items(items, n=args.n, seed=args.seed, difficulties=args.difficulty)
    print(f"context {len(context)}자 | {len(items)}문항 평가 시작 "
          f"(root={root_model}, sub={sub_model}, judge={judge_model})")

    root_llm = make_llm(root_model)
    sub_llm = make_llm(sub_model)
    judge_llm = make_llm(judge_model)

    results = []
    for idx, item in enumerate(items, 1):
        res = run_one(item, context, root_llm, sub_llm, judge_llm,
                      max_depth=args.max_depth, max_iterations=args.max_iterations,
                      question_field=args.question_field)
        flag = res.error or res.verdict.label
        print(f"[{idx}/{len(items)}] {item.id} ({item.difficulty}) "
              f"turns={res.turns} -> {flag}")
        results.append(res)

    agg = aggregate(results)
    _print_summary(agg)

    payload = {
        "config": {
            "set": args.set, "n": args.n, "seed": args.seed,
            "difficulty": args.difficulty, "question_field": args.question_field,
            "root_model": root_model, "sub_model": sub_model, "judge_model": judge_model,
            "max_iterations": args.max_iterations, "max_depth": args.max_depth,
        },
        "aggregate": agg,
        "results": [
            {
                "id": r.item.id, "difficulty": r.item.difficulty,
                "question": getattr(r.item, args.question_field, "") or r.item.question,
                "gold": r.item.answer, "model_answer": r.model_answer,
                "turns": r.turns, "label": r.verdict.label,
                "reason": r.verdict.reason, "error": r.error,
            }
            for r in results
        ],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: import·인자 파싱 스모크 테스트**

Run: `python -c "import eval_run"`
Expected: 오류 없음(키 없어도 import 시점엔 통과 — main()은 호출 안 됨)

Run: `python eval_run.py --help`
Expected: 인자 목록 출력, 정상 종료(0)

- [ ] **Step 4: 전체 단위 테스트 재확인**

Run: `pytest -v`
Expected: 모든 테스트 PASS

- [ ] **Step 5: 커밋**

```bash
git add eval_run.py rlm/__init__.py
git commit -m "평가 하니스: eval_run.py CLI 러너 추가"
```

---

## Task 7: 실제 스모크 실행(선택, 수동)

API 키가 있는 환경에서 소량 실제 실행으로 동작을 확인한다(자동 테스트 아님).

- [ ] **Step 1: 저난도 3문항 실제 평가**

Run: `python eval_run.py --set single --n 3 --difficulty low`
Expected: 3문항 실행 로그 + 요약표 출력, `data/eval_results.json` 생성. 답·판정이 그럴듯한지 눈으로 확인.

- [ ] **Step 2: 결과 확인 후 필요 시 judge 프롬프트 조정**

`data/eval_results.json`의 `reason`을 보고 채점이 합리적인지 점검. 명백히 어긋나면 `rlm/eval.py`의 `JUDGE_SYSTEM`을 다듬고 Task 3 테스트가 여전히 통과하는지 확인 후 커밋.

---

## 비고

- `data/eval_results.json`은 산출물이므로 커밋하지 않는다(필요 시 `.gitignore`에 추가 검토).
- RLM은 문항당 다수 API 호출 → 대량 평가는 비용·시간 큼. `--n`으로 소량부터.
- RAG 베이스라인·Streamlit 평가 페이지는 차후 별도 계획. `rlm/eval.py`의 `judge`/`aggregate`는 그대로 재사용 가능.
