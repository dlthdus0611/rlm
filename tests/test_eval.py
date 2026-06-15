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
