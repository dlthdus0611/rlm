import json

from eval.harness import QAItem, load_testset


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


from eval.harness import select_items


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

from eval.harness import Verdict, judge


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


def test_judge_parses_json_after_reasoning_with_braces():
    # 산문에 중괄호가 먼저 등장하고 그 뒤에 진짜 JSON이 오는 경우
    j = FakeJudge(['추론 {메모} 후 결론: {"label":"correct","reason":"맞음"}'])
    v = judge("q", "a", "b", j)
    assert v.label == "correct"
    assert j.i == 1  # 재시도 없이 첫 호출에서 파싱


def test_judge_parses_json_with_trailing_object():
    # 유효 JSON 뒤에 다른 객체가 붙는 경우
    j = FakeJudge(['{"label":"partial","reason":"일부"} {디버그:1}'])
    v = judge("q", "a", "b", j)
    assert v.label == "partial"
    assert j.i == 1


from eval.harness import EvalResult, run_one


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


from eval.harness import aggregate


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
