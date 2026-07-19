from langchain_core.messages import AIMessage

from eval.harness import QAItem, EvalResult, Verdict
from app.eval_run import run_eval_stream, to_payload


class FakeChat:
    """미리 정한 응답을 순서대로 반환하는 가짜 루트 모델."""
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
    def __init__(self, reply="고정응답"):
        self.reply = reply
    def invoke(self, prompt):
        return AIMessage(content=self.reply)
    def batch(self, prompts, config=None):
        return [AIMessage(content=self.reply) for _ in prompts]


class FakeJudge:
    """항상 정해진 라벨 JSON을 뱉는 가짜 채점 모델."""
    def __init__(self, label="correct"):
        self.label = label
    def invoke(self, messages):
        return AIMessage(content=f'{{"label": "{self.label}", "reason": "테스트"}}')


ANSWER_CODE = (
    '```repl\nanswer["content"] = "정답 42"\nanswer["ready"] = True\n```'
)


def _item(id="Q1", difficulty="low"):
    return QAItem(id=id, difficulty=difficulty, question="q", answer="정답 42")


def test_events_order_for_single_item():
    root = FakeChat([ANSWER_CODE])
    events = list(run_eval_stream(
        [_item()], "ctx", root, FakeSub(), FakeJudge(), max_iterations=5))
    kinds = [e.kind for e in events]
    # trace(들) → item_done → run_done 순서
    assert kinds[-1] == "run_done"
    assert "item_done" in kinds
    assert kinds.index("item_done") < kinds.index("run_done")
    assert any(k == "trace" for k in kinds[:kinds.index("item_done")])


def test_item_done_carries_scored_result():
    root = FakeChat([ANSWER_CODE])
    events = list(run_eval_stream(
        [_item()], "ctx", root, FakeSub(), FakeJudge("correct"), max_iterations=5))
    done = [e for e in events if e.kind == "item_done"][0]
    assert done.result.model_answer == "정답 42"
    assert done.result.verdict.label == "correct"
    assert done.result.turns >= 1


def test_run_done_aggregate_matches_results():
    root = FakeChat([ANSWER_CODE])
    events = list(run_eval_stream(
        [_item("Q1"), _item("Q2")], "ctx", root, FakeSub(), FakeJudge("correct"),
        max_iterations=5))
    run_done = events[-1]
    assert run_done.kind == "run_done"
    assert len(run_done.results) == 2
    assert run_done.aggregate["overall"]["total"] == 2
    assert run_done.aggregate["overall"]["correct"] == 2


def test_two_items_run_independently():
    root = FakeChat([ANSWER_CODE])  # 각 문항이 새 그래프라 첫 응답을 재사용
    events = list(run_eval_stream(
        [_item("Q1"), _item("Q2")], "ctx", root, FakeSub(), FakeJudge(),
        max_iterations=5))
    done_ids = [e.result.item.id for e in events if e.kind == "item_done"]
    assert done_ids == ["Q1", "Q2"]


def test_no_answer_submitted_is_incorrect():
    # 코드 블록이 없어 answer 미제출 → 턴 소진 → final_answer None
    root = FakeChat(["그냥 생각만 합니다(코드 없음)."])
    events = list(run_eval_stream(
        [_item()], "ctx", root, FakeSub(), FakeJudge("correct"), max_iterations=3))
    done = [e for e in events if e.kind == "item_done"][0]
    assert done.result.model_answer is None
    assert done.result.verdict.label == "incorrect"


def test_to_payload_shape_matches_runner():
    item = _item("Q1")
    results = [EvalResult(item, "정답 42", 2, Verdict("correct", "ok"))]
    agg = {"overall": {"total": 1}, "by_difficulty": {}}
    payload = to_payload({"set": "single"}, agg, results, "question")
    assert payload["config"] == {"set": "single"}
    assert payload["aggregate"] == agg
    row = payload["results"][0]
    assert row["id"] == "Q1"
    assert row["gold"] == "정답 42"
    assert row["model_answer"] == "정답 42"
    assert row["label"] == "correct"
    assert row["turns"] == 2
    assert row["error"] is None


def test_to_payload_uses_question_field():
    item = QAItem(id="Q1", difficulty="low", question="대충", answer="a",
                  question_textbook="정석질문")
    results = [EvalResult(item, "a", 1, Verdict("correct"))]
    payload = to_payload({}, {}, results, "question_textbook")
    assert payload["results"][0]["question"] == "정석질문"
