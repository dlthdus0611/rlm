from eval.harness import QAItem


class FakeJudge:
    """judge 프롬프트에 gold와 candidate가 함께 들어오며, 둘 다 '232'면 correct."""
    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        text = " ".join(getattr(m, "content", "") for m in messages)
        label = "correct" if text.count("232") >= 2 else "incorrect"
        return AIMessage(content='{"label": "%s", "reason": "t"}' % label)


class _P:
    def __init__(self, text):
        self.text = text


class StubSolver:
    def __init__(self, name, answer, passages=()):
        self.name = name
        self._answer = answer
        self._passages = passages

    def solve(self, question, context):
        from eval.systems import SolverOutput, Usage
        return SolverOutput(self._answer, Usage(10, 5, 15), 0.1,
                            trace=[], extra={"passages": [_P(t) for t in self._passages]})


def _item():
    return QAItem(id="Q1", difficulty="low", question="종속기업 몇 개?",
                  answer="232개", evidence=["종속기업은 232개다"])


def test_run_item_scores_each_system():
    from eval.compare import run_item
    item = _item()
    solvers = [StubSolver("rlm", "232개"),
               StubSolver("rag", "232개", passages=["종속기업은 232개다"])]
    rec = run_item(item, "ctx", solvers, FakeJudge(), "question")
    assert rec["per_system"]["rlm"]["label"] == "correct"
    assert rec["per_system"]["rag"]["evidence_hit"] is True
    assert rec["per_system"]["rlm"]["usage"].total_tokens == 15


def test_aggregate_compare_buckets_by_system():
    from eval.compare import run_item, aggregate_compare
    recs = [run_item(_item(), "ctx",
                     [StubSolver("rlm", "232개"), StubSolver("rag", "몰라")],
                     FakeJudge(), "question")]
    agg = aggregate_compare(recs, ["rlm", "rag"])
    assert agg["rlm"]["correct"] == 1
    assert agg["rag"]["correct"] == 0
    assert agg["rlm"]["avg_input_tokens"] == 10.0


def test_to_compare_payload_shape():
    from eval.compare import run_item, aggregate_compare, to_compare_payload
    recs = [run_item(_item(), "ctx", [StubSolver("rlm", "232개")], FakeJudge(), "question")]
    agg = aggregate_compare(recs, ["rlm"])
    payload = to_compare_payload({"set": "single"}, agg, recs)
    assert payload["config"]["set"] == "single"
    assert "rlm" in payload["systems"]
    assert payload["systems"]["rlm"]["aggregate"]["correct"] == 1
    assert payload["items"][0]["id"] == "Q1"
