from tests.test_compare import StubSolver, FakeJudge, _item


def test_run_compare_stream_events():
    from app.compare_run import run_compare_stream
    items = [_item(), _item()]
    solvers = [StubSolver("rlm", "232개"),
               StubSolver("rag", "232개", passages=["종속기업은 232개다"])]
    events = list(run_compare_stream(items, "ctx", solvers, FakeJudge()))
    kinds = [e.kind for e in events]
    assert kinds.count("item_done") == 2
    assert kinds[-1] == "run_done"
    assert events[-1].aggregate["rlm"]["correct"] == 2
