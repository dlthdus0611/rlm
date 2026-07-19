from tests.test_graph import FakeChat, FakeSub


def test_rlm_solver_solves_and_instruments():
    from eval.systems import RlmSolver
    root = FakeChat(['```repl\nanswer["content"] = "정답 42"\nanswer["ready"] = True\n```'])
    solver = RlmSolver(root, FakeSub(), max_iterations=5)
    out = solver.solve("q", "ctx")
    assert solver.name == "rlm"
    assert out.answer == "정답 42"
    assert out.latency_s >= 0.0
    assert out.usage.total_tokens == 0          # 가짜 모델은 콜백 미발화 → 0
    assert len(out.trace) >= 1


def test_base_solver_catches_run_error():
    from eval.systems import BaseSolver
    class Boom(BaseSolver):
        name = "boom"
        def _run(self, question, context, callbacks):
            raise ValueError("터짐")
    out = Boom().solve("q", "c")
    assert out.answer is None
    assert "터짐" in out.extra["error"]


def test_rag_solver_reuses_index_across_calls(tmp_path):
    from eval.systems import RagSolver
    from rlm.config import get_settings
    from tests.test_rag_index import FakeEmbedder
    settings = get_settings()
    solver = RagSolver(FakeChat(["232개"]), FakeSub("5"), FakeEmbedder(), settings,
                       cache_dir=str(tmp_path))
    solver.use_hyde = solver.use_rerank = False   # 결정성
    ctx = "종속기업은 232개다. " * 20
    out1 = solver.solve("종속기업 몇 개?", ctx)
    out2 = solver.solve("또 질문", ctx)
    assert out1.answer == "232개"
    assert out2.answer == "232개"
    assert solver._built_key is not None          # 리트리버 메모이즈됨


def test_build_solvers_registry():
    from eval.systems import build_solvers, SYSTEMS
    from rlm.config import get_settings
    from tests.test_rag_index import FakeEmbedder
    assert set(SYSTEMS) == {"rlm", "rag"}
    solvers = build_solvers(["rlm", "rag"], root_llm=FakeChat(["x"]), sub_llm=FakeSub(),
                            embeddings=FakeEmbedder(), settings=get_settings(),
                            max_depth=1, max_iterations=5)
    assert [s.name for s in solvers] == ["rlm", "rag"]
