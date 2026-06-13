from langchain_core.messages import AIMessage

from rlm.graph import build_rlm_graph


class FakeChat:
    """미리 정한 응답 문자열을 순서대로 반환하는 가짜 채팅 모델."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        idx = min(self.i, len(self.responses) - 1)
        self.i += 1
        return AIMessage(content=self.responses[idx])

    def batch(self, prompts, config=None):
        return [AIMessage(content=f"분류:{p[:8]}") for p in prompts]


class FakeSub:
    """llm_query/llm_query_batched가 호출하는 sub 모델. 고정 응답."""

    def __init__(self, reply="고정응답"):
        self.reply = reply

    def invoke(self, prompt):
        return AIMessage(content=self.reply)

    def batch(self, prompts, config=None):
        return [AIMessage(content=self.reply) for _ in prompts]


def test_graph_single_turn_answer():
    root = FakeChat([
        '계획: 바로 답합니다.\n```repl\nanswer["content"] = "정답 42"\nanswer["ready"] = True\n```'
    ])
    graph = build_rlm_graph(root, FakeSub(), max_iterations=5)
    result = graph.invoke({"question": "q", "context": "ctx", "depth": 0})
    assert result["final_answer"] == "정답 42"


def test_graph_context_not_in_model_messages():
    big_context = "비밀본문" * 100
    root = FakeChat([
        '```repl\nanswer["content"] = "done"\nanswer["ready"] = True\n```'
    ])
    graph = build_rlm_graph(root, FakeSub(), max_iterations=5)
    graph.invoke({"question": "q", "context": big_context, "depth": 0})
    first_call_text = " ".join(
        m.content for m in root.invocations[0] if isinstance(m.content, str)
    )
    assert "비밀본문" not in first_call_text
    assert "q" in first_call_text


def test_graph_multi_turn_inspect_then_answer():
    root = FakeChat([
        '먼저 살펴봅니다.\n```repl\nprint(context[:5])\n```',
        '이제 답합니다.\n```repl\nanswer["content"] = "ok"\nanswer["ready"] = True\n```',
    ])
    graph = build_rlm_graph(root, FakeSub(), max_iterations=5)
    result = graph.invoke({"question": "q", "context": "안녕하세요", "depth": 0})
    assert result["final_answer"] == "ok"
    assert root.i == 2


def test_graph_llm_query_used_in_code():
    root = FakeChat([
        '```repl\nr = llm_query("분류해줘")\nanswer["content"] = r\nanswer["ready"] = True\n```'
    ])
    graph = build_rlm_graph(root, FakeSub(reply="환불=예"), max_iterations=5)
    result = graph.invoke({"question": "q", "context": "c", "depth": 0})
    assert result["final_answer"] == "환불=예"


def test_graph_llm_query_batched_preserves_order():
    root = FakeChat([
        '```repl\nrs = llm_query_batched(["a", "b", "c"])\n'
        'answer["content"] = str(len(rs))\nanswer["ready"] = True\n```'
    ])
    graph = build_rlm_graph(root, FakeSub(reply="x"), max_iterations=5)
    result = graph.invoke({"question": "q", "context": "c", "depth": 0})
    assert result["final_answer"] == "3"


def test_rlm_query_falls_back_to_llm_at_depth_limit():
    # max_depth=0: 루트(depth 0)의 rlm_query는 next_depth=1 > 0 이라 llm_query 폴백.
    root = FakeChat([
        '```repl\nr = rlm_query("하위 질문", "하위 컨텍스트")\n'
        'answer["content"] = r\nanswer["ready"] = True\n```'
    ])
    graph = build_rlm_graph(root, FakeSub(reply="폴백답"), max_depth=0, max_iterations=5)
    result = graph.invoke({"question": "q", "context": "c", "depth": 0})
    assert result["final_answer"] == "폴백답"


def test_rlm_query_recurses_when_allowed():
    # max_depth=1: 루트의 rlm_query는 자식 그래프를 depth=1로 invoke.
    # 부모/자식이 같은 root_llm을 쓰므로 응답 시퀀스를 합쳐 설계.
    root = FakeChat([
        '```repl\nr = rlm_query("자식아 풀어", "자식 컨텍스트")\n'
        'answer["content"] = "부모가 받은: " + r\nanswer["ready"] = True\n```',
        '```repl\nanswer["content"] = "자식답"\nanswer["ready"] = True\n```',
    ])
    graph = build_rlm_graph(root, FakeSub(), max_depth=1, max_iterations=5)
    result = graph.invoke({"question": "q", "context": "c", "depth": 0})
    assert result["final_answer"] == "부모가 받은: 자식답"


def test_graph_terminates_at_max_iterations_without_answer():
    # 모델이 절대 answer를 세팅하지 않음 -> max_iterations 에서 종료, final_answer None
    root = FakeChat(['```repl\nprint("계속 탐색만")\n```'])  # 항상 같은 응답
    graph = build_rlm_graph(root, FakeSub(), max_iterations=3)
    result = graph.invoke({"question": "q", "context": "c", "depth": 0})
    assert result.get("final_answer") is None
    assert result["iteration"] == 3
