from typing import Optional

from .graph import build_rlm_graph
from .llm import make_llm, DEFAULT_ROOT_MODEL, DEFAULT_SUB_MODEL


def run(
    question: str,
    context: str,
    root_model: str = DEFAULT_ROOT_MODEL,
    sub_model: str = DEFAULT_SUB_MODEL,
    max_depth: int = 1,
    max_iterations: int = 10,
) -> Optional[str]:
    """질문 + 거대 context를 받아 RLM으로 답을 구한다. 실제 OpenRouter 호출."""
    graph = build_rlm_graph(
        make_llm(root_model), make_llm(sub_model), max_depth, max_iterations
    )
    result = graph.invoke(
        {"question": question, "context": context, "depth": 0},
        config={"recursion_limit": 2 * max_iterations + 10},
    )
    return result.get("final_answer")
