from typing import Optional

from .config import get_settings
from .graph import build_rlm_graph, recursion_limit_for
from .llm import make_llm


def run(
    question: str,
    context: str,
    root_model: Optional[str] = None,
    sub_model: Optional[str] = None,
    max_depth: int = 1,
    max_iterations: int = 10,
) -> Optional[str]:
    """질문 + 거대 context를 받아 RLM으로 답을 구한다. 실제 OpenRouter 호출.

    root_model / sub_model 을 명시하지 않으면 설정(RLM_ROOT_MODEL /
    RLM_SUB_MODEL 환경변수 또는 기본값)을 호출 시점에 읽어 쓴다.
    """
    settings = get_settings()
    root = root_model or settings.rlm_root_model
    sub = sub_model or settings.rlm_sub_model
    graph = build_rlm_graph(
        make_llm(root), make_llm(sub), max_depth, max_iterations
    )
    result = graph.invoke(
        {"question": question, "context": context, "depth": 0},
        config={"recursion_limit": recursion_limit_for(max_iterations)},
    )
    return result.get("final_answer")
