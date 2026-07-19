from typing import Optional

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage

from .parsing import parse_code_blocks, _truncate
from .repl import REPL
from .prompts import (
    SYSTEM_PROMPT, ORCHESTRATOR_ADDENDUM,
    build_metadata_message, build_turn_prompt,
)

BATCH_CONCURRENCY = 8


def recursion_limit_for(max_iterations: int) -> int:
    """LangGraph invoke/stream의 recursion_limit(노드 스텝 상한).

    setup(1) + max_iterations*(call_model + execute_code) + 여유. LangGraph 기본값 25를
    넘기지 않도록 잡는다. graph/api/harness/eval_run 등 모든 호출부가 이 한 함수를 쓴다.
    """
    return 2 * max_iterations + 10


class RLMState(TypedDict, total=False):
    question: str
    context: str
    messages: Annotated[list, add_messages]
    depth: int
    iteration: int
    final_answer: Optional[str]
    repl: object


def build_rlm_graph(root_llm, sub_llm, max_depth: int = 1, max_iterations: int = 10):
    """RLM 제어 루프를 구현한 compiled LangGraph를 반환한다.

    root_llm / sub_llm 은 .invoke(...)/.batch(...) 를 가진 Runnable(또는 가짜 객체).
    """
    _recursion_limit = recursion_limit_for(max_iterations)

    def _llm_query(prompt: str) -> str:
        return sub_llm.invoke(prompt).content

    def _llm_query_batched(prompts) -> list:
        prompts = list(prompts)
        results = sub_llm.batch(prompts, config={"max_concurrency": BATCH_CONCURRENCY})
        return [r.content for r in results]

    def setup(state: RLMState) -> dict:
        depth = state.get("depth", 0)

        def _rlm_query(question: str, context: str) -> str:
            next_depth = depth + 1
            if next_depth > max_depth:
                return _llm_query(f"{question}\n\n{context}")
            child = graph.invoke(
                {"question": question, "context": context, "depth": next_depth},
                config={"recursion_limit": _recursion_limit},
            )
            return child.get("final_answer") or ""

        repl = REPL(state["context"], _llm_query, _llm_query_batched, _rlm_query)
        system = SYSTEM_PROMPT + "\n\n" + ORCHESTRATOR_ADDENDUM
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=build_metadata_message(state["question"], state["context"])),
        ]
        return {"repl": repl, "messages": messages, "iteration": 0}

    def call_model(state: RLMState) -> dict:
        turn = HumanMessage(content=build_turn_prompt(state["iteration"], max_iterations))
        response = root_llm.invoke(state["messages"] + [turn])
        return {"messages": [turn, response]}

    def execute_code(state: RLMState) -> dict:
        ai = state["messages"][-1]
        content = ai.content if isinstance(ai.content, str) else str(ai.content)
        blocks = parse_code_blocks(content)
        repl = state["repl"]
        if not blocks:
            feedback = "```repl``` 블록이 없습니다. 코드를 ```repl ... ``` 블록으로 작성하세요."
        else:
            outputs = []
            for block in blocks:
                out = repl.run(block)
                outputs.append(out if out else "(출력 없음)")
            feedback = _truncate("\n".join(outputs))
        return {
            "messages": [HumanMessage(content=feedback)],
            "iteration": state["iteration"] + 1,
            "final_answer": repl.final_answer,
        }

    def should_continue(state: RLMState):
        if state.get("final_answer") is not None:
            return END
        if state["iteration"] >= max_iterations:
            return END
        return "call_model"

    builder = StateGraph(RLMState)
    builder.add_node("setup", setup)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_code", execute_code)
    builder.add_edge(START, "setup")
    builder.add_edge("setup", "call_model")
    builder.add_edge("call_model", "execute_code")
    builder.add_conditional_edges("execute_code", should_continue, ["call_model", END])
    graph = builder.compile()
    return graph
