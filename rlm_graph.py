import io
import os
import re
import traceback
from contextlib import redirect_stdout
from typing import Callable, Optional

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage

from prompts import (
    SYSTEM_PROMPT, ORCHESTRATOR_ADDENDUM,
    build_metadata_message, build_turn_prompt,
)

CODE_BLOCK_RE = re.compile(r"```repl[ \t]*\n(.*?)```", re.DOTALL)


def parse_code_blocks(text: str) -> list[str]:
    """모델 응답에서 ```repl ... ``` 블록의 코드만 추출한다."""
    return [block.strip("\n") for block in CODE_BLOCK_RE.findall(text)]


class _AnswerDict(dict):
    """answer["ready"] = True 가 설정되면 콜백으로 content를 캡처하는 dict."""

    def __init__(self, on_ready: Callable[[str], None]):
        super().__init__()
        super().__setitem__("content", "")
        super().__setitem__("ready", False)
        self._on_ready = on_ready

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "ready" and value:
            self._on_ready(self.get("content", ""))


_RESERVED = {
    "context", "llm_query", "llm_query_batched", "rlm_query",
    "answer", "SHOW_VARS", "__builtins__",
}


class REPL:
    """모델이 생성한 코드를 실행하는 영속 Python 네임스페이스.

    ⚠️ in-process exec(). 신뢰된 입력에만 사용.
    """

    def __init__(self, context, llm_query, llm_query_batched, rlm_query):
        self.final_answer: str | None = None
        self.ns: dict = {
            "context": context,
            "llm_query": llm_query,
            "llm_query_batched": llm_query_batched,
            "rlm_query": rlm_query,
            "answer": _AnswerDict(self._capture_answer),
            "SHOW_VARS": self._show_vars,
        }

    def _capture_answer(self, content) -> None:
        self.final_answer = str(content)

    def _show_vars(self) -> str:
        names = [
            k for k in self.ns
            if not k.startswith("_") and k not in _RESERVED
        ]
        return ", ".join(sorted(names))

    def run(self, code: str) -> str:
        """code를 실행하고 stdout(또는 예외 traceback)을 문자열로 반환."""
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, self.ns)
        except Exception:
            buf.write(traceback.format_exc())
        return buf.getvalue()


MAX_OUTPUT_CHARS = 8000


def _truncate(s: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """REPL 출력이 한도를 넘으면 잘라내고 남은 글자 수를 표기한다."""
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[+{len(s) - limit} chars truncated]"


BATCH_CONCURRENCY = 8


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
    # 한 번의 invoke 동안 실행되는 노드 스텝 수 상한(LangGraph 기본값 25를 넘기지 않도록).
    # setup(1) + max_iterations*(call_model + execute_code) + 여유.
    _recursion_limit = 2 * max_iterations + 10

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


from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ROOT_MODEL = "openai/gpt-4o-mini"
DEFAULT_SUB_MODEL = "openai/gpt-4o-mini"


def make_llm(model: str) -> ChatOpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY 환경변수가 필요합니다.")
    return ChatOpenAI(model=model, base_url=OPENROUTER_BASE_URL, api_key=key)


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
