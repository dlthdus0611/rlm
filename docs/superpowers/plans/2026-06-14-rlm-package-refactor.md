# rlm 패키지 리팩터링 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `rlm_graph.py` 한 파일(220줄)에 섞여 있는 책임을 `rlm/` 파이썬 패키지의 책임별 모듈(parsing/repl/graph/llm/api)로 분리하고, 테스트를 `tests/`로 모듈별로 나눈다. 동작은 한 줄도 바꾸지 않는다.

**Architecture:** 순수 코드 이동 리팩터링. 새 모듈은 기존 `rlm_graph.py`/`prompts.py`의 내용을 책임별로 잘라 옮기고 내부 import 경로만 상대 import로 갱신한다. 단방향 의존(parsing/repl/prompts ← graph ← api ← __init__). 검증 기준은 "기존 27개 테스트가 단언 수정 없이 그대로 초록".

**Tech Stack:** Python 3.10+, LangGraph, langchain-core, langchain-openai, pytest.

---

## File Structure

신규 생성:
- `rlm/__init__.py` — 공개 표면 재노출
- `rlm/parsing.py` — 코드 블록 파싱 / 출력 축약 (stdlib만)
- `rlm/repl.py` — 코드 실행 샌드박스 (stdlib만)
- `rlm/llm.py` — OpenRouter 팩토리 (langchain_openai)
- `rlm/graph.py` — LangGraph 제어 루프
- `rlm/api.py` — 공개 진입점 `run()`
- `rlm/prompts.py` — 루트 `prompts.py` 이동 (내용 무수정)
- `tests/test_parsing.py`, `tests/test_repl.py`, `tests/test_graph.py`, `tests/test_prompts.py`, `tests/test_llm.py`
- `conftest.py` — 빈 파일, 레포 루트를 sys.path에 올림

수정:
- `demo_tickets.py:10` — import 경로

삭제:
- `rlm_graph.py`, `prompts.py`(루트), `test_rlm.py`(루트)

작업 순서는 매 커밋마다 `pytest`가 초록이도록 설계: ① 새 패키지 추가(구 코드 공존) → ② 테스트를 새 패키지로 이전 → ③ 구 코드 삭제 + 데모/README 갱신.

---

### Task 1: 리프 모듈 생성 (`parsing`, `repl`, `llm`, `prompts`)

내부 의존이 없는 모듈부터 만든다. 이 시점엔 구 `rlm_graph.py`/`prompts.py`가 그대로 남아 있어 기존 테스트는 계속 초록이다.

**Files:**
- Create: `rlm/__init__.py` (이 태스크에선 빈 파일)
- Create: `rlm/parsing.py`
- Create: `rlm/repl.py`
- Create: `rlm/llm.py`
- Create: `rlm/prompts.py`

- [ ] **Step 1: 빈 패키지 초기화 파일 생성**

`rlm/__init__.py` 를 빈 파일로 생성한다 (Task 2에서 재노출 내용을 채운다).

- [ ] **Step 2: `rlm/parsing.py` 생성**

```python
import re

CODE_BLOCK_RE = re.compile(r"```repl[ \t]*\n(.*?)```", re.DOTALL)


def parse_code_blocks(text: str) -> list[str]:
    """모델 응답에서 ```repl ... ``` 블록의 코드만 추출한다."""
    return [block.strip("\n") for block in CODE_BLOCK_RE.findall(text)]


MAX_OUTPUT_CHARS = 8000


def _truncate(s: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """REPL 출력이 한도를 넘으면 잘라내고 남은 글자 수를 표기한다."""
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[+{len(s) - limit} chars truncated]"
```

- [ ] **Step 3: `rlm/repl.py` 생성**

```python
import io
import traceback
from contextlib import redirect_stdout
from typing import Callable


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
```

- [ ] **Step 4: `rlm/llm.py` 생성**

중간에 끼어 있던 `from langchain_openai import ChatOpenAI`(구 `rlm_graph.py:189`)를 모듈 상단으로 정리한다.

```python
import os

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ROOT_MODEL = "openai/gpt-4o-mini"
DEFAULT_SUB_MODEL = "openai/gpt-4o-mini"


def make_llm(model: str) -> ChatOpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY 환경변수가 필요합니다.")
    return ChatOpenAI(model=model, base_url=OPENROUTER_BASE_URL, api_key=key)
```

- [ ] **Step 5: `rlm/prompts.py` 생성 (루트 `prompts.py` 내용 그대로 복사)**

루트 `prompts.py`의 전체 내용을 `rlm/prompts.py`로 복사한다. 내용은 한 글자도 바꾸지 않는다 (`SYSTEM_PROMPT`, `ORCHESTRATOR_ADDENDUM`, `build_metadata_message`, `build_turn_prompt`). 루트 `prompts.py`는 아직 삭제하지 않는다.

- [ ] **Step 6: 리프 모듈 import 검증**

Run: `python -c "from rlm.parsing import parse_code_blocks, _truncate, MAX_OUTPUT_CHARS; from rlm.repl import REPL; from rlm.llm import make_llm; from rlm.prompts import SYSTEM_PROMPT; print('ok')"`
Expected: `ok` 출력, 오류 없음

- [ ] **Step 7: 기존 테스트가 여전히 초록인지 확인**

Run: `pytest -q`
Expected: `27 passed` (구 `rlm_graph.py`/루트 `prompts.py` 경로 그대로라 영향 없음)

- [ ] **Step 8: 커밋**

```bash
git add rlm/__init__.py rlm/parsing.py rlm/repl.py rlm/llm.py rlm/prompts.py
git commit -m "rlm 패키지 리프 모듈 추가 (parsing/repl/llm/prompts)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `graph`, `api` 모듈 생성 + `__init__` 재노출 채우기

리프 모듈에 의존하는 상위 모듈을 만들고 공개 표면을 완성한다. 구 코드는 여전히 공존하므로 기존 테스트는 계속 초록.

**Files:**
- Create: `rlm/graph.py`
- Create: `rlm/api.py`
- Modify: `rlm/__init__.py` (빈 파일 → 재노출)

- [ ] **Step 1: `rlm/graph.py` 생성**

구 `rlm_graph.py`의 `RLMState`/`BATCH_CONCURRENCY`/`build_rlm_graph`를 본문 변경 없이 옮기고, import만 상대 import로 바꾼다. `parse_code_blocks`/`_truncate`는 `.parsing`, `REPL`은 `.repl`, 프롬프트는 `.prompts`에서 가져온다.

```python
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
```

- [ ] **Step 2: `rlm/api.py` 생성**

구 `rlm_graph.py`의 `run()`을 옮기고, `build_rlm_graph`는 `.graph`, `make_llm`/기본 모델 상수는 `.llm`에서 가져온다.

```python
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
```

- [ ] **Step 3: `rlm/__init__.py` 를 재노출로 채우기**

빈 파일을 다음 내용으로 덮어쓴다:

```python
from .api import run
from .graph import build_rlm_graph
from .llm import make_llm
from .repl import REPL
from .parsing import parse_code_blocks

__all__ = ["run", "build_rlm_graph", "make_llm", "REPL", "parse_code_blocks"]
```

- [ ] **Step 4: 공개 표면 import 검증**

Run: `python -c "import rlm; from rlm import run, build_rlm_graph, make_llm, REPL, parse_code_blocks; print('ok')"`
Expected: `ok` 출력, 오류 없음

- [ ] **Step 5: 기존 테스트가 여전히 초록인지 확인**

Run: `pytest -q`
Expected: `27 passed`

- [ ] **Step 6: 커밋**

```bash
git add rlm/graph.py rlm/api.py rlm/__init__.py
git commit -m "rlm 패키지 graph/api 모듈 + 공개 표면 추가

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: 테스트를 `tests/`로 모듈별 분리 + `conftest.py` 추가

테스트가 새 `rlm.*` 경로를 쓰도록 옮긴다. 단언/로직은 한 줄도 바꾸지 않는다 — 모듈별로 나누고 import만 갱신.

**Files:**
- Create: `conftest.py` (루트, 빈 파일)
- Create: `tests/test_parsing.py`
- Create: `tests/test_repl.py`
- Create: `tests/test_graph.py`
- Create: `tests/test_prompts.py`
- Create: `tests/test_llm.py`
- Delete: `test_rlm.py` (루트)

- [ ] **Step 1: 루트 `conftest.py` 생성 (빈 파일)**

빈 `conftest.py`를 레포 루트에 만든다. pytest(prepend 임포트 모드)가 이 파일이 있는 디렉터리(=레포 루트)를 sys.path에 올려 `tests/`에서 `import rlm`이 가능해진다.

- [ ] **Step 2: `tests/test_parsing.py` 생성**

```python
from rlm.parsing import parse_code_blocks, _truncate, MAX_OUTPUT_CHARS


def test_parse_no_block():
    assert parse_code_blocks("그냥 텍스트, 코드 없음") == []


def test_parse_single_block():
    text = "계획을 세웁니다.\n```repl\nprint(1)\n```\n끝."
    assert parse_code_blocks(text) == ["print(1)"]


def test_parse_multiple_blocks():
    text = "```repl\na = 1\n```\n사이 텍스트\n```repl\nprint(a)\n```"
    assert parse_code_blocks(text) == ["a = 1", "print(a)"]


def test_parse_ignores_other_languages():
    text = "```python\nprint('x')\n```\n```repl\nprint('y')\n```"
    assert parse_code_blocks(text) == ["print('y')"]


def test_truncate_short_passthrough():
    assert _truncate("짧은 문자열") == "짧은 문자열"


def test_truncate_long_marks_remainder():
    s = "a" * (MAX_OUTPUT_CHARS + 100)
    out = _truncate(s)
    assert out.startswith("a" * MAX_OUTPUT_CHARS)
    assert "+100 chars" in out
    assert len(out) < len(s)
```

- [ ] **Step 3: `tests/test_repl.py` 생성**

```python
from rlm.repl import REPL


def _make_repl(context="컨텍스트"):
    return REPL(
        context=context,
        llm_query=lambda p: f"LLM:{p}",
        llm_query_batched=lambda ps: [f"LLM:{p}" for p in ps],
        rlm_query=lambda q, c: f"RLM:{q}",
    )


def test_repl_captures_stdout():
    repl = _make_repl()
    out = repl.run("print('안녕')")
    assert "안녕" in out


def test_repl_context_available():
    repl = _make_repl(context="비밀값")
    out = repl.run("print(context)")
    assert "비밀값" in out


def test_repl_exception_becomes_traceback():
    repl = _make_repl()
    out = repl.run("raise ValueError('펑')")
    assert "ValueError" in out and "펑" in out
    assert "ok" in repl.run("print('ok')")


def test_repl_persists_variables_across_runs():
    repl = _make_repl()
    repl.run("x = 41")
    out = repl.run("print(x + 1)")
    assert "42" in out


def test_repl_answer_ready_sets_final_answer():
    repl = _make_repl()
    assert repl.final_answer is None
    repl.run('answer["content"] = "최종 답"\nanswer["ready"] = True')
    assert repl.final_answer == "최종 답"


def test_repl_answer_not_ready_keeps_none():
    repl = _make_repl()
    repl.run('answer["content"] = "아직"')
    assert repl.final_answer is None


def test_repl_llm_query_injected():
    repl = _make_repl()
    out = repl.run("print(llm_query('질문'))")
    assert "LLM:질문" in out


def test_repl_show_vars():
    repl = _make_repl()
    repl.run("myvar = 1")
    out = repl.run("print(SHOW_VARS())")
    assert "myvar" in out
    assert "context" not in out
```

- [ ] **Step 4: `tests/test_prompts.py` 생성**

```python
from rlm.prompts import (
    SYSTEM_PROMPT, ORCHESTRATOR_ADDENDUM,
    build_metadata_message, build_turn_prompt,
)


def test_system_prompt_mentions_key_tokens():
    for token in ["context", "llm_query", "rlm_query", "answer", "repl"]:
        assert token in SYSTEM_PROMPT


def test_metadata_message_includes_question_and_length():
    msg = build_metadata_message("환불 몇 건?", "x" * 1234)
    assert "환불 몇 건?" in msg
    assert "1234" in msg


def test_turn_prompt_iteration_zero_has_safeguard():
    msg = build_turn_prompt(0, 10)
    assert "턴 1/10" in msg
    assert "먼저 context" in msg


def test_turn_prompt_later_iteration_is_short():
    msg = build_turn_prompt(3, 10)
    assert "턴 4/10" in msg
    assert "먼저 context" not in msg
```

- [ ] **Step 5: `tests/test_graph.py` 생성**

`FakeChat`/`FakeSub`는 그래프 테스트에서만 쓰이므로 이 파일 안에 둔다.

```python
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
```

- [ ] **Step 6: `tests/test_llm.py` 생성**

```python
import pytest

from rlm.llm import make_llm


def test_make_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_llm("openai/gpt-4o-mini")
```

- [ ] **Step 7: 루트 `test_rlm.py` 삭제**

```bash
rm test_rlm.py
```

- [ ] **Step 8: 전체 테스트 초록 확인 (새 경로)**

Run: `pytest -q`
Expected: `27 passed` (5개 파일 합계: parsing 6 + repl 8 + prompts 4 + graph 8 + llm 1)

- [ ] **Step 9: 커밋**

```bash
git add conftest.py tests/ && git rm test_rlm.py
git commit -m "테스트를 tests/로 모듈별 분리 + 루트 conftest 추가

parsing/repl/graph/prompts/llm 테스트를 rlm.* 경로로 이전. 단언 무수정.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 구 코드 삭제 + 데모/README 갱신

이제 새 패키지만 사용하므로 구 `rlm_graph.py`/루트 `prompts.py`를 지우고, 데모와 README를 새 경로로 맞춘다.

**Files:**
- Delete: `rlm_graph.py`
- Delete: `prompts.py` (루트)
- Modify: `demo_tickets.py:10`
- Modify: `README.md` (구조 섹션)

- [ ] **Step 1: `demo_tickets.py` import 경로 갱신**

`demo_tickets.py:10` 의 import를 바꾼다:

```python
from rlm import run
```

(기존 `from rlm_graph import run` 한 줄만 교체. 나머지 줄은 손대지 않는다.)

- [ ] **Step 2: 구 모듈 삭제**

```bash
git rm rlm_graph.py prompts.py
```

- [ ] **Step 3: 데모가 import 가능한지 검증 (API 호출 없음)**

Run: `python -c "import demo_tickets; print('ok')"`
Expected: `ok` 출력, 오류 없음 (모듈 로드 시 `load_dotenv()`만 실행되고 네트워크 호출은 `main()` 안에 있어 import만으로는 호출되지 않음)

- [ ] **Step 4: README "구조" 섹션 갱신**

`README.md`의 `## 구조` 섹션(현재 `- rlm_graph.py ...` 목록)을 아래로 교체한다:

```markdown
## 구조
- `rlm/` — 패키지
  - `parsing.py` — 코드 블록 파싱 / 출력 축약
  - `repl.py` — 코드 실행 샌드박스(REPL)
  - `graph.py` — LangGraph 제어 루프
  - `llm.py` — OpenRouter LLM 팩토리
  - `api.py` — 공개 진입점 `run()`
  - `prompts.py` — 한글 프롬프트
- `tests/` — 모듈별 단위/그래프 테스트
- `demo_tickets.py` — 데모
- `docs/` — 설계 spec, 프롬프트 참고, 본 계획
```

- [ ] **Step 5: 전체 테스트 + 공개 표면 최종 검증**

Run: `pytest -q && python -c "import rlm; from rlm import run, build_rlm_graph, make_llm, REPL, parse_code_blocks; print('import ok')"`
Expected: `27 passed`, 이어서 `import ok`

- [ ] **Step 6: 구 경로가 완전히 사라졌는지 확인**

Run: `grep -rn "rlm_graph" --include=*.py . ; grep -rn "^from prompts\|^import prompts" --include=*.py .`
Expected: 출력 없음 (구 모듈 참조 0)

- [ ] **Step 7: 커밋**

```bash
git add demo_tickets.py README.md && git rm rlm_graph.py prompts.py
git commit -m "구 rlm_graph.py/prompts.py 삭제, 데모·README를 rlm 패키지로 갱신

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## 검증 요약 (동작 불변 증명)

- 매 태스크 커밋 시점에 `pytest -q` 가 `27 passed` (Task 1~2는 구 경로, Task 3~4는 새 경로).
- 테스트 단언은 한 줄도 바뀌지 않았으므로, 통과 = 코드 이동이 동작을 보존했다는 증거.
- `python -c "import rlm; ..."` 로 공개 표면 import 무오류.
- `grep` 으로 구 모듈 참조 0 확인.
- (선택, 키 필요) `python demo_tickets.py` 수동 스모크.
