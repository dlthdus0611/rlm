# Streamlit RLM 플레이그라운드 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 임의의 context와 질문을 받아 `graph.stream`으로 RLM의 턴별 추론(생성 코드 + REPL 출력)을 실시간 표시하는 Streamlit 플레이그라운드를 만든다. 기존 티켓 데이터는 원클릭 샘플.

**Architecture:** UI 레이어(`streamlit_app.py`)와 순수 트레이스 포맷터(`app_trace.py`)를 분리한다. `app_trace`는 `graph.stream(stream_mode="updates")` 업데이트 dict를 화면용 `TraceEntry`로 변환(streamlit 비의존, 단위 테스트 가능). UI는 기존 `rlm` 패키지와 `demo_tickets.make_tickets`를 호출만 한다. 그래프/REPL 동작 변경 없음.

**Tech Stack:** Python 3.10+, Streamlit, LangGraph, langchain-core, pytest. 패키지 venv는 `.venv/`.

---

## File Structure

신규 생성:
- `app_trace.py` (루트) — `TraceEntry` 데이터클래스 + `format_update()` 순수 함수. streamlit 비의존.
- `streamlit_app.py` (루트) — Streamlit UI. `main()`을 `if __name__ == "__main__"` 가드로 호출(= `streamlit run`에서 실행, 평범한 import 시엔 미실행 → import 스모크 가능).
- `tests/test_app_trace.py` — `format_update` 단위 테스트(네트워크/streamlit 없음).

수정:
- `requirements.txt` — `streamlit>=1.30` 추가.
- `README.md` — "데모" 섹션에 `streamlit run streamlit_app.py` 추가.

재사용: `rlm.build_rlm_graph`, `rlm.make_llm`, `rlm.config.get_settings`, `rlm.parsing.parse_code_blocks`, `demo_tickets.make_tickets`.

---

### Task 1: `app_trace.py` 순수 트레이스 포맷터 (TDD)

**Files:**
- Create: `app_trace.py`
- Test: `tests/test_app_trace.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_app_trace.py`:
```python
from langchain_core.messages import AIMessage, HumanMessage

from app_trace import format_update, TraceEntry


def test_call_model_update_extracts_code_and_increments_turn():
    ai = AIMessage(content='계획을 세웁니다.\n```repl\nprint(1)\n```')
    update = {"call_model": {"messages": [HumanMessage(content="턴 1/12:"), ai]}}
    entries, turn, final = format_update(update, 0)
    assert turn == 1
    assert final is None
    assert len(entries) == 1
    e = entries[0]
    assert e.kind == "model"
    assert e.turn == 1
    assert e.code_blocks == ["print(1)"]
    assert "계획" in e.text


def test_execute_code_update_without_final():
    update = {"execute_code": {
        "messages": [HumanMessage(content="1\n")],
        "iteration": 1, "final_answer": None,
    }}
    entries, turn, final = format_update(update, 1)
    assert final is None
    assert turn == 1
    assert len(entries) == 1
    assert entries[0].kind == "exec"
    assert "1" in entries[0].text


def test_execute_code_update_with_final():
    update = {"execute_code": {
        "messages": [HumanMessage(content="(출력 없음)")],
        "iteration": 2, "final_answer": "정답 42",
    }}
    entries, turn, final = format_update(update, 1)
    assert final == "정답 42"
    assert entries[0].kind == "exec"


def test_unknown_node_yields_no_entries():
    update = {"setup": {"messages": [], "iteration": 0}}
    entries, turn, final = format_update(update, 3)
    assert entries == []
    assert turn == 3
    assert final is None
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_app_trace.py -q`
Expected: collection/ImportError — `No module named 'app_trace'`.

- [ ] **Step 3: 최소 구현 작성**

`app_trace.py`:
```python
"""graph.stream(stream_mode="updates") 업데이트를 화면용 트레이스 항목으로 변환.

streamlit에 의존하지 않는 순수 모듈 — 단위 테스트가 쉽다.
"""
from dataclasses import dataclass, field

from rlm.parsing import parse_code_blocks


@dataclass
class TraceEntry:
    kind: str                       # "model" | "exec"
    turn: int                       # 1-based 턴 번호
    text: str                       # 모델 산문/코드 또는 REPL 출력
    code_blocks: list = field(default_factory=list)  # kind=="model"의 ```repl``` 코드


def _content_str(msg) -> str:
    content = getattr(msg, "content", "")
    return content if isinstance(content, str) else str(content)


def format_update(update: dict, turn: int):
    """업데이트 dict 하나를 (entries, new_turn, final_answer|None)로 변환한다.

    update 예:
      {"call_model": {"messages": [turn_msg, ai_msg]}}
      {"execute_code": {"messages": [human_msg], "iteration": k, "final_answer": ...}}
    그 외 노드(setup 등)는 빈 목록을 낸다.
    """
    entries: list[TraceEntry] = []
    final_answer = None
    for node, payload in update.items():
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        last = messages[-1] if messages else None
        if node == "call_model":
            turn += 1
            text = _content_str(last) if last is not None else ""
            entries.append(TraceEntry("model", turn, text, parse_code_blocks(text)))
        elif node == "execute_code":
            text = _content_str(last) if last is not None else ""
            entries.append(TraceEntry("exec", turn, text))
            if isinstance(payload, dict) and payload.get("final_answer") is not None:
                final_answer = payload["final_answer"]
    return entries, turn, final_answer
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_app_trace.py -q`
Expected: `4 passed`.

- [ ] **Step 5: 전체 스위트 회귀 확인**

Run: `.venv/bin/python -m pytest -q`
Expected: 기존 + 신규 통과(`35 passed` — 기존 31 + app_trace 4).

- [ ] **Step 6: 커밋**

```bash
git add app_trace.py tests/test_app_trace.py
git commit -m "graph.stream 업데이트를 트레이스 항목으로 바꾸는 app_trace 추가

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `streamlit_app.py` UI + streamlit 의존성

**Files:**
- Modify: `requirements.txt`
- Create: `streamlit_app.py`

- [ ] **Step 1: streamlit 의존성 추가 + 설치**

`requirements.txt`에 한 줄 추가(`pytest` 줄 위, 알파벳/논리 순서 무방):
```
streamlit>=1.30
```
그다음 설치:

Run: `.venv/bin/pip install "streamlit>=1.30"`
Expected: 설치 성공(이미 있으면 "already satisfied").

- [ ] **Step 2: `streamlit_app.py` 작성**

```python
"""RLM 플레이그라운드 — 임의의 context와 질문을 넣고 RLM이 푸는 과정을 실시간으로 본다.

실행: streamlit run streamlit_app.py
⚠️ 모델이 생성한 Python 코드를 in-process exec()로 실행한다(샌드박스 없음). 신뢰 환경 한정.
"""
import streamlit as st
from dotenv import load_dotenv

from app_trace import format_update
from demo_tickets import make_tickets
from rlm import build_rlm_graph, make_llm
from rlm.config import get_settings

load_dotenv()

TICKET_QUESTION = (
    "context에는 여러 개의 고객 지원 티켓이 '=== 티켓 #N ===' 로 구분되어 있습니다. "
    "이 주문에 대해 실제로 '환불을 요청'한 티켓 수와, 그 환불 요청들 중 사유가 "
    "'배송 지연'(늦은 배송)인 티켓 수를 구하세요. "
    "환불 정책 문의·교환 요청처럼 환불을 요청하지 않은 단순 언급은 세지 마세요. "
    "표현이 제각각이니('돈 돌려주세요', '결제 취소' 등도 환불 요청) 의미로 판단하세요. "
    "최종 답은 정확히 '환불=<수>, 배송지연=<수>' 형식으로만 제출하세요."
)


def _api_key_present() -> bool:
    return bool(get_settings().openrouter_api_key)


def _run_rlm(context, question, root_model, sub_model, max_depth, max_iterations):
    try:
        graph = build_rlm_graph(
            make_llm(root_model), make_llm(sub_model), max_depth, max_iterations
        )
    except RuntimeError as e:
        st.error(str(e))
        return

    st.subheader("추론 트레이스")
    turn = 0
    final_answer = None
    try:
        for update in graph.stream(
            {"question": question, "context": context, "depth": 0},
            config={"recursion_limit": 2 * max_iterations + 10},
            stream_mode="updates",
        ):
            entries, turn, maybe_final = format_update(update, turn)
            for e in entries:
                if e.kind == "model":
                    with st.chat_message("assistant"):
                        st.markdown(f"**🧠 턴 {e.turn}**")
                        if e.text:
                            st.markdown(e.text)
                        for block in e.code_blocks:
                            st.code(block, language="python")
                elif e.kind == "exec":
                    with st.chat_message("user"):
                        st.markdown(f"**💻 턴 {e.turn} 실행 결과**")
                        st.code(e.text or "(출력 없음)")
            if maybe_final is not None:
                final_answer = maybe_final
    except Exception as e:  # noqa: BLE001 - 데모이므로 오류를 화면에 그대로 노출
        st.exception(e)
        return

    st.subheader("결과")
    if final_answer is None:
        st.info("최종 답이 제출되지 않았습니다(턴 소진). max_iterations를 늘려보세요.")
        return
    st.success(f"모델 답: {final_answer}")

    expected = st.session_state.get("sample_expected")
    if expected is not None and question == TICKET_QUESTION:
        refund, delay = expected
        normalized = final_answer.replace(" ", "")
        ok = f"환불={refund}" in normalized and f"배송지연={delay}" in normalized
        st.write(f"정답: 환불={refund}, 배송지연={delay} — {'✅ 일치' if ok else '❌ 불일치'}")


def main():
    st.set_page_config(page_title="RLM 플레이그라운드", page_icon="🧠", layout="wide")
    st.title("🧠 Recursive Language Model 플레이그라운드")
    st.caption(
        "임의의 긴 context와 질문을 넣으면, 모델이 context를 REPL 변수로만 두고 "
        "코드를 생성·실행하며 답을 쌓아 올립니다. 그 과정을 턴별로 보여줍니다."
    )
    st.warning(
        "모델이 생성한 Python 코드를 in-process `exec()`로 실행합니다(샌드박스 없음). "
        "신뢰할 수 있는 입력·본인 머신에서만 사용하세요.",
        icon="⚠️",
    )

    st.session_state.setdefault("context", "")
    st.session_state.setdefault("question", "")
    st.session_state.setdefault("sample_expected", None)

    with st.sidebar:
        st.header("설정")
        max_iterations = st.slider("max_iterations", 4, 20, 12)
        max_depth = st.slider("max_depth", 0, 2, 1)
        settings = get_settings()
        root_model = st.text_input("root 모델", settings.rlm_root_model)
        sub_model = st.text_input("sub 모델", settings.rlm_sub_model)
        if _api_key_present():
            st.success("OPENROUTER_API_KEY 감지됨")
        else:
            st.error("OPENROUTER_API_KEY 없음 — .env 또는 환경변수에 설정하세요.")

    with st.expander("샘플 데이터 채우기", expanded=False):
        col1, col2 = st.columns(2)
        n = col1.number_input("티켓 수", 10, 120, 60)
        seed = col2.number_input("seed", 0, 9999, 42)
        if st.button("지저분한 티켓으로 채우기"):
            ctx, refund, delay = make_tickets(int(n), int(seed))
            st.session_state.context = ctx
            st.session_state.question = TICKET_QUESTION
            st.session_state.sample_expected = (refund, delay)

    uploaded = st.file_uploader(
        "context 파일 업로드(텍스트)", type=["txt", "md", "csv", "json", "log"]
    )
    if uploaded is not None:
        st.session_state.context = uploaded.getvalue().decode("utf-8", errors="replace")
        st.session_state.sample_expected = None

    context = st.text_area("context", key="context", height=240)
    st.caption(
        f"context 길이: {len(context)}자 — 모델에게는 이 길이 등 메타데이터만 전달됩니다."
    )
    question = st.text_area("질문(question)", key="question", height=100)

    run_clicked = st.button("▶ RLM 실행", type="primary", disabled=not _api_key_present())
    if run_clicked:
        if not context.strip() or not question.strip():
            st.warning("context와 질문을 모두 입력하세요.")
            return
        _run_rlm(context, question, root_model, sub_model, max_depth, max_iterations)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 구문/임포트 스모크**

Run: `.venv/bin/python -m py_compile streamlit_app.py && .venv/bin/python -c "import streamlit_app; print('import ok')"`
Expected: `import ok`. (`main()`은 `__name__=="__main__"` 가드라 import 시 실행되지 않음 — Streamlit 런타임 경고가 떠선 안 됨.)

- [ ] **Step 4: 전체 스위트 회귀 확인**

Run: `.venv/bin/python -m pytest -q`
Expected: `35 passed`(streamlit 추가가 기존 테스트에 영향 없음).

- [ ] **Step 5: (수동) Streamlit 스모크 — 선택**

Run(사람이 직접, API 키 필요): `.venv/bin/python -m streamlit run streamlit_app.py`
브라우저에서: "지저분한 티켓으로 채우기" → "▶ RLM 실행" → 턴별 코드/출력 트레이스와 ✅ 일치 확인.
(자동 검증 불가 — 컨트롤러가 실행 여부를 사용자에게 안내.)

- [ ] **Step 6: 커밋**

```bash
git add requirements.txt streamlit_app.py
git commit -m "Streamlit RLM 플레이그라운드 앱 추가

임의 context+질문을 graph.stream으로 실행해 턴별 추론을 실시간 표시.
티켓 데이터는 원클릭 샘플. requirements에 streamlit 추가.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: README 실행법 추가

**Files:**
- Modify: `README.md`

- [ ] **Step 1: "데모" 섹션 갱신**

먼저 현재 데모 섹션을 확인: `sed -n '/## 데모/,/^## /p' README.md`.
현재는 `## 데모 (실제 API 필요)` 제목 아래 bash 코드블록에 `python demo_tickets.py` 한 줄만 있다.

그 bash 코드블록 **안의 내용**을 아래 4줄로 바꾼다(코드블록 펜스 ```` ```bash ```` / ```` ``` ````는 그대로 두고 내부만 교체):
```
# CLI: 지저분한 티켓 전수 집계 + 채점
python demo_tickets.py

# 웹 UI: 임의 context/질문으로 RLM 실행 + 추론 과정 실시간 표시
streamlit run streamlit_app.py
```
그리고 그 코드블록 바로 아래 줄에 설명 한 줄을 추가한다(일반 텍스트, 인용/펜스 아님):
`임의 텍스트를 붙여넣거나 파일을 올려 쓸 수 있고, "지저분한 티켓" 샘플도 원클릭으로 채울 수 있습니다.`

- [ ] **Step 2: 구조 섹션에 새 파일 반영(있으면)**

`README.md`의 `## 구조` 목록에 다음 두 줄을 `demo_tickets.py` 항목 아래 추가:
```markdown
- `streamlit_app.py` — 웹 플레이그라운드(임의 입력 + 실시간 추론 트레이스)
- `app_trace.py` — 스트리밍 업데이트를 트레이스 항목으로 변환(순수)
```

- [ ] **Step 3: 렌더 확인**

Run: `sed -n '/## 데모/,/## /p' README.md`
Expected: 두 명령(`python demo_tickets.py`와 `streamlit run streamlit_app.py`)이 모두 보임.

- [ ] **Step 4: 커밋**

```bash
git add README.md
git commit -m "README: Streamlit 플레이그라운드 실행법·구조 추가

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## 검증 요약

- `tests/test_app_trace.py` 4개 + 기존 31개 = **35 passed**(네트워크 없음).
- `python -c "import streamlit_app"` 무오류(가드 덕에 UI 미실행).
- 수동: `streamlit run streamlit_app.py`로 티켓 샘플 → 실행 → 트레이스/✅ 확인, 임의 텍스트로도 동작.
- 그래프/REPL/기존 동작 불변.
