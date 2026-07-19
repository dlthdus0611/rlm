# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 개요

논문 "Recursive Language Models"(arXiv:2512.24601, 원본 repo: alexzhang13/rlm)의 핵심
메커니즘을 LangGraph로 재현한 **학습용 토이**. 거대 입력을 모델 컨텍스트에 직접 넣지 않고
Python REPL 변수(`context`)로만 두고, 모델이 코드를 생성·실행하며 답을 쌓아 올린다.

## 명령어

```bash
# 셋업 (Python 3.10+)
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 테스트 — 네트워크/API 키 불필요 (가짜 모델 사용)
pytest -v
pytest tests/test_graph.py -v                       # 단일 파일
pytest tests/test_graph.py::test_graph_single_turn_answer -v   # 단일 테스트

# 웹 플레이그라운드 — 문서 업로드 후 그 문서로 QA + 실시간 추론 트레이스 (실제 OpenRouter 호출)
streamlit run app/streamlit_app.py
```

설정(API 키·엔드포인트·기본 모델)은 환경변수로 오버라이드한다 — `OPENROUTER_API_KEY`(필수),
`OPENROUTER_BASE_URL`, `RLM_ROOT_MODEL`/`RLM_SUB_MODEL`(기본 `openai/gpt-5.5`·`openai/gpt-5.5-mini`).

## ⚠️ 보안 — 핵심 제약

`rlm/repl.py`는 **모델이 생성한 Python 코드를 in-process `exec()`로 그대로 실행**한다
(샌드박스 없음). 원본 repo의 docker/e2b/modal 샌드박스를 의도적으로 단순화한 것.
신뢰된 task·본인 머신 한정. 이 설계 결정을 약화시키거나 우회하지 말 것.

## 아키텍처

핵심은 `rlm/graph.py`의 `build_rlm_graph()`가 만드는 LangGraph StateGraph 제어 루프다.
세 노드가 순환한다: `setup → call_model → execute_code → (조건부) call_model | END`.

- **컨텍스트=핸들 (`graph.py` + `prompts.py`)**: 거대 입력은 `RLMState.context`에만 있고,
  모델에게 가는 메시지(`build_metadata_message`)에는 글자 수 등 메타데이터만 담는다.
  `test_graph_context_not_in_model_messages`가 본문 누출을 검증한다.
- **REPL 네임스페이스 (`rlm/repl.py`)**: 턴 간 상태가 유지되는 영속 dict 네임스페이스.
  모델에 노출되는 이름 — `context`, `llm_query`, `llm_query_batched`, `rlm_query`,
  `answer`, `SHOW_VARS`. `run()`은 stdout(또는 예외 traceback)만 문자열로 반환한다.
- **종료 신호 (`_AnswerDict`)**: 모델이 `answer["ready"] = True`를 세팅하면
  dict의 `__setitem__` 훅이 콜백을 쏴 `REPL.final_answer`를 캡처. `should_continue`가
  이를 보고 END로 분기. 외에 `iteration >= max_iterations`에서도 종료(미제출 시 final_answer=None).
- **재귀 sub-call (`setup` 내부 `_rlm_query`)**: `rlm_query(question, context)`는 같은
  `graph`를 `depth+1`로 invoke한다. `next_depth > max_depth`면 `llm_query`로 폴백.
  클로저로 `graph`를 참조하므로 `build_rlm_graph` 안에서 `graph = builder.compile()`이
  마지막에 와도 동작한다.
- **출력 축약 (`rlm/parsing.py`)**: REPL stdout은 `_truncate`로 8000자 제한(원본 ~20K 단순화).
  `parse_code_blocks`는 응답에서 ```repl ... ``` 블록만 정규식으로 추출한다.
- **재귀 한도**: LangGraph 기본 step 상한(25)을 넘기지 않도록 `recursion_limit`을
  `2 * max_iterations + 10`으로 잡는다. `api.run()`과 `graph.py` 양쪽에 같은 공식이 있다 —
  `max_iterations` 관련 변경 시 두 곳을 함께 맞출 것.

## LLM 경계와 테스트 전략

`graph.py`는 LLM을 `root_llm` / `sub_llm` 인자로만 받는다 — `.invoke()`/`.batch()`를
가진 Runnable이면 무엇이든 된다. 실제 모델 생성은 `rlm/llm.py`(`make_llm`,
OpenRouter 경유 `ChatOpenAI`)로 격리돼 있고, `rlm/api.py`의 `run()`이 이를 묶는 공개 진입점.

모델·엔드포인트·API 키는 `rlm/config.py`의 `Settings`(pydantic-settings)가 **호출 시점에**
환경변수에서 읽는다(`get_settings()`). import 시점이 아니라 호출 시점이라, demo의
`load_dotenv()`가 패키지 import 뒤에 와도 `.env` 값이 반영된다. `run()`의 모델 인자는
명시값 > 환경변수 > 기본값 우선순위.

따라서 **테스트는 네트워크 없이 가짜 모델로 그래프 전체를 돈다**:
`tests/test_graph.py`의 `FakeChat`(미리 정한 응답을 순서대로 반환)·`FakeSub`(고정 응답).
새 그래프 동작을 테스트할 때 이 패턴을 따를 것 — 실제 API를 호출하지 말 것.

## UI (`app/`)

UI는 `app/` 패키지(응용 계층, `eval/`과 대칭)에 모여 있고 코어 `rlm`을 소비한다.
streamlit은 스크립트 폴더만 sys.path에 넣으므로 `app/streamlit_app.py` 상단에서 프로젝트
루트를 sys.path에 추가한다 — 이게 없으면 `streamlit run`에서 `rlm`/`app` import가 깨진다.

- `app/streamlit_app.py` — 문서(텍스트)를 업로드하면 그 문서를 context로 두고 질문에 답하는
  QA 플레이그라운드. `graph.stream(stream_mode="updates")`로 턴별 생성 코드/REPL 출력을 실시간 렌더.
- `app/app_trace.py` — 스트림 업데이트를 화면용 `TraceEntry`로 바꾸는 **streamlit 비의존 순수 함수**.
  `tests/test_app_trace.py`로 검증(네트워크·streamlit 없이).

## 한글 프롬프트

`rlm/prompts.py`의 `SYSTEM_PROMPT` + `ORCHESTRATOR_ADDENDUM`이 모델 행동을 정의한다
(오케스트레이터로서 위임·검증·종료). 프롬프트는 한글이며, 원본 영문 프롬프트 대조는
`docs/prompts-reference.md` 참고. 설계 배경은 `docs/superpowers/specs/`,
`docs/superpowers/plans/`에 있다.

## 규약

- 커밋 메시지는 제목·본문 모두 한글 (전역 규칙). `Co-Authored-By` 트레일러는 유지.
- 코드 주석·docstring·프롬프트는 한글로 작성한다 (기존 코드 스타일).
