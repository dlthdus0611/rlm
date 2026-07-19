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

# RLM vs RAG 비교 — 같은 문서·테스트셋·judge로 두 시스템을 정확도·비용·지연 3축 대조
#   (실제 OpenRouter + RAG는 OpenAI 임베딩 호출. OPENROUTER_API_KEY·OPENAI_API_KEY 필요)
python -m eval.compare --systems rlm,rag --set single --n 10
python -m eval.compare --systems rlm,rag --set cross --n 5 --question-field question_textbook

# 웹 플레이그라운드 — 문서 업로드 후 그 문서로 QA + 실시간 추론 트레이스 (실제 OpenRouter 호출)
streamlit run app/playground.py

# 디버거로 띄우기 — VS Code는 .vscode/launch.json의 "Streamlit: 앱 디버그" 실행,
# 터미널/기타 디버거는 아래 진입점 사용 (playground.py·rlm/ 중단점이 잡힌다)
python -m app
```

설정(API 키·엔드포인트·기본 모델)은 환경변수로 오버라이드한다 — `OPENROUTER_API_KEY`(필수),
`OPENROUTER_BASE_URL`, `RLM_ROOT_MODEL`/`RLM_SUB_MODEL`(기본 `openai/gpt-5.6-sol`·`openai/gpt-5.6-luna`).
RAG 비교를 쓸 때는 `OPENAI_API_KEY`(검색 임베딩용)와 `RAG_*` 파라미터도 환경변수로 읽는다.

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
  `2 * max_iterations + 10`으로 잡는다. 이 공식은 `rlm/graph.py`의 `recursion_limit_for()`
  **한 곳**에만 있고 `api.run()`·`eval/harness.py`·`app/eval_run.py`·`app/playground.py`가
  모두 이를 호출한다 — 상한을 바꾸려면 이 함수만 고치면 된다.

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
streamlit은 스크립트 폴더만 sys.path에 넣으므로 `app/playground.py` 상단에서 프로젝트
루트를 sys.path에 추가한다 — 이게 없으면 `streamlit run`에서 `rlm`/`app` import가 깨진다.

- `app/playground.py` — 문서(텍스트)를 업로드하면 그 문서를 context로 두고 질문에 답하는
  QA 플레이그라운드(멀티페이지 진입점). `graph.stream(stream_mode="updates")`로 턴별 생성 코드/REPL 출력을 실시간 렌더.
- `app/trace.py` — 스트림 업데이트를 화면용 `TraceEntry`로 바꾸는 **streamlit 비의존 순수 함수**.
  `tests/test_trace.py`로 검증(네트워크·streamlit 없이).
- `app/ui.py` — 공용 UI 디자인 시스템(그라디언트 히어로·스탯 카드·판정 배지·트레이스 렌더러).
  두 페이지가 같은 룩을 쓰게 하고, 시스템 다크/라이트를 자동 추종한다(`.streamlit/config.toml` 테마).
- `app/pages/1_평가.py` — 멀티페이지 **평가 페이지**. `data/`의 QA 테스트셋을 테스트셋·난이도·
  문항 수로 골라 라이브로 실행하면 진행 표시와 함께 집계표·문항별 드릴다운(판정+추론 트레이스)을
  렌더하고 결과를 JSON으로 내려받는다. streamlit이 `app/pages/`를 자동 인식하며, 여기서도 상단에서
  루트를 sys.path에 추가한다.
- `app/eval_run.py` — 문항별로 그래프를 stream하며 트레이스를 뽑고 `harness.judge`로 채점해
  `EvalEvent`(`trace`/`item_done`/`run_done`)를 yield하는 **streamlit 비의존 순수 오케스트레이션**
  (`trace`와 대칭). `to_payload`는 `eval/runner.py`와 동일한 다운로드 구조를 만든다.
  `tests/test_eval_run.py`로 가짜 모델(FakeChat/FakeSub/FakeJudge) 검증(네트워크 없이).
- `app/__main__.py` — `python -m app` 진입점. streamlit CLI 대신 파이썬 진입점으로
  streamlit을 in-process(`streamlit.web.cli`)로 띄워 디버거를 붙일 수 있게 한다.
  `.vscode/launch.json`이 이 방식(및 `module: streamlit` 표준 방식)을 구성으로 제공.

## RLM vs RAG 비교 벤치

프로젝트 목표(RLM이 기본 RAG보다 실제 효과가 있는지 실증)의 핵심으로, 두 시스템을 **바꿔 끼우는
플러그인 구조** 위에서 붙인다. **공정성 계약**: 같은 문서(`data/samsung_2023.txt`)·같은 문항
(`select_items` seed 고정)·같은 judge·같은 생성 모델(RAG 생성기 = RLM `root_model`)을 고정하고,
**context 접근 전략만** 다르게 둔다 — RLM은 코드로 전체 문서를 훑고, RAG는 top-k passage만 넣는다.
정확도·비용(토큰)·지연(시간)에 더해 RAG는 근거적중률(검색 passage가 gold `evidence` 포함 여부)을 잰다.

- **공통 플러그인 (`eval/systems.py`)**: `BaseSolver`가 공통 껍데기 — (질문, context)+주입 LLM →
  계측된 답(토큰·지연·트레이스) — 를 소유한다(`solve()`가 타이머 + `UsageMetadataCallbackHandler`
  성격의 `_UsageCollector`를 부착). 내부 흐름만 `_run`으로 갈린다: `RlmSolver`(그래프 stream 루프)·
  `RagSolver`(선형 파이프라인, context 해시로 리트리버 메모이즈). `SYSTEMS` 레지스트리 +
  `build_solvers()`로 CLI·UI가 시스템을 모른 채 고른다. 3번째(long-context·BM25 등)는 상속+등록 한 줄.
- **RAG 코어 (`rag/`, `rlm/`과 대칭)**: `embeddings.py`(OpenAI 임베딩 팩토리)·`index.py`(청킹+FAISS,
  내용·파라미터 해시 디스크 캐시)·`retriever.py`(HyDE→벡터 MMR→LLM 리랭킹, 단계 토글 — 전부 끄면
  표준 top-k 파생)·`pipeline.py`(검색→단일 생성)·`prompts.py`·`api.py`(공개 진입점, `rlm/api`와 대칭).
  LLM·임베딩·리트리버는 주입이라 테스트는 `FakeEmbedder`/`FakeStore`/`FakeChat`으로 네트워크 없이 돈다.
- **비교 하니스 (`eval/compare.py`)**: `run_item`이 한 문항을 등록 solver 전부에 통과시켜 동일 judge로
  채점, `aggregate_compare`가 시스템×난이도로 집계, `to_compare_payload`가 다운로드 스키마를 만든다.
  `python -m eval.compare`가 CLI. `recursion_limit_for` 등 기존 harness 로직(`judge`·`select_items`)을 재사용.
- **비교 UI (`app/compare_run.py` + `app/pages/2_비교.py`)**: `app/eval_run.py`와 대칭인 streamlit
  비의존 오케스트레이션이 `CompareEvent`를 yield하고, 비교 페이지가 좌우 대조표 + 문항 드릴다운
  (RLM 추론 트레이스 ↔ RAG 검색 passage)을 `app/ui.py` 컴포넌트로 렌더한다.

## 한글 프롬프트

`rlm/prompts.py`의 `SYSTEM_PROMPT` + `ORCHESTRATOR_ADDENDUM`이 모델 행동을 정의한다
(오케스트레이터로서 위임·검증·종료). 프롬프트는 한글이며, 원본 영문 프롬프트 대조는
`docs/prompts-reference.md` 참고. 설계 배경은 `docs/superpowers/specs/`,
`docs/superpowers/plans/`에 있다.

## 규약

- 커밋 메시지는 제목·본문 모두 한글 (전역 규칙). `Co-Authored-By` 트레일러는 유지.
- 코드 주석·docstring·프롬프트는 한글로 작성한다 (기존 코드 스타일).
