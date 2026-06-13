# rlm 패키지 리팩터링 설계 문서

- 날짜: 2026-06-14
- 상태: 설계 승인됨 (구현 전 사용자 검토 단계)
- 종류: **구조 리팩터링** — 기능/동작 변경 없음
- 선행 설계: `docs/superpowers/specs/2026-06-13-mini-rlm-design.md` (원 구현 설계)

## 1. 목적

현재 `rlm_graph.py` 한 파일(220줄)이 REPL 샌드박스, 코드/출력 파싱, LangGraph 제어 루프,
OpenRouter LLM 팩토리, 공개 진입점(`run`)을 모두 안고 있다. 책임이 섞여 있고, 중간에
`from langchain_openai import ChatOpenAI`(line 189)가 끼어 있는 등 구조 냄새가 있다.

이를 **책임별 모듈로 분리한 `rlm/` 파이썬 패키지**로 재구성한다. **동작은 한 줄도 바꾸지 않는다** —
순수하게 코드 위치·import 경로 재배치다.

## 2. 비목표 (YAGNI)

- 기능 추가/변경 일절 없음 (`rlm_query_batched`, 샌드박스, 비동기, 체크포인터 등은 여전히 비목표).
- 로직 리팩터링 없음 — 함수 본문/클로저는 그대로 옮기기만 한다.
- 하위 폴더 계층(`rlm/core/` 등) 없음 — 파일이 적어 평평한 패키지로 충분 (과설계 회피).
- 후방 호환 shim(`rlm_graph.py` 잔존) 없음 — 토이라 외부 소비자 없음, 깔끔히 삭제.
- `pyproject.toml`/`setup.py` 같은 패키징 메타데이터 없음 — 설치형이 아니라 레포 내 import로 충분.

## 3. 대상 디렉터리 구조

```
mini-rlm/                 # 레포 루트 (이름 그대로; 패키지는 언더스코어 불가라 rlm)
├── rlm/                  # ← 신규 패키지 (유일한 신규 폴더)
│   ├── __init__.py       # 공개 표면 재노출
│   ├── parsing.py        # 텍스트 파싱/축약 (순수, stdlib만)
│   ├── repl.py           # 코드 실행 샌드박스
│   ├── graph.py          # LangGraph 제어 루프
│   ├── llm.py            # OpenRouter 팩토리
│   ├── api.py            # 공개 진입점 run()
│   └── prompts.py        # 루트 prompts.py 이동 (내용 무수정)
├── tests/                # ← 신규 폴더
│   ├── test_parsing.py
│   ├── test_repl.py
│   ├── test_graph.py
│   ├── test_prompts.py
│   └── test_llm.py
├── conftest.py           # ← 신규 (빈 파일): 레포 루트를 sys.path에 올림
├── demo_tickets.py       # 루트 유지, import 경로만 갱신
├── README.md             # "구조" 섹션 갱신
├── requirements.txt      # 무수정
└── docs/                 # 무수정 (+ 본 spec)
```

삭제 대상: 루트 `rlm_graph.py`, 루트 `prompts.py`, 루트 `test_rlm.py` (모두 패키지/tests로 이동).

## 4. 모듈별 책임과 내용

각 모듈은 한 가지 책임만 가지며, 한 화면에 들어온다.

### 4.1 `rlm/parsing.py` — 텍스트 파싱/축약 (의존: stdlib `re`만)
`rlm_graph.py`에서 이동:
- `CODE_BLOCK_RE`
- `parse_code_blocks(text) -> list[str]`
- `MAX_OUTPUT_CHARS`
- `_truncate(s, limit) -> str`

모델 출력에서 코드 블록을 뽑고 REPL 출력을 한도에서 자르는, REPL 경계의 텍스트 마샬링.
순수 함수라 단독 테스트가 쉽다.

### 4.2 `rlm/repl.py` — 코드 실행 샌드박스 (의존: stdlib `io`/`traceback`/`contextlib`, `typing`)
`rlm_graph.py`에서 이동:
- `_AnswerDict`
- `_RESERVED`
- `REPL`

모델이 생성한 코드를 영속 네임스페이스에서 `exec`로 실행하고 stdout/traceback을 문자열로 반환.
내부 의존 0.

### 4.3 `rlm/graph.py` — LangGraph 제어 루프 (의존: `parsing`, `repl`, `prompts`, langgraph, langchain_core)
`rlm_graph.py`에서 이동:
- `RLMState`
- `BATCH_CONCURRENCY`
- `build_rlm_graph(root_llm, sub_llm, max_depth, max_iterations)` — 내부 클로저
  (`_llm_query`, `_llm_query_batched`, `setup`, `call_model`, `execute_code`,
  `should_continue`, `_rlm_query`)와 `_recursion_limit` 계산을 **본문 변경 없이** 그대로 이동.

import 갱신: `from .parsing import parse_code_blocks, _truncate`,
`from .repl import REPL`, `from .prompts import (...)`.

### 4.4 `rlm/llm.py` — OpenRouter 팩토리 (의존: stdlib `os`, langchain_openai)
`rlm_graph.py`에서 이동 (중간 끼어있던 import를 모듈 상단으로 정리):
- `OPENROUTER_BASE_URL`, `DEFAULT_ROOT_MODEL`, `DEFAULT_SUB_MODEL`
- `make_llm(model) -> ChatOpenAI`

### 4.5 `rlm/api.py` — 공개 진입점 (의존: `graph`, `llm`)
`rlm_graph.py`에서 이동:
- `run(question, context, root_model, sub_model, max_depth, max_iterations)`

import 갱신: `from .graph import build_rlm_graph`, `from .llm import make_llm,
DEFAULT_ROOT_MODEL, DEFAULT_SUB_MODEL`.

### 4.6 `rlm/prompts.py` — 프롬프트 (의존: 없음)
루트 `prompts.py`를 **내용 변경 없이** 이동.

### 4.7 `rlm/__init__.py` — 공개 표면
재노출만 한다 (로직 없음):
```python
from .api import run
from .graph import build_rlm_graph
from .llm import make_llm
from .repl import REPL
from .parsing import parse_code_blocks
```

## 5. 의존 방향 (단방향, 순환 없음)

```
parsing.py   (stdlib만)
repl.py      (stdlib만)
prompts.py   (없음)
   ▲   ▲   ▲
   └───┴───┴──< graph.py  (+ langgraph, langchain_core)
                   ▲
llm.py  (langchain_openai) ─┐
                            ▼
                         api.py
                            ▲
                       __init__.py (재노출)
```

## 6. 외부 소비자 import 갱신

### 6.1 `demo_tickets.py`
- `from rlm_graph import run` → `from rlm import run`
- 그 외 무수정 (`load_dotenv()`, `make_tickets`, `main` 등 그대로).

### 6.2 `tests/` (모듈별 분리, 내용 이동만 — 단언/로직 무수정)
기존 `test_rlm.py`의 테스트를 대상 모듈에 맞춰 나눈다:

| 새 파일 | 옮겨오는 기존 테스트 | import |
|---|---|---|
| `test_parsing.py` | `test_parse_*`(4), `test_truncate_*`(2) | `from rlm.parsing import parse_code_blocks, _truncate, MAX_OUTPUT_CHARS` |
| `test_repl.py` | `test_repl_*`(8), `_make_repl` 헬퍼 | `from rlm.repl import REPL` |
| `test_graph.py` | `FakeChat`, `FakeSub`, `test_graph_*`·`test_rlm_query_*`(8) | `from rlm.graph import build_rlm_graph`; `from langchain_core.messages import AIMessage` |
| `test_prompts.py` | `test_system_prompt_*`, `test_metadata_message_*`, `test_turn_prompt_*`(4) | `from rlm.prompts import SYSTEM_PROMPT, ORCHESTRATOR_ADDENDUM, build_metadata_message, build_turn_prompt` |
| `test_llm.py` | `test_make_llm_requires_api_key` | `from rlm.llm import make_llm` |

`FakeChat`/`FakeSub`는 그래프 테스트에서만 쓰이므로 `test_graph.py` 안에 둔다(공유 conftest 불필요).

### 6.3 루트 `conftest.py` (신규, 빈 파일)
`tests/`로 옮기면 pytest(prepend 임포트 모드)가 `tests/`만 sys.path에 넣어 `import rlm`이
깨진다. 루트에 빈 `conftest.py`를 두면 그 디렉터리(레포 루트)가 sys.path에 올라가 해결된다.
(대안: `pyproject.toml`의 `[tool.pytest.ini_options] pythonpath = "."` — 하지만 패키징
메타데이터 비목표에 맞춰 더 가벼운 빈 conftest.py를 택한다.)

## 7. README 갱신

"## 구조" 섹션을 새 레이아웃으로 교체:
```
- rlm/ — 패키지
  - parsing.py — 코드 블록 파싱 / 출력 축약
  - repl.py    — 코드 실행 샌드박스(REPL)
  - graph.py   — LangGraph 제어 루프
  - llm.py     — OpenRouter LLM 팩토리
  - api.py     — 공개 진입점 run()
  - prompts.py — 한글 프롬프트
- tests/ — 모듈별 단위/그래프 테스트
- demo_tickets.py — 티켓 집계 데모
- docs/ — 설계 spec, 프롬프트 참고
```
"테스트" 섹션의 `pytest -v` 실행법은 그대로 유효(루트에서 실행).

## 8. 검증 (동작 불변 증명)

이 작업의 성공 기준은 **새 코드 없이 기존 동작이 유지됨**이다:
1. 루트에서 `pytest -v` → 기존 27개 테스트 전부 통과(네트워크/키 불필요). 단언은 한 줄도
   바꾸지 않았으므로, 통과 = 이동/재배치가 동작을 보존했다는 증거.
2. `python -c "import rlm; from rlm import run, build_rlm_graph, make_llm, REPL, parse_code_blocks"`
   → 공개 표면 import 무오류.
3. (수동, 키 필요) `python demo_tickets.py` 스모크 — 선택.

## 9. 작업 순서 (구현 계획에서 상세화)

1. `rlm/` 패키지 생성, `rlm_graph.py`/`prompts.py` 내용을 모듈별로 이동 + 내부 import 갱신.
2. `rlm/__init__.py` 재노출 작성.
3. 루트 `rlm_graph.py`, `prompts.py` 삭제.
4. `tests/` 생성, `test_rlm.py`를 모듈별로 분리 이동 + import 갱신, 루트 `test_rlm.py` 삭제.
5. 루트 빈 `conftest.py` 추가.
6. `demo_tickets.py` import 갱신.
7. README "구조" 섹션 갱신.
8. `pytest -v`로 전체 초록 확인.
