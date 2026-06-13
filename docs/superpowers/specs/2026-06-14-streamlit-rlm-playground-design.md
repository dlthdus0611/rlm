# Streamlit RLM 플레이그라운드 설계 문서

- 날짜: 2026-06-14
- 상태: 설계 승인됨 (구현 전 사용자 검토 단계)
- 종류: 신규 기능 (데모 UI)

## 1. 목적

특정 티켓 예시 전용이 아니라, **임의의 context + 질문 무엇이든 넣고 RLM을 돌려볼 수 있는
범용 플레이그라운드**를 Streamlit으로 만든다. RLM이 푸는 과정(턴마다 생성한 코드와 REPL
출력)을 실시간 트레이스로 보여줘 메커니즘을 눈으로 확인하게 한다. 기존 지저분한 티켓
데이터셋은 원클릭 "샘플"로 격하된다.

## 2. 비목표 (YAGNI)

- 멀티 세션/계정/영속화, 결과 DB 저장, 인증 없음.
- 임의 입력에 대한 자동 채점 없음(정답을 모름) — 채점은 티켓 샘플로 채운 경우에만.
- 그래프/REPL 동작 변경 없음. 앱은 기존 `rlm` 패키지를 호출만 한다.
- 파일 업로드는 텍스트로 디코드 가능한 파일만(.txt/.md/.csv/.json 등). 바이너리·PDF 파싱 안 함.

## 3. 아키텍처 / 파일 구성

- `streamlit_app.py` (루트) — UI 레이어. Streamlit 위젯, 입력 수집, `graph.stream` 구동,
  트레이스/결과 렌더. `app_trace`와 `rlm`, `demo_tickets.make_tickets`를 호출.
- `app_trace.py` (루트) — **streamlit 비의존 순수 모듈**. `graph.stream(stream_mode="updates")`가
  주는 업데이트 dict를 화면용 항목으로 변환하는 로직. 단위 테스트 대상.
- `tests/test_app_trace.py` — `app_trace`의 순수 함수 테스트(합성 dict, 네트워크 없음).
- 재사용: `rlm.build_rlm_graph`, `rlm.make_llm`, `rlm.config.get_settings`, `demo_tickets.make_tickets`.

### 의존 방향
`streamlit_app.py` → (`app_trace`, `rlm`, `demo_tickets`). `app_trace` → `rlm.parse_code_blocks`만
(streamlit·UI 의존 없음). 단방향, 순환 없음.

## 4. `app_trace.py` (순수 로직)

```python
from dataclasses import dataclass
from rlm.parsing import parse_code_blocks

@dataclass
class TraceEntry:
    kind: str       # "model" | "exec"
    turn: int       # 1-based 턴 번호
    text: str       # 표시할 본문(모델 산문, 코드, 또는 REPL 출력)
    code_blocks: list[str]  # kind=="model"일 때 추출된 ```repl``` 코드들

# 최종 답은 TraceEntry가 아니라 format_update 반환값 final_answer로 전달한다.

def format_update(update: dict, turn: int) -> tuple[list[TraceEntry], int, str | None]:
    """graph.stream(stream_mode='updates')의 한 업데이트를 트레이스 항목들로 변환.

    update 예: {"call_model": {"messages": [turn_msg, ai_msg]}} 또는
               {"execute_code": {"messages": [human_msg], "iteration": k, "final_answer": ...}}.
    반환: (생성된 TraceEntry 목록, 갱신된 turn, final_answer 또는 None).
    """
```

- `call_model` 업데이트 → 마지막 AIMessage content에서 `parse_code_blocks`로 코드 추출,
  `TraceEntry(kind="model", turn, text=프롬프트 산문, code_blocks=...)`. turn += 1.
- `execute_code` 업데이트 → 마지막 메시지 content(REPL 피드백)를 `TraceEntry(kind="exec", turn, text=...)`.
  `final_answer`가 None이 아니면 함께 반환.
- 알 수 없는 노드 키(setup 등) → 빈 목록.

메시지 content가 str이 아닐 수 있으니 `str(...)`로 방어(그래프 `execute_code`와 동일 관례).

## 5. `streamlit_app.py` (UI)

### 레이아웃
- 제목 + 한 줄 설명 + ⚠️ 보안 경고(모델 생성 코드를 in-process `exec`; 신뢰 환경 한정).
- **사이드바(컨트롤)**: `max_iterations`(slider 4–20, 기본 12), `max_depth`(0–2, 기본 1),
  root 모델 / sub 모델(text, 기본 `get_settings()` 값), 🔑 `OPENROUTER_API_KEY` 상태 표시.
- **메인 입력**:
  - "샘플 채우기" 영역: 버튼 "지저분한 티켓 60건". 누르면 `make_tickets(n, seed)`로 context를
    세션에 채우고, 질문을 티켓용 기본 질문으로 프리필, 정답 카운트를 세션에 저장(샘플 채점용).
    n/seed 입력은 이 영역에만 노출.
  - `context` textarea(큰 높이) + 파일 업로드(텍스트 디코드). 업로드 시 내용으로 textarea 대체.
  - `question` textarea.
  - context 글자수 표시(모델이 실제로 보는 건 메타데이터뿐임을 캡션으로 강조).
- **▶ RLM 실행** 버튼.

### 실행 흐름
1. 입력 검증: context/question 비어 있으면 경고하고 중단. API 키 없으면 `st.error` 안내.
2. `graph = build_rlm_graph(make_llm(root), make_llm(sub), max_depth, max_iterations)`.
3. `st.status`/컨테이너 안에서
   `for upd in graph.stream({"question":q, "context":c, "depth":0}, config={"recursion_limit": 2*max_iterations+10}, stream_mode="updates"):`
   - `app_trace.format_update(upd, turn)` → 각 항목을 렌더:
     - `kind=="model"`: "🧠 턴 k" 헤더 + 산문 + 코드블록(`st.code(language="python")`).
     - `kind=="exec"`: "💻 실행 결과" + `st.code`(stdout/traceback).
   - `final_answer`가 잡히면 루프 후 결과 섹션 표시.
4. **결과**: 모델 답을 강조 표시. 세션에 샘플 정답 카운트가 있고 이번 실행이 그 샘플이면
   `expected`와 비교해 `✅/❌ 일치`(`demo_tickets`의 채점 규칙 재사용 — 공백 제거 후 부분문자열).
   임의 입력이면 답만.

### 상태 처리
- `st.session_state`로 context/question/샘플 정답 보존(버튼 클릭과 rerun 사이 유지).
- 실행을 `try/except`로 감싸 `make_llm` RuntimeError(키 없음)·API 오류를 `st.error`/`st.exception`으로.

## 6. 의존성 / 실행

- `requirements.txt`에 `streamlit>=1.30` 추가, venv 설치.
- 실행: `streamlit run streamlit_app.py`. README "데모" 섹션에 추가(기존 CLI `python demo_tickets.py`는 유지).
- `OPENROUTER_API_KEY` 필요(`.env` 또는 환경변수). 앱이 `load_dotenv()` 호출.

## 7. 테스트

- `tests/test_app_trace.py`: `format_update`를
  - `call_model` 업데이트(코드 블록 1개) → `kind="model"`, 코드 추출, turn 증가.
  - `execute_code` 업데이트(피드백, final_answer=None) → `kind="exec"`, final None.
  - `execute_code` 업데이트(final_answer 채워짐) → final 반환.
  - 알 수 없는 노드 키 → 빈 목록.
  네트워크/Streamlit 없이 합성 dict로 검증.
- Streamlit UI 자체는 `streamlit run`으로 수동 스모크(데모와 동일 방침).
- 기존 전체 `pytest`는 계속 그린(앱/트레이스 추가가 기존 동작에 영향 없음).

## 8. 보안

모델 생성 Python을 in-process `exec`하는 기존 설계 그대로. 앱 상단과 README에 신뢰 환경 한정
경고 표기. 앱이 새로 약화시키는 부분 없음.

## 9. 산출물

```
streamlit_app.py        # Streamlit UI
app_trace.py            # 순수 트레이스 포맷터
tests/test_app_trace.py # 단위 테스트
requirements.txt        # + streamlit
README.md               # 실행법 추가
```
