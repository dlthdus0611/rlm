# mini-rlm 설계 문서 (LangGraph 기반)

- 날짜: 2026-06-13
- 상태: 설계 승인됨 (구현 전 사용자 검토 단계)
- 참고 구현: https://github.com/alexzhang13/rlm (논문 "Recursive Language Models", arXiv:2512.24601)
- 프레임워크: **LangGraph** (LangChain 생태계 중간 레이어 — 커스텀 제어 루프/재귀에 최적)

## 1. 목적

논문/레포의 **Recursive Language Model(RLM) 핵심 메커니즘**을 LangGraph로 재현하는 학습용 토이.
RLM은 본질적으로 "모델이 코드 생성 → REPL 실행 → 축약 stdout 피드백 → 종료조건까지 반복 + 코드 안 재귀 sub-call" 이라는 **커스텀 제어 루프**이며, 이는 LangGraph `StateGraph`의 노드/조건부 엣지로 논문 Algorithm 1과 거의 1:1로 매핑된다. 부수 효과로 **LangSmith 트레이싱으로 재귀·sub-call 트리를 시각화**할 수 있어 학습에 유리.

재현할 핵심 메커니즘 4가지:
1. **컨텍스트를 핸들로** — 거대 입력을 모델 메시지에 넣지 않고 REPL 변수 `context`로 적재, 모델은 메타데이터만 봄.
2. **코드로 다루기** — 모델이 ```` ```repl ```` 블록 생성 → 파싱·실행 → stdout만 관찰.
3. **재귀 sub-call** — REPL 안 `llm_query`(단일 호출) / `rlm_query`(child 그래프를 depth+1로 invoke).
4. **출력 축약 + 종료 신호** — stdout을 일정 길이에서 truncate, `answer["ready"]=True` 로 최종 답 반환.

## 2. 비목표 (YAGNI)

소켓 LMHandler, 외부 샌드박스(docker/e2b/modal), compaction, 모델 학습/distill, 다중 백엔드 추상화, `*_batched` 병렬 호출, 비동기 실행, 체크포인터/영속화(인메모리 단발 invoke로 충분).

## 3. 의존성 / 환경

- **Python 3.10+** 필요(LangGraph 요구). 시스템 기본은 3.9.6 → **venv(또는 uv)로 3.10+ 가상환경** 구성. README에 명시.
- 패키지: `langgraph`, `langchain-openai`, `langchain-core`. (정확한 최소 버전은 구현 시 `langchain-dependencies` 스킬로 확인 후 `requirements.txt`에 핀.)
- 백엔드: **OpenRouter** (OpenAI 호환). `ChatOpenAI(model=..., base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])`.
  - 루트 모델/sub 모델을 분리(상수). 루트=코드 생성 잘하는 모델, sub=저렴한 모델.
- 인증: `OPENROUTER_API_KEY`(필수). 없으면 즉시 명확한 에러.
- 관측성(선택): `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT` 설정 시 자동 트레이싱.

## 4. 아키텍처

### 4.1 State 스키마 (`RLMState`, TypedDict)
- `question: str` — 짧은 질의(모델이 직접 봄).
- `context: str` — 거대 본문(REPL 핸들로만 적재, 메시지에 안 들어감).
- `messages: Annotated[list, add_messages]` — system + metadata + 턴별 AI/Human 메시지 누적.
- `depth: int` — 현재 재귀 깊이(루트=0).
- `iteration: int` — 루프 카운터(노드에서 증가).
- `final_answer: str | None` — `answer["ready"]` 시 채워짐.
- `repl: object` — 영속 REPL 인스턴스(턴 간 변수 유지용). *체크포인터 미사용이라 비직렬 객체를 state에 두는 토이 단순화.*

### 4.2 노드
1. **`setup`** (진입): `REPL` 생성 → `context` 적재, `llm_query`/`rlm_query`/`answer`/`SHOW_VARS` 네임스페이스 주입. 초기 `messages = [SystemMessage(시스템프롬프트), HumanMessage(메타데이터+question)]` 구성. 반환 `{messages, repl, iteration: 0}`.
   - **프롬프트는 원본 repo를 최대한 그대로 사용**한다. 시스템 프롬프트 = (적응판 `RLM_SYSTEM_PROMPT`) + (적응판 `ORCHESTRATOR_ADDENDUM`). 메타데이터·턴 프롬프트도 동일. 전문은 `docs/prompts-reference.md` 참고(원본 verbatim + 우리 적응판 §4 모두 수록). 적응 = `*_batched`·커스텀툴 제거, 20K→8K, `context`=str.
   - 턴 프롬프트(`"Turn {i+1}/{max_iterations}:"`, 턴 0 안전장치 포함)는 `call_model` 직전에 user 메시지로 추가한다(원본 `build_user_prompt`와 동일 취지).
2. **`call_model`**: 루트 `ChatOpenAI`로 `state["messages"]` 호출 → `{messages: [AIMessage]}`.
3. **`execute_code`**: 마지막 AIMessage에서 ```` ```repl ```` 블록 파싱 → 각 블록 `repl.run` → stdout을 `MAX_OUTPUT_CHARS(=8000)`에서 truncate(`"...[+N chars]"`) → `{messages: [HumanMessage(피드백)], iteration: state["iteration"]+1, final_answer?}`. `iteration`은 reducer 없는 필드라 **기존값+1을 계산해 덮어쓰기**(reducer 함정 회피). 코드 블록 0개면 "```repl``` 블록으로 작성하라" 안내를 피드백으로.

### 4.3 엣지
- `START → setup → call_model → execute_code`
- `execute_code` 이후 **조건부 엣지** `should_continue`: `final_answer is not None` 또는 `iteration >= max_iterations` → `END`, 아니면 → `call_model`.
- (무한 루프 방지: 종료조건 + max_iterations 이중 안전장치.)

### 4.4 컴포넌트 (그래프 외부 순수 모듈)
- **`REPL`**: 영속 Python 네임스페이스. `run(code: str) -> str`(stdout 반환, 예외는 traceback 문자열). 주입: `context`, `llm_query`, `rlm_query`, `answer`(특수 dict, `_AnswerDict` 축소판), `SHOW_VARS()`. 구현: `exec` + `contextlib.redirect_stdout`.
- **`parse_code_blocks(text) -> list[str]`**: 정규식 `r"```repl\s*\n(.*?)```"` (re.DOTALL).
- **`make_llm(model) -> ChatOpenAI`**: OpenRouter 클라이언트 팩토리.
- **`build_rlm_graph(root_model, sub_model, max_depth, max_iterations) -> CompiledGraph`**: 위 노드/엣지를 조립·compile. `setup` 노드가 REPL에 주입하는 함수들:
  - `llm_query(prompt)` → `make_llm(sub_model)` 단일 호출.
  - `rlm_query(question, context)` → `next_depth = depth+1`; `next_depth > max_depth`면 `llm_query(question+"\n\n"+context)` 폴백, 아니면 **같은 compiled 그래프를 `invoke({question, context, depth: next_depth})`** 하고 `final_answer` 반환.

## 5. 데이터 흐름

```
invoke({question, context, depth=0})
        │
     [setup] REPL 생성·context 적재, 모델엔 메타데이터만
        │
        ▼
   ┌► [call_model] ──► AIMessage(```repl``` 코드)
   │      │
   │      ▼
   │  [execute_code] repl.run → llm_query / rlm_query(→ child graph, depth+1)
   │      │            stdout truncate → messages
   │      ▼
   │  should_continue?  final_answer? or iter>=max ── yes ──► END(final_answer)
   └───────── no ──────────────────────────────────────┘
```

## 6. 데모: 티켓 집계 (`demo_tickets.py`)

"문서가 거대하지 않아도 전수 집계는 RLM이 빛난다"는 두 번째 축을 보여주는 예제.

- **데이터**: N개(기본 60) 가짜 지원 티켓 합성. 각 티켓에 (a) 환불 언급 여부, (b) 원인(배송지연/품질불량/단순변심/해당없음). 정답 카운트를 생성 시점 기록(채점용). `random.seed` 고정.
- **질문**: "환불을 언급한 티켓 수와, 그중 '배송 지연'이 원인인 수."
- **기대 동작**: 루트가 `context`를 티켓 단위 split → `for` 루프로 각 티켓 `llm_query` 분류 → 집계 → `answer` 반환.
- **출력**: 모델 답 vs 실제 정답, 일치 여부, 사용 턴 수. (LangSmith 켜져 있으면 트레이스 URL.)

## 7. 에러 처리

- `OPENROUTER_API_KEY` 없음 → 시작 시 명확한 종료.
- REPL 코드 예외 → traceback을 피드백에 담아 모델이 다음 턴 자가 수정(레포 철학과 동일).
- 코드 블록 0개 → 안내 피드백 후 계속.
- `max_iterations` 도달 → 미완료 안내 + 마지막 상태 반환.
- 재귀 폭주 방지 → `max_depth` 하드 제한.
- 네트워크/레이트리밋 → (선택) `call_model` 노드에 `RetryPolicy(max_attempts=3)`.

## 8. ⚠️ 보안

모델 생성 Python을 **in-process `exec()`로 실제 실행**. 신뢰된 task·본인 머신 한정, 프로덕션 금지. README 상단 경고. (레포는 샌드박스로 해결하나 토이 비목표.)

## 9. 테스트

- `parse_code_blocks`: 0/1/다수/언어태그 변형.
- `REPL.run`: stdout 캡처, 예외→traceback, `answer["ready"]` 콜백, 턴 간 변수 영속, truncate.
- **그래프 루프**: 루트 모델을 `FakeMessagesListChatModel`(또는 `GenericFakeChatModel`)로 교체해 미리 정한 ```repl``` 응답 시퀀스를 흘려보내며 종료조건·`messages` 누적·`execute_code` 동작을 **네트워크/키 없이** 검증.
- **depth 폴백**: `rlm_query`가 `max_depth` 초과 시 `llm_query` 폴백 경로 검증(역시 fake model).
- 데모는 실제 API 필요 → 수동 스모크. README에 실행법.

## 10. 산출물

```
side-projects/mini-rlm/
├── rlm_graph.py     # 핵심: State, 노드, 엣지, build_rlm_graph, REPL, parse, make_llm
├── demo_tickets.py  # 티켓 집계 데모 + 채점
├── test_rlm.py      # fake model 기반 단위/그래프 테스트
├── requirements.txt # 핀된 의존성
├── README.md        # venv 셋업, 실행법, 보안 경고, 논문 대조표
└── docs/superpowers/specs/2026-06-13-mini-rlm-design.md
```
```
