# mini-rlm

**거대한 텍스트를 모델 컨텍스트에 욱여넣는 대신, Python REPL 변수로 던져두면 어떻게 될까?**

mini-rlm은 모델이 긴 입력을 직접 읽지 않습니다. 입력은 REPL 안의 `context` 변수로만 존재하고,
모델은 **코드를 써서** 그것을 훑고·쪼개고·sub-LLM에게 위임하며 답을 쌓아 올립니다. 논문
*"Recursive Language Models"*([arXiv:2512.24601](https://arxiv.org), repo: alexzhang13/rlm)의
핵심 메커니즘을 LangGraph로 재현한 **학습용 토이**입니다.

```
질문 + 거대 context ─▶ [모델은 context를 안 본다. 글자 수 같은 메타데이터만 본다]
                         │  ```repl``` 코드 생성
                         ▼
                      REPL 실행: context 슬라이싱 / llm_query(...) / rlm_query(...)
                         │  stdout(8K로 축약)만 다시 모델에게
                         ▼
                      답이 설 때까지 반복 ─▶ answer["ready"]=True ─▶ 최종 답
```

---

## 왜 흥미로운가 — 키워드론 못 푸는 걸 의미로 푼다

`demo_tickets.py`는 표현이 제각각이고 함정이 섞인 고객 지원 티켓 60건을 만듭니다:

```
=== 티켓 #7 ===
수고 많으십니다, ORD-1007 건인데요 발송이 자꾸 미뤄져서 돈 돌려받을 수 있을까요? ㅠㅠ
  → 환불 요청인데 '환불'이라는 단어가 없음. 사유는 '배송 지연'(역시 리터럴 없음).

=== 티켓 #0 ===
수고 많으십니다, 환불은 안 하셔도 돼요. 배송지가 맞는지만 확인 부탁드려요. 주문 #1000
  → '환불' 단어는 있지만 환불 요청이 아님(함정).
```

질문은 *"환불을 **요청**한 티켓 수와, 그중 사유가 배송 지연인 수"*. 결과:

```
[정답]   환불=25, 배송지연=6
[참고]   '환불' 단어가 박힌 티켓(순진한 카운트): 31개   ← 키워드로 세면 틀린다
[모델 답] 환불=25, 배송지연=6
[일치]    ✅
```

`context.count("환불")` 같은 한 방 카운트는 31로 빗나가지만, RLM은 티켓을 sub-LLM으로
**의미 분류**해 25/6을 맞힙니다. 입력이 거대하지 않아도, 전수·체계 집계에서 RLM이 빛나는 지점.

## 빠른 시작

```bash
# 1) 셋업 (Python 3.10+)
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # 필수 (https://openrouter.ai/keys)

# 2) CLI 데모 — 위 티켓 집계를 실제로 돌려본다
python demo_tickets.py

# 3) 웹 플레이그라운드 — 아무 텍스트·질문이나 넣고 RLM이 푸는 과정을 실시간으로 본다
streamlit run streamlit_app.py
```

플레이그라운드에서는 임의의 긴 텍스트를 붙여넣거나 파일을 올리고 질문을 적으면, **턴마다 모델이
생성한 코드와 그 실행 결과**가 차례로 흐르며 마지막에 답이 뜹니다("지저분한 티켓" 샘플은 원클릭).

테스트는 네트워크·키 없이 돕니다(가짜 모델 사용):
```bash
pytest -v
```

## 어떻게 도는가 (RLM 핵심 메커니즘)

1. **컨텍스트=핸들** — 거대 입력은 REPL 변수 `context`로만 적재. 모델 메시지엔 글자 수 등 메타데이터만.
2. **코드로 다루기** — 모델이 ```` ```repl ```` 블록을 생성 → 실행 → stdout만 피드백으로 본다.
3. **재귀 sub-call** — `llm_query`(단발 위임) / `rlm_query`(같은 그래프를 depth+1로, 한도 초과 시 폴백).
4. **출력 축약 + 종료** — stdout은 8K자에서 잘리고, `answer["ready"]=True`로 최종 답을 제출.

## 논문/원본 repo와의 차이 (토이 단순화)

| 항목 | 원본(alexzhang13/rlm) | 이 토이 |
|---|---|---|
| 제어 루프 | 자체 구현 | LangGraph StateGraph |
| 코드 실행 | docker/e2b/modal 샌드박스 옵션 | in-process `exec()` (신뢰 환경 한정) |
| 병렬 | llm/rlm_query_batched | llm_query_batched만 |
| 종료 | answer dict (`ready`) | 동일 |
| 출력 축약 | ~20K자 | ~8K자 |
| 학습/compaction | 있음 | 없음 |

## ⚠️ 보안

모델이 생성한 Python 코드를 **in-process `exec()`로 그대로 실행**합니다(샌드박스 없음).
신뢰할 수 있는 입력과 본인 머신에서만 쓰세요. 프로덕션 금지.

## 더 보기

- **설정**: `OPENROUTER_API_KEY`(필수). 모델·엔드포인트는 환경변수로 오버라이드 — `RLM_ROOT_MODEL`
  /`RLM_SUB_MODEL`(기본 `openai/gpt-5.5`·`openai/gpt-5.5-mini`), `OPENROUTER_BASE_URL`. `.env`로도
  가능(`.env.example` 참고).
- **코드 구조**: 핵심은 `rlm/` 패키지(`graph`·`repl`·`parsing`·`llm`·`config`·`api`·`prompts`).
  데모는 `demo_tickets.py`(CLI)·`streamlit_app.py`(웹)·`app_trace.py`(트레이스 변환).
- **개발·아키텍처 상세**: 내부 설계·불변식·규약은 [`CLAUDE.md`](./CLAUDE.md)에. 설계 배경은 `docs/`에.
