# mini-rlm

거대한 입력을 모델 컨텍스트에 넣지 않고 **Python REPL 변수 `context`로만 두면**, 모델이 코드를
써서 그것을 훑고·쪼개고·sub-LLM에 위임하며 답을 쌓아 올린다. 논문 *"Recursive Language Models"*
(arXiv:2512.24601, repo: alexzhang13/rlm)의 핵심 메커니즘을 LangGraph로 재현한 학습용 토이.

## 왜

`demo_tickets.py`는 표현이 제각각이고 함정이 섞인 티켓 60건을 만든다("돈 돌려주세요"=환불 요청,
"환불 정책 문의"=함정). 단순 `count("환불")`은 31로 빗나가지만, RLM은 의미로 분류해 정답
(환불=25, 배송지연=6)을 맞힌다. 입력이 작아도 전수 집계에서 RLM이 빛난다.

## 빠른 시작

```bash
python3.10 -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # 필수 (openrouter.ai/keys)

python demo_tickets.py             # CLI: 티켓 집계 데모
streamlit run streamlit_app.py     # 웹: 아무 텍스트·질문이나 넣고 추론 과정을 실시간으로
pytest -v                          # 테스트 (네트워크·키 불필요)
```

## 메커니즘

1. **컨텍스트=핸들** — 거대 입력은 REPL 변수 `context`로만, 모델 메시지엔 메타데이터만.
2. **코드로 다루기** — 모델이 ```` ```repl ```` 블록 생성 → 실행 → stdout만 피드백.
3. **재귀 sub-call** — `llm_query`(위임) / `rlm_query`(depth+1, 한도 초과 시 폴백).
4. **출력 축약 + 종료** — stdout 8K 축약, `answer["ready"]=True`로 제출.

## 원본과의 차이 (토이 단순화)

| 항목 | 원본(alexzhang13/rlm) | 이 토이 |
|---|---|---|
| 제어 루프 | 자체 구현 | LangGraph StateGraph |
| 코드 실행 | docker/e2b/modal 샌드박스 | in-process `exec()` (신뢰 환경 한정) |
| 출력 축약 | ~20K자 | ~8K자 |
| 학습/compaction | 있음 | 없음 |

## ⚠️ 보안

모델이 생성한 Python을 **in-process `exec()`로 실행**한다(샌드박스 없음). 신뢰 환경·본인 머신 한정.

## 더 보기

설정(`RLM_ROOT_MODEL`/`RLM_SUB_MODEL`, 기본 `openai/gpt-5.5`·`openai/gpt-5.5-mini` 등)은 `.env.example`,
코드 구조·아키텍처·규약은 [`CLAUDE.md`](./CLAUDE.md), 설계 배경은 `docs/`.
