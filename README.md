# mini-rlm

논문 "Recursive Language Models"(arXiv:2512.24601, repo: alexzhang13/rlm)의 핵심
메커니즘을 LangGraph로 재현한 학습용 토이.

## ⚠️ 보안 경고
이 프로젝트는 **모델이 생성한 Python 코드를 in-process `exec()`로 실제 실행**합니다.
신뢰할 수 있는 task와 본인 머신에서만 쓰세요. 프로덕션 사용 금지.

## 셋업
```bash
python3.10 -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # 필수
# (선택) LangSmith 트레이싱
# export LANGSMITH_API_KEY=...; export LANGSMITH_TRACING=true; export LANGSMITH_PROJECT=mini-rlm
```

## 테스트 (네트워크/키 불필요)
```bash
pytest -v
```

## 데모 (실제 API 필요)
```bash
# CLI: 지저분한 티켓 전수 집계 + 채점
python demo_tickets.py

# 웹 UI: 임의 context/질문으로 RLM 실행 + 추론 과정 실시간 표시
streamlit run streamlit_app.py
```
임의 텍스트를 붙여넣거나 파일을 올려 쓸 수 있고, "지저분한 티켓" 샘플도 원클릭으로 채울 수 있습니다.

## 이게 보여주는 것 (RLM 핵심 메커니즘)
1. **컨텍스트=핸들**: 거대 입력은 REPL 변수 `context`로만 적재 — 모델 메시지엔 메타데이터만.
2. **코드로 다루기**: 모델이 ```repl``` 블록을 생성 → `execute_code` 노드가 실행, stdout만 피드백.
3. **재귀 sub-call**: `rlm_query`가 같은 그래프를 depth+1로 invoke, 한도 초과 시 `llm_query` 폴백.
4. **출력 축약 + 종료**: stdout 8K 축약, `answer["ready"]=True`로 최종 답 반환.

## 논문/원본 repo와의 차이 (토이 단순화)
| 항목 | 원본(alexzhang13/rlm) | 이 토이 |
|---|---|---|
| 제어 루프 | 자체 구현 | LangGraph StateGraph |
| 코드 실행 | docker/e2b/modal 샌드박스 옵션 | in-process `exec()` (신뢰 환경 한정) |
| 병렬 | llm/rlm_query_batched | llm_query_batched만 |
| 종료 | answer dict (`ready`) | 동일 |
| 출력 축약 | ~20K자 | ~8K자 |
| 학습/compaction | 있음 | 없음 |

## 구조
- `rlm/` — 패키지
  - `parsing.py` — 코드 블록 파싱 / 출력 축약
  - `repl.py` — 코드 실행 샌드박스(REPL)
  - `graph.py` — LangGraph 제어 루프
  - `llm.py` — OpenRouter LLM 팩토리
  - `api.py` — 공개 진입점 `run()`
  - `config.py` — 환경변수 설정(pydantic-settings)
  - `prompts.py` — 한글 프롬프트
- `tests/` — 모듈별 단위/그래프 테스트
- `demo_tickets.py` — CLI 티켓 집계 데모
- `streamlit_app.py` — 웹 플레이그라운드(임의 입력 + 실시간 추론 트레이스)
- `app_trace.py` — 스트리밍 업데이트를 트레이스 항목으로 변환(순수)
- `docs/` — 설계 spec, 프롬프트 참고, 본 계획
