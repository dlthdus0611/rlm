# mini-rlm

논문 *"Recursive Language Models"*(arXiv:2512.24601, repo: alexzhang13/rlm)의 핵심 메커니즘을
LangGraph로 재현한 학습용 토이.

## RLM이란?

LLM은 컨텍스트가 길어질수록 비싸지고 정확도도 떨어진다. **Recursive Language Model(RLM)**은
긴 입력을 한 번에 읽는 대신 **모델이 코드로 탐색하는 환경**으로 둔다: 입력은 Python REPL의
`context` 변수로만 존재하고, 모델은 그것을 슬라이싱·검색하거나 조각을 다른 LLM 호출(`llm_query`)
또는 **자기 자신의 재귀 인스턴스**(`rlm_query`)에 넘겨 처리한 뒤 결과만 모은다. 모델이 부분 문제를
또 다른 모델 호출로 푼다는 데서 "재귀(recursive)"라는 이름이 온다.

## 빠른 시작

```bash
python3.10 -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # 필수 (openrouter.ai/keys)

streamlit run app/playground.py # 웹앱: 문서 업로드→QA 플레이그라운드 + 평가 페이지(사이드바)
python -m eval --set single --n 10 # CLI 평가: 삼성 사업보고서 QA 테스트셋으로 정확도 측정
pytest -v                          # 테스트 (네트워크·키 불필요)
```

웹앱은 멀티페이지다 — **플레이그라운드**(임의 문서로 추론 과정 관찰)와 **평가 페이지**(`data/`의
QA 테스트셋을 난이도별로 골라 라이브 실행·채점, 집계·문항별 드릴다운·JSON 다운로드).

## 메커니즘

1. **컨텍스트=핸들** — 거대 입력은 REPL 변수 `context`로만, 모델 메시지엔 메타데이터만.
2. **코드로 다루기** — 모델이 ```` ```repl ```` 블록 생성 → 실행 → stdout만 피드백.
3. **재귀 sub-call** — `llm_query`(위임) / `rlm_query`(depth+1, 한도 초과 시 폴백).
4. **출력 축약 + 종료** — stdout 8K 축약, `answer["ready"]=True`로 제출.

## ⚠️ 보안

모델이 생성한 Python을 **in-process `exec()`로 실행**한다(샌드박스 없음). 신뢰 환경·본인 머신 한정.

## 더 보기

설정(`RLM_ROOT_MODEL`/`RLM_SUB_MODEL`, 기본 `openai/gpt-5.6-sol`·`openai/gpt-5.6-luna` 등)은 `.env.example`,
코드 구조·아키텍처·규약은 [`CLAUDE.md`](./CLAUDE.md), 설계 배경은 `docs/`.
