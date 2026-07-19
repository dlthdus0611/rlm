# mini-rlm

긴 문서를 통째로 모델에 넣는 대신, **모델이 코드를 써서 그 문서를 탐색하게** 하면 어떨까?
RAG처럼 미리 잘라 검색해 두는 방식과는 다른 이 접근이 실제로 통하는지 직접 돌려보고 정확도까지
재보려는 학습용 토이다. 논문 *"Recursive Language Models"*(arXiv:2512.24601, 원본
repo: alexzhang13/rlm)의 핵심 메커니즘을 LangGraph로 최소한으로 재현했다.

![플레이그라운드 — 임의의 문서를 넣고 RLM이 푸는 과정을 실시간으로 본다](assets/screenshot-playground.png)

## RLM이 뭔가

LLM은 컨텍스트가 길어질수록 비싸지고 정확도도 떨어진다. **Recursive Language Model(RLM)**은
긴 입력을 한 번에 읽는 대신, 그것을 **모델이 코드로 뒤지는 환경**으로 바꾼다.

입력은 Python REPL의 `context` 변수로만 존재한다. 모델에게 가는 메시지에는 본문이 아니라
글자 수 같은 메타데이터만 담긴다. 모델은 코드를 생성해 `context`를 슬라이싱·검색하고, 필요한
조각을 다른 LLM 호출(`llm_query`)이나 **자기 자신의 재귀 인스턴스**(`rlm_query`)에 넘겨
처리한 뒤 결과만 모아 답을 쌓아 올린다. 부분 문제를 또 다른 모델 호출로 푼다는 데서
"재귀(recursive)"라는 이름이 왔다.

## 빠른 시작

```bash
python3.10 -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # 필수 (openrouter.ai/keys)
export OPENAI_API_KEY=...            # RAG 비교를 쓸 때만 (검색 임베딩용)

streamlit run app/playground.py      # 웹앱 (플레이그라운드 · 평가 · 비교 페이지)
pytest -v                            # 테스트 — 네트워크·키 불필요 (가짜 모델)
```

웹앱은 세 페이지로 나뉜다.

- **플레이그라운드** — 아무 문서나 붙여넣고 질문하면, 모델이 그 문서를 `context`로 두고
  코드를 생성·실행하며 답을 찾아가는 과정을 턴별로 보여준다.
- **평가 페이지** — `data/`에 준비된 QA 테스트셋(삼성 2023 사업보고서 기반)을 난이도별로
  골라 라이브로 돌리고, LLM 채점자가 정답과 대조해 매긴 정확도를 집계로 보여준다.
  같은 평가는 CLI로도 돌릴 수 있다: `python -m eval --set single --n 10`.
- **비교 페이지** — 같은 문항을 RLM과 벡터 DB 기반 RAG에 나란히 통과시켜 대조한다(아래).

![평가 페이지 — 테스트셋을 돌려 난이도별 정확도를 집계한다 (예시 데이터)](assets/screenshot-eval.png)

## RLM vs RAG 비교

도입부의 질문 — *코드로 문서를 뒤지는 RLM이 미리 잘라 검색해 두는 RAG보다 실제로 나은가* — 을
직접 재보는 벤치다. **공정성 계약**으로 "RLM이냐 RAG냐" 한 변수만 다르게 둔다: 같은 문서·같은
문항·같은 LLM 채점자·같은 생성 모델을 고정하고, **문서에 접근하는 방식만** 다르다 — RLM은 코드로
전체 문서를 훑고, RAG는 임베딩·벡터 검색으로 top-k 조각만 꺼낸다(HyDE 쿼리 확장·MMR·LLM 리랭킹을
얹은 강화 RAG). 정확도·비용(토큰)·지연(시간)을 3축으로, 그리고 RAG는 근거적중률(검색 조각이 정답
근거를 포함했는지)까지 대조한다.

```bash
export OPENAI_API_KEY=...             # RAG 검색 임베딩용 (추가로 필요)
python -m eval.compare --systems rlm,rag --set single --n 10
```

CLI는 시스템별 정확도·평균 토큰·평균 지연·근거적중률을 한 표로 출력하고 결과를 JSON으로 저장한다.
비교 페이지는 같은 실행을 라이브로 돌려 문항마다 두 시스템의 판정·추론 트레이스·검색 조각을 나란히
보여준다. 두 시스템은 공통 `Solver` 플러그인 구조(`eval/systems.py`) 위에 얹혀 있어, 세 번째
전략(예: 전체 문서 통째로 넣기)도 한 줄 등록으로 표에 합류시킬 수 있다.

## 어떻게 도는가

한 번의 실행은 네 조각으로 이뤄진다.

1. **컨텍스트는 핸들로만** — 거대 입력은 REPL 변수 `context`에만 있고, 모델 메시지엔
   메타데이터만 간다. 본문이 프롬프트로 새지 않는지는 테스트가 검증한다.
2. **코드로 다룬다** — 모델이 ```` ```repl ```` 블록을 생성하면 실행하고, 그 stdout만
   되돌려준다. 모델은 그걸 보고 다음 코드를 정한다.
3. **재귀 sub-call** — `llm_query`로 조각을 위임하거나, `rlm_query`로 depth+1 재귀
   호출한다(깊이 한도를 넘으면 단순 위임으로 폴백).
4. **축약과 종료** — REPL 출력은 8K자로 축약하고, 모델이 `answer["ready"] = True`를
   세팅하면 그 답을 최종 제출로 본다.

## ⚠️ 보안

모델이 생성한 Python 코드를 **in-process `exec()`로 그대로 실행한다**(샌드박스 없음).
원본 repo의 docker/e2b 샌드박스를 의도적으로 단순화한 것이니, 신뢰할 수 있는 입력과 본인
머신에서만 쓸 것.

## 더 보기

모델·엔드포인트 설정(`RLM_ROOT_MODEL`/`RLM_SUB_MODEL`, 기본 `openai/gpt-5.6-sol`·
`openai/gpt-5.6-luna`)은 [`.env.example`](./.env.example)에, 코드 구조·아키텍처·규약은
[`CLAUDE.md`](./CLAUDE.md)에, 설계 배경 문서는 `docs/`에 있다.
