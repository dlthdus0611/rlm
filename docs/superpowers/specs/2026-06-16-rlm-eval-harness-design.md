# RLM 평가 하니스 설계

작성일: 2026-06-16

## 목적

삼성전자 2023 사업보고서 기반 QA 테스트셋(`data/qa_testset.json` 100문항 + `data/qa_crosssection.json` 30문항)으로
**RLM의 정답 정확도를 측정**하는 평가 하니스를 만든다. 프로젝트 최종 목표(RLM이 기본 RAG보다 실제 효과가 있는지
실증→팀 공유)의 첫 단계로, RAG 베이스라인은 이번 범위에서 제외하고 **RLM 단독 평가**에 집중한다.

## 범위

- 포함: 평가 로직 모듈(`rlm/eval.py`), CLI 러너(`eval_run.py`), 단위 테스트(`tests/test_eval.py`)
- 제외(차후): RAG 베이스라인, Streamlit 평가 페이지. 핵심 로직(`rlm/eval.py`)은 차후 어느 프론트엔드든 재사용 가능하게 설계.

## 아키텍처

### `rlm/eval.py` — streamlit/CLI 비의존 순수 로직

기존 `app_trace.py`처럼 streamlit·네트워크에 의존하지 않아 FakeChat으로 단위 테스트 가능하게 한다.
LLM은 `.invoke()`를 가진 Runnable로 **인자 주입**한다(`graph.py`의 root_llm/sub_llm 패턴과 동일).

데이터클래스:
- `QAItem`: `id, difficulty, question, answer, page, section(또는 sections), question_textbook` 등 테스트셋 필드. 단일/교차 공통 처리.
- `Verdict`: `label`("correct"|"partial"|"incorrect"), `reason`(str)
- `EvalResult`: `item: QAItem, model_answer: Optional[str], turns: int, verdict: Verdict, error: Optional[str]`

함수:
- `load_testset(path) -> list[QAItem]` — `qa_testset.json`/`qa_crosssection.json`을 읽어 QAItem 리스트로. 누락 필드는 기본값.
- `judge(question, gold, candidate, judge_llm) -> Verdict` — LLM-as-judge. 프롬프트를 만들어 `judge_llm.invoke()` 호출, 응답에서 JSON `{label, reason}` 파싱. 파싱 실패 시 1회 재시도 후 `incorrect`(reason에 원문 일부 기록).
- `run_one(item, context, root_llm, sub_llm, judge_llm, max_depth, max_iterations) -> EvalResult` — `build_rlm_graph`로 그래프를 만들어 실행, 최종 답과 턴 수를 얻고 `judge`로 채점. 문항별 예외는 잡아 `EvalResult(error=...)`로 기록(배치 계속). `final_answer` 미제출 시 judge 건너뛰고 `incorrect`.
- `aggregate(results) -> dict` — 전체·난이도별 집계(아래 지표 참조).

턴 수 캡처: `run_one`은 `graph.stream(stream_mode="updates")`로 돌며 `app_trace.format_update`를 재사용해 최종 답과 턴 수를 수집한다(`app_trace`는 streamlit 비의존이라 import 가능).

### `eval_run.py` — 레포 루트 CLI

`demo_tickets.py`처럼 `load_dotenv()`를 패키지 import 뒤에 호출(설정은 호출 시점에 환경변수에서 읽힘).

인자:
- `--set {single,cross,both}` (기본 single)
- `--n N` (샘플 수; 미지정 시 전체)
- `--seed` (샘플링 재현용, 기본 42)
- `--difficulty {low,medium,high,expert}` (반복 지정 가능, 미지정 시 전체)
- `--question-field {question,question_textbook}` (기본 `question` = 대충질문체. 정석 버전 비교용 토글)
- `--judge-model` (기본 root 모델)
- `--root-model` / `--sub-model` / `--max-iterations` / `--max-depth`
- `--out PATH` (결과 JSON 저장 경로, 기본 `data/eval_results.json`)

흐름: `data/samsung_2023.txt`를 context로 로드 → 테스트셋 로드·필터(set/difficulty)·샘플(n, seed) → 순차 실행하며 진행률·문항별 한 줄 출력 → 요약표 출력 → 결과 JSON 저장.

## 데이터 흐름

```
testset JSON → QAItem 리스트 → 필터(set/difficulty) → 샘플(n, seed)
  → for each item: run_one(context = 삼성 문서)
        → build_rlm_graph + stream → 최종답 + 턴수
        → judge(question, gold, 최종답) → Verdict
        → EvalResult
  → aggregate → 요약표 출력 + 결과 JSON
```

## 채점 (LLM-as-judge)

- 심판에게 **질문 · 정답(gold) · 모델답변** 3개를 주고 JSON `{label, reason}`을 요청하는 한글 프롬프트.
- `label` 기준:
  - `correct`: 핵심 사실/수치가 정답과 일치(표현 차이는 무관, 숫자는 일치해야 함)
  - `partial`: 다중 파트 답에서 일부만 맞거나, 핵심 수치는 맞고 요구된 부가정보 누락
  - `incorrect`: 틀림/핵심 누락/무응답
- 숫자 일치를 엄격히 보도록 명시(반올림·단위 주의). 정답에 여러 값이 있으면 모두 맞아야 correct.
- RLM이 `final_answer=None`이면 judge 건너뛰고 `incorrect`.
- judge 출력 파싱 실패 → 1회 재시도 → 그래도 실패면 `incorrect`(reason에 원문 기록).
- judge 모델 기본값: root 모델(채점 신뢰성), `--judge-model`로 변경.

## 지표 (요약표)

- 전체: `score = (correct + 0.5·partial) / total`, correct%·partial%·incorrect%, 에러 수, 평균 턴 수
- 난이도별(low/medium/high/expert): 동일 집계
- 결과 JSON: 문항별 `{id, difficulty, question, gold, model_answer, turns, label, reason, error}`

## 에러 처리

- 문항별 API/실행 예외 → `EvalResult(error=...)`, 에러로 집계(정답 아님), 배치는 계속 진행.
- judge 출력 파싱 실패 → 재시도 → `incorrect`.
- context 파일 부재 등 치명적 오류는 CLI 시작 시 명확히 안내하고 중단.

## 테스트 (`tests/test_eval.py`, 네트워크 없이 FakeChat)

기존 `tests/test_graph.py`의 FakeChat/FakeSub 패턴을 따른다.

- `judge()`: FakeChat이 정해진 JSON을 반환 → correct/partial/incorrect 및 파싱 실패 폴백 검증
- `aggregate()`: 합성 EvalResult로 전체·난이도별 집계 검증
- `load_testset()`: 작은 픽스처 JSON 파싱 검증
- `run_one()`: 기존 FakeChat(RLM 응답)+FakeSub+가짜 judge로 전 경로를 오프라인 검증(최종답·턴수·verdict)

## 규약

- 주석·docstring·프롬프트·CLI 도움말은 한글(기존 스타일).
- 커밋 메시지 한글.
- `rlm/eval.py`는 LLM을 Runnable 인자로만 받는다(직접 모델 생성 금지) — 테스트 가능성·기존 경계 유지.
