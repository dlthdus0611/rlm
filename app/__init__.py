"""RLM 응용 계층 — UI(streamlit 플레이그라운드)와 CLI 데모.

코어 rlm 패키지를 소비하는 화면 계층. eval(평가 계층)과 대칭이다.
- playground: 문서 업로드→그 문서로 QA 플레이그라운드 (`streamlit run app/playground.py`)
- pages/1_평가: data/ 테스트셋을 라이브로 실행·채점하는 평가 페이지
- trace: 스트림 업데이트→화면용 트레이스 변환 (streamlit 비의존 순수 모듈)
- eval_run: 문항별 스트리밍 실행·채점 오케스트레이션 (순수, trace와 대칭)
- ui: 공용 UI 디자인 시스템(히어로·스탯 카드·배지·트레이스 렌더러)
"""
