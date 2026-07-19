"""RLM 응용 계층 — UI(streamlit 플레이그라운드)와 CLI 데모.

코어 rlm 패키지를 소비하는 화면 계층. eval(평가 계층)과 대칭이다.
- streamlit_app: 문서 업로드→그 문서로 QA 플레이그라운드 (`streamlit run app/streamlit_app.py`)
- app_trace: 스트림 업데이트→화면용 트레이스 변환 (streamlit 비의존 순수 모듈)
"""
