"""스트림릿 공용 UI 조각 — 두 페이지(플레이그라운드·평가)가 같은 디자인 언어를 쓰게 한다.

로직 없음(순수 렌더). app_trace의 TraceEntry를 그리는 render_trace 등 공통 요소를 모아
중복을 없애고 톤을 통일한다. streamlit에만 의존.
"""
import streamlit as st

VERDICT_ICON = {"correct": "✅", "partial": "🟡", "incorrect": "❌"}

_CSS = """
<style>
  /* 상단 여백을 줄여 첫 화면 밀도를 높인다 */
  .block-container { padding-top: 2.4rem; }
  /* 코드 블록·expander를 조금 더 촘촘하게 */
  [data-testid="stExpander"] summary { font-weight: 600; }
</style>
"""


def page_header(icon: str, title: str, subtitle: str) -> None:
    """set_page_config + 제목 + 한 줄 설명 + 공통 CSS. 페이지 첫 호출로 둔다."""
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title(f"{icon} {title}")
    st.caption(subtitle)


def security_note() -> None:
    """샌드박스 없음 경고 — 큰 박스 대신 담백한 캡션으로."""
    st.caption(
        "⚠️ 모델이 생성한 Python 코드를 in-process `exec()`로 실행합니다(샌드박스 없음). "
        "신뢰할 수 있는 입력·본인 머신에서만 사용하세요."
    )


def api_key_badge(present: bool) -> None:
    """사이드바 하단의 API 키 상태 배지."""
    if present:
        st.success("OPENROUTER_API_KEY 감지됨", icon="✅")
    else:
        st.error("OPENROUTER_API_KEY 없음 — .env 또는 환경변수에 설정하세요.", icon="🚫")


def render_trace(entries) -> None:
    """TraceEntry 목록을 턴별 코드/실행 결과로 그린다(두 페이지 공용)."""
    for e in entries:
        if e.kind == "model":
            st.markdown(f"**🧠 턴 {e.turn}**")
            if e.text:
                st.markdown(e.text)
            for block in e.code_blocks:
                st.code(block, language="python")
        elif e.kind == "exec":
            st.markdown(f"**💻 턴 {e.turn} · 실행 결과**")
            st.code(e.text or "(출력 없음)")
