"""스트림릿 공용 UI — 커스텀 CSS 디자인 시스템으로 두 페이지의 룩을 통일한다.

로직 없음(순수 렌더). 히어로 헤더·스탯 카드·판정 배지·트레이스 렌더러를 제공한다.
색은 색상 두 스톱의 액센트 그라디언트를 쓰고, 표면색은 prefers-color-scheme/색상 혼합으로
시스템 다크·라이트를 자동으로 따른다. streamlit에만 의존.
"""
import streamlit as st

ACCENT_1 = "#8B5CF6"   # violet
ACCENT_2 = "#5B8CFF"   # indigo-blue
VERDICT_ICON = {"correct": "✅", "partial": "🟡", "incorrect": "❌"}

_THEME_CSS = f"""
<style>
:root {{ --a1: {ACCENT_1}; --a2: {ACCENT_2}; }}

/* 크롬 정리 + 중앙 정렬 레이아웃 (레이아웃에 영향 없는 상단 우측 툴바/푸터만 숨김) */
footer, [data-testid="stStatusWidget"], [data-testid="stToolbar"] {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1160px; }}

/* 히어로 헤더 */
.rlm-hero {{
  position: relative; border-radius: 20px; padding: 26px 30px; margin-bottom: 14px;
  border: 1px solid color-mix(in srgb, var(--a1) 30%, transparent);
  background:
    radial-gradient(130% 150% at 0% 0%, color-mix(in srgb, var(--a1) 18%, transparent), transparent 58%),
    radial-gradient(130% 150% at 100% 0%, color-mix(in srgb, var(--a2) 16%, transparent), transparent 55%);
}}
.rlm-hero h1 {{
  font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 6px 0;
  background: linear-gradient(92deg, var(--a1), var(--a2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.rlm-hero p {{ margin: 0; opacity: 0.72; font-size: 0.94rem; line-height: 1.5; }}
.rlm-chip {{
  display: inline-flex; align-items: center; gap: 6px; margin-top: 14px;
  font-size: 0.72rem; padding: 4px 11px; border-radius: 999px; opacity: 0.85;
  border: 1px solid color-mix(in srgb, var(--a1) 35%, transparent);
  background: color-mix(in srgb, var(--a1) 8%, transparent);
}}

/* 스탯 카드 */
.rlm-stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 8px; }}
.rlm-stat {{
  flex: 1; min-width: 116px; border-radius: 16px; padding: 15px 18px;
  border: 1px solid color-mix(in srgb, var(--a1) 16%, #88888833);
  background: color-mix(in srgb, var(--a1) 5%, transparent);
  transition: transform .12s ease, border-color .12s ease;
}}
.rlm-stat:hover {{ transform: translateY(-2px);
  border-color: color-mix(in srgb, var(--a1) 45%, transparent); }}
.rlm-stat .v {{ font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }}
.rlm-stat .l {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em; opacity: 0.6; margin-top: 3px; }}
.rlm-stat.hl {{ border-color: color-mix(in srgb, var(--a1) 40%, transparent);
  background: color-mix(in srgb, var(--a1) 10%, transparent); }}
.rlm-stat.hl .v {{
  background: linear-gradient(92deg, var(--a1), var(--a2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}

/* 판정 배지 */
.rlm-badge {{ padding: 2px 10px; border-radius: 999px; font-size: 0.74rem; font-weight: 700; white-space: nowrap; }}
.rlm-badge.correct   {{ background: #10b98122; color: #10b981; }}
.rlm-badge.partial   {{ background: #f59e0b22; color: #f59e0b; }}
.rlm-badge.incorrect {{ background: #ef444422; color: #ef4444; }}

/* Primary 버튼 그라디언트 */
.stButton button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background: linear-gradient(92deg, var(--a1), var(--a2)) !important;
  border: none !important; font-weight: 700 !important;
  box-shadow: 0 6px 18px color-mix(in srgb, var(--a1) 35%, transparent);
}}
.stButton button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
  filter: brightness(1.07); }}
</style>
"""


def hero(icon: str, title: str, subtitle: str, chip: str = "") -> None:
    """set_page_config + 그라디언트 히어로 헤더 + 공통 테마. 페이지 첫 호출로 둔다."""
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
    chip_html = f'<div class="rlm-chip">{chip}</div>' if chip else ""
    st.markdown(
        f'<div class="rlm-hero"><h1>{icon} {title}</h1>'
        f'<p>{subtitle}</p>{chip_html}</div>',
        unsafe_allow_html=True,
    )


def stat_cards(cards) -> str:
    """(label, value, highlight) 목록을 스탯 카드 HTML 문자열로. st.empty 갱신에도 쓴다."""
    body = "".join(
        f'<div class="rlm-stat{" hl" if hl else ""}">'
        f'<div class="v">{value}</div><div class="l">{label}</div></div>'
        for label, value, hl in cards
    )
    return f'<div class="rlm-stats">{body}</div>'


def verdict_badge(label: str) -> str:
    """판정 라벨을 알약형 배지 HTML로."""
    return f'<span class="rlm-badge {label}">{label}</span>'


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
