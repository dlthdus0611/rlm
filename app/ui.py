"""스트림릿 공용 UI — 커스텀 CSS 디자인 시스템으로 "이게 streamlit?" 소리를 노린다.

로직 없음(순수 렌더). Pretendard 폰트·기본 크롬 제거·배경 그라디언트 메쉬·사이드바 브랜딩·
글래스 히어로/스탯 카드/판정 배지/트레이스 렌더러를 제공한다. 색은 두 스톱 액센트 그라디언트,
표면색은 color-mix로 시스템 다크·라이트를 자동 추종. streamlit에만 의존.

폰트는 런타임에 jsDelivr CDN에서 Pretendard를 받는다(오프라인이면 시스템 한글 폰트로 폴백).
"""
import streamlit as st

ACCENT_1 = "#8B5CF6"   # violet
ACCENT_2 = "#5B8CFF"   # indigo-blue
VERDICT_ICON = {"correct": "✅", "partial": "🟡", "incorrect": "❌"}

_FONT = ("'Pretendard Variable','Pretendard',-apple-system,BlinkMacSystemFont,"
         "'Apple SD Gothic Neo','Noto Sans KR','Segoe UI',Roboto,sans-serif")

_THEME_CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.css');

:root {{ --a1: {ACCENT_1}; --a2: {ACCENT_2};
         --ring: color-mix(in srgb, var(--a1) 40%, transparent);
         --line: color-mix(in srgb, var(--a1) 14%, #8080801f); }}

/* 폰트 전면 적용 */
html, body, .stApp, [class^="st-"], [class*=" st-"],
button, input, textarea, select, h1, h2, h3, h4, h5, h6 {{ font-family: {_FONT}; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}

/* 스트림릿 기본 크롬 감추기 (헤더는 남겨 사이드바 펼침 버튼을 살린다) */
[data-testid="stDecoration"], [data-testid="stToolbar"], #MainMenu, footer {{ display: none; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* 배경 그라디언트 메쉬 — 다크/라이트 위에 은은한 컬러 워시 */
.stApp {{
  background:
    radial-gradient(1100px 560px at 12% -8%, color-mix(in srgb, var(--a1) 14%, transparent), transparent 60%),
    radial-gradient(920px 480px at 100% -6%, color-mix(in srgb, var(--a2) 12%, transparent), transparent 55%);
  background-attachment: fixed;
}}
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1180px; }}

/* 사이드바 — 유리판 느낌 + 브랜드 워드마크 */
[data-testid="stSidebar"] {{
  border-right: 1px solid var(--line);
  background: color-mix(in srgb, var(--a1) 5%, transparent);
  backdrop-filter: blur(8px);
}}
.rlm-brand {{ display: flex; align-items: center; gap: 9px; padding: 4px 2px 14px;
  font-weight: 800; font-size: 1.05rem; letter-spacing: -0.01em; }}
.rlm-brand .dot {{ width: 11px; height: 11px; border-radius: 4px;
  background: linear-gradient(135deg, var(--a1), var(--a2)); box-shadow: 0 0 14px var(--ring); }}
.rlm-brand .sub {{ font-weight: 500; opacity: 0.5; font-size: 0.82rem; }}

/* 히어로 헤더 — 유리 카드 + 그라디언트 제목 */
.rlm-hero {{
  position: relative; border-radius: 22px; padding: 28px 32px; margin-bottom: 16px;
  border: 1px solid color-mix(in srgb, var(--a1) 30%, transparent);
  background:
    radial-gradient(130% 150% at 0% 0%, color-mix(in srgb, var(--a1) 20%, transparent), transparent 58%),
    radial-gradient(130% 150% at 100% 0%, color-mix(in srgb, var(--a2) 18%, transparent), transparent 55%);
  box-shadow: 0 22px 48px -30px color-mix(in srgb, var(--a1) 70%, transparent);
  backdrop-filter: blur(6px);
}}
.rlm-hero h1 {{
  font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 6px 0;
  background: linear-gradient(92deg, var(--a1), var(--a2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.rlm-hero p {{ margin: 0; opacity: 0.72; font-size: 0.96rem; line-height: 1.55; }}
.rlm-chip {{
  display: inline-flex; align-items: center; gap: 6px; margin-top: 15px;
  font-size: 0.72rem; padding: 4px 12px; border-radius: 999px; opacity: 0.9;
  border: 1px solid color-mix(in srgb, var(--a1) 35%, transparent);
  background: color-mix(in srgb, var(--a1) 10%, transparent);
}}

/* 스탯 카드 */
.rlm-stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 8px; }}
.rlm-stat {{
  flex: 1; min-width: 116px; border-radius: 16px; padding: 16px 18px;
  border: 1px solid var(--line);
  background: color-mix(in srgb, var(--a1) 5%, transparent);
  transition: transform .14s ease, border-color .14s ease, box-shadow .14s ease;
}}
.rlm-stat:hover {{ transform: translateY(-3px);
  border-color: color-mix(in srgb, var(--a1) 45%, transparent);
  box-shadow: 0 16px 30px -22px var(--ring); }}
.rlm-stat .v {{ font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }}
.rlm-stat .l {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.6; margin-top: 4px; }}
.rlm-stat.hl {{ border-color: color-mix(in srgb, var(--a1) 40%, transparent);
  background: color-mix(in srgb, var(--a1) 11%, transparent); }}
.rlm-stat.hl .v {{
  background: linear-gradient(92deg, var(--a1), var(--a2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}

/* 판정 배지 */
.rlm-badge {{ padding: 2px 10px; border-radius: 999px; font-size: 0.74rem; font-weight: 700; white-space: nowrap; }}
.rlm-badge.correct   {{ background: #10b98122; color: #10b981; }}
.rlm-badge.partial   {{ background: #f59e0b22; color: #f59e0b; }}
.rlm-badge.incorrect {{ background: #ef444422; color: #ef4444; }}

/* 위젯 라운드/테두리 정돈 */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="select"] > div {{ border-radius: 10px !important; }}
[data-testid="stExpander"] {{ border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }}
[data-testid="stExpander"] summary {{ font-weight: 600; }}
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

/* Primary 버튼 그라디언트 */
.stButton button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background: linear-gradient(92deg, var(--a1), var(--a2)) !important;
  border: none !important; font-weight: 700 !important; border-radius: 12px !important;
  box-shadow: 0 10px 24px -8px var(--ring);
}}
.stButton button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
  filter: brightness(1.08); transform: translateY(-1px); }}
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


def sidebar_brand(active: str) -> None:
    """사이드바 최상단 워드마크 — 앱 아이덴티티를 준다."""
    st.sidebar.markdown(
        f'<div class="rlm-brand"><span class="dot"></span>mini-RLM'
        f'<span class="sub">· {active}</span></div>',
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
