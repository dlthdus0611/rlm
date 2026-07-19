"""RLM 플레이그라운드 — 임의의 context와 질문을 넣고 RLM이 푸는 과정을 실시간으로 본다.

실행: streamlit run app/playground.py
⚠️ 모델이 생성한 Python 코드를 in-process exec()로 실행한다(샌드박스 없음). 신뢰 환경 한정.
"""
import sys
from pathlib import Path

# streamlit은 스크립트 폴더(app/)만 sys.path에 넣으므로, 루트의 rlm·app 패키지를
# import하려면 프로젝트 루트를 직접 추가한다. (pytest 등 다른 진입점에선 이미 루트가 잡혀 무해)
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from dotenv import load_dotenv

from app import ui
from app.trace import format_update
from rlm import build_rlm_graph, make_llm
from rlm.config import get_settings

load_dotenv()


def _run_rlm(context, question, root_model, sub_model, max_depth, max_iterations):
    try:
        graph = build_rlm_graph(
            make_llm(root_model), make_llm(sub_model), max_depth, max_iterations
        )
    except RuntimeError as e:
        st.error(str(e))
        return

    st.subheader("추론 트레이스")
    turn = 0
    final_answer = None
    try:
        for update in graph.stream(
            {"question": question, "context": context, "depth": 0},
            config={"recursion_limit": 2 * max_iterations + 10},
            stream_mode="updates",
        ):
            entries, turn, maybe_final = format_update(update, turn)
            ui.render_trace(entries)
            if maybe_final is not None:
                final_answer = maybe_final
    except Exception as e:  # noqa: BLE001 - 데모이므로 오류를 화면에 그대로 노출
        st.exception(e)
        return

    st.divider()
    st.subheader("결과")
    if final_answer is None:
        st.info("최종 답이 제출되지 않았습니다(턴 소진). max_iterations를 늘려보세요.", icon="⏱️")
        return
    st.success(f"**모델 답**  {final_answer}", icon="🎯")


def main():
    ui.hero("🧠", "Recursive Language Model 플레이그라운드",
            "긴 context를 REPL 변수로만 두고, 모델이 코드를 생성·실행하며 답을 쌓는 과정을 턴별로 봅니다.",
            chip="⚠️ in-process exec() · 신뢰할 수 있는 입력·본인 머신 전용")

    st.session_state.setdefault("context", "")
    st.session_state.setdefault("question", "")

    settings = get_settings()
    api_key_present = bool(settings.openrouter_api_key)

    with st.sidebar:
        st.header("설정")
        max_iterations = st.slider("max_iterations", 4, 20, 12)
        max_depth = st.slider("max_depth", 0, 2, 1)
        with st.expander("모델 설정"):
            root_model = st.text_input("root 모델", settings.rlm_root_model)
            sub_model = st.text_input("sub 모델", settings.rlm_sub_model)
        ui.api_key_badge(api_key_present)

    uploaded = st.file_uploader(
        "context 파일 업로드(텍스트)", type=["txt", "md", "csv", "json", "log"]
    )
    if uploaded is not None:
        st.session_state.context = uploaded.getvalue().decode("utf-8", errors="replace")

    context = st.text_area("context", key="context", height=240)
    st.caption(
        f"context 길이: {len(context):,}자 — 모델에게는 이 길이 등 메타데이터만 전달됩니다."
    )
    question = st.text_area("질문(question)", key="question", height=100)

    if st.button("▶ RLM 실행", type="primary", disabled=not api_key_present,
                 use_container_width=True):
        if not context.strip() or not question.strip():
            st.warning("context와 질문을 모두 입력하세요.")
            return
        _run_rlm(context, question, root_model, sub_model, max_depth, max_iterations)


if __name__ == "__main__":
    main()
