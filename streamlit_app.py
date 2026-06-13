"""RLM 플레이그라운드 — 임의의 context와 질문을 넣고 RLM이 푸는 과정을 실시간으로 본다.

실행: streamlit run streamlit_app.py
⚠️ 모델이 생성한 Python 코드를 in-process exec()로 실행한다(샌드박스 없음). 신뢰 환경 한정.
"""
import streamlit as st
from dotenv import load_dotenv

from app_trace import format_update
from demo_tickets import TICKET_QUESTION, make_tickets
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
            for e in entries:
                if e.kind == "model":
                    with st.chat_message("assistant"):
                        st.markdown(f"**🧠 턴 {e.turn}**")
                        if e.text:
                            st.markdown(e.text)
                        for block in e.code_blocks:
                            st.code(block, language="python")
                elif e.kind == "exec":
                    with st.chat_message("user"):
                        st.markdown(f"**💻 턴 {e.turn} 실행 결과**")
                        st.code(e.text or "(출력 없음)")
            if maybe_final is not None:
                final_answer = maybe_final
    except Exception as e:  # noqa: BLE001 - 데모이므로 오류를 화면에 그대로 노출
        st.exception(e)
        return

    st.subheader("결과")
    if final_answer is None:
        st.info("최종 답이 제출되지 않았습니다(턴 소진). max_iterations를 늘려보세요.")
        return
    st.success(f"모델 답: {final_answer}")

    expected = st.session_state.get("sample_expected")
    if expected is not None and question == TICKET_QUESTION:
        refund, delay = expected
        normalized = final_answer.replace(" ", "")
        ok = f"환불={refund}" in normalized and f"배송지연={delay}" in normalized
        st.write(f"정답: 환불={refund}, 배송지연={delay} — {'✅ 일치' if ok else '❌ 불일치'}")


def main():
    st.set_page_config(page_title="RLM 플레이그라운드", page_icon="🧠", layout="wide")
    st.title("🧠 Recursive Language Model 플레이그라운드")
    st.caption(
        "임의의 긴 context와 질문을 넣으면, 모델이 context를 REPL 변수로만 두고 "
        "코드를 생성·실행하며 답을 쌓아 올립니다. 그 과정을 턴별로 보여줍니다."
    )
    st.warning(
        "모델이 생성한 Python 코드를 in-process `exec()`로 실행합니다(샌드박스 없음). "
        "신뢰할 수 있는 입력·본인 머신에서만 사용하세요.",
        icon="⚠️",
    )

    st.session_state.setdefault("context", "")
    st.session_state.setdefault("question", "")
    st.session_state.setdefault("sample_expected", None)

    settings = get_settings()
    api_key_present = bool(settings.openrouter_api_key)

    with st.sidebar:
        st.header("설정")
        max_iterations = st.slider("max_iterations", 4, 20, 12)
        max_depth = st.slider("max_depth", 0, 2, 1)
        root_model = st.text_input("root 모델", settings.rlm_root_model)
        sub_model = st.text_input("sub 모델", settings.rlm_sub_model)
        if api_key_present:
            st.success("OPENROUTER_API_KEY 감지됨")
        else:
            st.error("OPENROUTER_API_KEY 없음 — .env 또는 환경변수에 설정하세요.")

    with st.expander("샘플 데이터 채우기", expanded=False):
        col1, col2 = st.columns(2)
        n = col1.number_input("티켓 수", 10, 10_000_000, 60)
        seed = col2.number_input("seed", 0, 9999, 42)
        if st.button("지저분한 티켓으로 채우기"):
            ctx, refund, delay = make_tickets(int(n), int(seed))
            st.session_state.context = ctx
            st.session_state.question = TICKET_QUESTION
            st.session_state.sample_expected = (refund, delay)

    uploaded = st.file_uploader(
        "context 파일 업로드(텍스트)", type=["txt", "md", "csv", "json", "log"]
    )
    if uploaded is not None:
        st.session_state.context = uploaded.getvalue().decode("utf-8", errors="replace")
        st.session_state.sample_expected = None

    context = st.text_area("context", key="context", height=240)
    st.caption(
        f"context 길이: {len(context)}자 — 모델에게는 이 길이 등 메타데이터만 전달됩니다."
    )
    question = st.text_area("질문(question)", key="question", height=100)

    run_clicked = st.button("▶ RLM 실행", type="primary", disabled=not api_key_present)
    if run_clicked:
        if not context.strip() or not question.strip():
            st.warning("context와 질문을 모두 입력하세요.")
            return
        _run_rlm(context, question, root_model, sub_model, max_depth, max_iterations)


if __name__ == "__main__":
    main()
