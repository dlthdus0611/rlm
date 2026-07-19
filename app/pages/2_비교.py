"""RLM vs RAG 비교 페이지 — 같은 문서·테스트셋·judge로 두 시스템을 나란히 실행·채점한다.

멀티페이지: streamlit이 app/pages/를 자동 인식한다. 공정성 계약(동일 문항·동일 judge·동일 생성 모델)
위에서 정확도·비용(토큰)·지연(시간)·근거적중을 대조한다.
⚠️ 실제 OpenRouter(+RAG는 OpenAI 임베딩) 호출 + 모델 생성 코드를 in-process exec() 실행. 신뢰 환경 한정.
"""
import json
import sys
from pathlib import Path

# streamlit은 스크립트 폴더만 sys.path에 넣으므로 루트를 직접 추가한다(다른 페이지와 동일).
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from dotenv import load_dotenv

from app import ui
from app.compare_run import run_compare_stream, to_compare_payload
from eval.datasets import CONTEXT_PATH, TESTSETS
from eval.harness import load_testset, question_of, select_items
from eval.systems import build_solvers
from rlm import make_llm
from rlm.config import get_settings

load_dotenv()

SETS = {
    "single (qa_testset, 100)": "single",
    "cross (qa_crosssection, 30)": "cross",
    "both": "both",
}
_DIFF_ORDER = ["low", "medium", "high", "expert"]
_SYS_LABEL = {"rlm": "RLM", "rag": "RAG"}
MAX_ITERATIONS = 12
MAX_DEPTH = 1


@st.cache_data(show_spinner=False)
def _load_items(set_label):
    items = []
    for p in TESTSETS[SETS[set_label]]:
        items += load_testset(p)
    return items


def _summary_table(agg, names):
    """시스템별 종합 지표 대조표."""
    rows = []
    for n in names:
        b = agg[n]
        rows.append({
            "시스템": _SYS_LABEL.get(n, n), "score": b["score"],
            "✅": b["correct"], "🟡": b["partial"], "❌": b["incorrect"],
            "평균 in토큰": b["avg_input_tokens"], "평균 out토큰": b["avg_output_tokens"],
            "평균 지연(s)": b["avg_latency_s"], "근거적중률": b["evidence_hit_rate"],
        })
    st.dataframe(
        rows, use_container_width=True, hide_index=True,
        column_config={
            "시스템": st.column_config.TextColumn(width="small"),
            "score": st.column_config.ProgressColumn(
                "score", min_value=0.0, max_value=1.0, format="%.3f"),
            "근거적중률": st.column_config.ProgressColumn(
                "근거적중률", min_value=0.0, max_value=1.0, format="%.3f"),
        },
    )


def _system_panel(name, ps):
    """드릴다운 한 시스템 열 — 판정·모델답·트레이스/passage."""
    st.markdown(f"**{_SYS_LABEL.get(name, name)}**")
    st.markdown(
        f'{ui.verdict_badge(ps["label"])}'
        f'<span style="opacity:.65"> &nbsp;{ps["reason"]}</span>',
        unsafe_allow_html=True)
    st.markdown(f"모델답 · {ps['answer'] if ps['answer'] is not None else '_(미제출)_'}")
    u = ps["usage"]
    st.caption(f"토큰 in {u.input_tokens} / out {u.output_tokens} · 지연 {ps['latency_s']:.2f}s")
    if ps.get("error"):
        st.error(ps["error"])
    if name == "rag":
        hit = "✅ 근거 포함" if ps["evidence_hit"] else "❌ 근거 미포함"
        st.caption(f"검색 passage {len(ps.get('passages', []))}개 · {hit}")
        for i, text in enumerate(ps.get("passages", []), 1):
            with st.expander(f"passage {i}"):
                st.write(text)
    else:
        st.caption("추론 트레이스")
        ui.render_trace(ps.get("trace", []))


def _item_expander(item, rec, names, question_field):
    per = rec["per_system"]
    icons = " ".join(ui.VERDICT_ICON.get(per[n]["label"], "·") for n in names)
    head = f"{icons}  {item.id}  ·  {item.difficulty}"
    with st.expander(head):
        st.markdown(f"**질문**  {question_of(item, question_field)}")
        st.markdown(f"**정답**  {item.answer}")
        st.divider()
        cols = st.columns(len(names))
        for col, n in zip(cols, names):
            with col:
                _system_panel(n, per[n])


def _run(items, context, names, cfg):
    root_llm = make_llm(cfg["root_model"])
    sub_llm = make_llm(cfg["sub_model"])
    judge_llm = make_llm(cfg["judge_model"])
    embeddings = None
    if "rag" in names:
        from rag.embeddings import make_embeddings
        embeddings = make_embeddings()
    settings = get_settings()
    solvers = build_solvers(names, root_llm=root_llm, sub_llm=sub_llm,
                            embeddings=embeddings, settings=settings,
                            max_depth=cfg["max_depth"], max_iterations=cfg["max_iterations"])

    total = len(items)
    st.subheader("진행 상황")
    progress = st.progress(0.0, text=f"0/{total} 실행 중")
    st.subheader("문항별 결과")
    done = 0
    for ev in run_compare_stream(items, context, solvers, judge_llm,
                                 question_field=cfg["question_field"]):
        if ev.kind == "item_done":
            done += 1
            progress.progress(done / total, text=f"{done}/{total} 실행 중")
            _item_expander(ev.item, ev.record, names, cfg["question_field"])
        elif ev.kind == "run_done":
            progress.progress(1.0, text=f"{total}/{total} 완료 🎉")
            st.divider()
            st.subheader("종합 — RLM vs RAG")
            _summary_table(ev.aggregate, names)
            payload = to_compare_payload(cfg, ev.aggregate, ev.records)
            st.download_button(
                "결과 JSON 다운로드", json.dumps(payload, ensure_ascii=False, indent=2),
                file_name="compare_results.json", mime="application/json",
                icon=":material/download:")


def main():
    ui.hero("⚖️", "RLM vs RAG 비교",
            "같은 문서·테스트셋·judge로 RLM과 강화 RAG를 나란히 실행해 "
            "정확도·비용(토큰)·지연을 대조합니다.",
            chip="⚠️ in-process exec() · 실제 OpenRouter+OpenAI 호출 · 신뢰 환경 전용")

    settings = get_settings()
    has_openrouter = bool(settings.openrouter_api_key)
    has_openai = bool(settings.openai_api_key)

    with st.sidebar:
        st.header("설정")
        set_label = st.selectbox("테스트셋", list(SETS))
        items_all = _load_items(set_label)
        diffs_present = sorted({it.difficulty for it in items_all}, key=_DIFF_ORDER.index)
        difficulties = [d for d in diffs_present if st.checkbox(f"난이도: {d}", True)]
        n = st.number_input("문항 수(n, seed 고정 샘플링)", 1, len(items_all),
                            min(5, len(items_all)))
        seed = st.number_input("seed", 0, 9999, 42)
        names = st.multiselect("비교 시스템", ["rlm", "rag"], default=["rlm", "rag"],
                               format_func=lambda k: _SYS_LABEL[k])
        question_field = st.radio(
            "질문 형태", ["question", "question_textbook"],
            format_func=lambda k: "대충질문" if k == "question" else "정석")
        ui.api_key_badge(has_openrouter)
        if "rag" in names and not has_openai:
            st.markdown('<span class="rlm-keybadge no">● OPENAI_API_KEY 없음 · RAG 임베딩 필요</span>',
                        unsafe_allow_html=True)

    ready = has_openrouter and (has_openai or "rag" not in names) and bool(names)

    if n > 20:
        st.info(f"{n}문항 × 시스템 {len(names)}개(각 실행 + judge)라 시간·비용이 큽니다.", icon="⏱️")

    if st.button("▶ 비교 실행", type="primary", disabled=not ready, use_container_width=True):
        if not difficulties:
            st.warning("난이도를 하나 이상 선택하세요.")
            return
        if not names:
            st.warning("비교할 시스템을 하나 이상 선택하세요.")
            return
        items = select_items(items_all, n=int(n), seed=int(seed), difficulties=difficulties)
        if not items:
            st.warning("선택한 조건에 해당하는 문항이 없습니다.")
            return
        with open(CONTEXT_PATH, encoding="utf-8") as f:
            context = f.read()
        cfg = {
            "set": set_label, "n": int(n), "seed": int(seed), "systems": names,
            "difficulty": difficulties, "question_field": question_field,
            "root_model": settings.rlm_root_model, "sub_model": settings.rlm_sub_model,
            "judge_model": settings.rlm_sub_model,
            "max_iterations": MAX_ITERATIONS, "max_depth": MAX_DEPTH,
        }
        _run(items, context, names, cfg)
    else:
        picked = select_items(items_all, n=int(n), seed=int(seed), difficulties=difficulties)
        ui.empty_state(
            "▶ 비교 실행을 누르면 여기에 RLM vs RAG 대조 결과가 나타납니다",
            [
                f"선택: <b>{set_label}</b> · 난이도 {', '.join(difficulties) or '(없음)'} · "
                f"<b>{len(picked)}문항</b> · 시스템 {', '.join(_SYS_LABEL[n] for n in names) or '(없음)'}",
                "같은 문항을 RLM(코드로 전체 문서 탐색)과 RAG(벡터 검색 top-k)에 각각 통과시키고 "
                "동일 judge로 채점합니다 — 검색 방식만 다르고 생성 모델·판정은 동일합니다.",
                "문항별로 두 시스템의 판정·모델답·추론 트레이스/검색 passage를 나란히 비교하고, "
                "종합에서 정확도·평균 토큰·평균 지연·근거적중률을 대조합니다.",
            ],
        )


if __name__ == "__main__":
    main()
