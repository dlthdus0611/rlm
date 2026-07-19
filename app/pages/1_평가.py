"""RLM 평가 페이지 — data/의 QA 테스트셋을 골라 라이브로 실행·채점한다.

멀티페이지: streamlit이 app/pages/를 자동 인식한다. 메인은 streamlit_app.py(플레이그라운드).
⚠️ 실제 OpenRouter 호출 + 모델 생성 코드를 in-process exec()로 실행(샌드박스 없음). 신뢰 환경 한정.
"""
import json
import sys
from pathlib import Path

# streamlit은 스크립트 폴더만 sys.path에 넣으므로 루트를 직접 추가한다(메인 페이지와 동일).
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from dotenv import load_dotenv

from app import ui
from app.eval_run import run_eval_stream, to_payload
from eval.harness import load_testset, select_items
from rlm import make_llm
from rlm.config import get_settings

load_dotenv()

SETS = {
    "single (qa_testset, 100)": ["data/qa_testset.json"],
    "cross (qa_crosssection, 30)": ["data/qa_crosssection.json"],
    "both": ["data/qa_testset.json", "data/qa_crosssection.json"],
}
CONTEXT_PATH = "data/samsung_2023.txt"
_DIFF_ORDER = ["low", "medium", "high", "expert"]


def _load_items(set_label):
    items = []
    for p in SETS[set_label]:
        items += load_testset(p)
    return items


def _live_counts(box, counts, done, total):
    """실행 중 실시간 집계 카드 4장을 갱신한다."""
    with box.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("진행", f"{done}/{total}")
        c2.metric("✅ correct", counts["correct"])
        c3.metric("🟡 partial", counts["partial"])
        c4.metric("❌ incorrect", counts["incorrect"])


def _summary_cards(overall):
    """최종 종합 지표를 카드로."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("종합 score", f"{overall['score']:.3f}", border=True)
    c2.metric("✅ correct", overall["correct"], border=True)
    c3.metric("🟡 partial", overall["partial"], border=True)
    c4.metric("❌ incorrect", overall["incorrect"], border=True)
    c5.metric("평균 turns", overall["avg_turns"], border=True)


def _difficulty_table(by_difficulty):
    """난이도별 집계를 정돈된 표로."""
    rows = [{"난이도": d, **b} for d, b in by_difficulty.items()]
    if not rows:
        return
    st.markdown("###### 난이도별")
    st.dataframe(
        rows, use_container_width=True, hide_index=True,
        column_config={
            "난이도": st.column_config.TextColumn(width="small"),
            "score": st.column_config.ProgressColumn(
                "score", min_value=0.0, max_value=1.0, format="%.3f"),
            "total": "문항", "correct": "✅", "partial": "🟡",
            "incorrect": "❌", "errors": "오류", "avg_turns": "avg turns",
        },
    )


def _item_expander(r, cfg, entries):
    """문항 하나의 드릴다운(판정 요약 + 추론 트레이스)."""
    icon = ui.VERDICT_ICON.get(r.verdict.label, "·")
    head = f"{icon}  {r.item.id}  ·  {r.item.difficulty}  ·  turns {r.turns}"
    with st.expander(head):
        q = getattr(r.item, cfg["question_field"], "") or r.item.question
        st.markdown(f"**질문**  {q}")
        st.markdown(f"**정답**  {r.item.answer}")
        model_ans = r.model_answer if r.model_answer is not None else "_(미제출)_"
        st.markdown(f"**모델답**  {model_ans}")
        st.markdown(f"**judge**  `{r.verdict.label}` — {r.verdict.reason}")
        if r.error:
            st.error(r.error)
        st.divider()
        st.caption("추론 트레이스")
        ui.render_trace(entries)


def _run(items, context, cfg, judge_model):
    root_llm = make_llm(cfg["root_model"])
    sub_llm = make_llm(cfg["sub_model"])
    judge_llm = make_llm(judge_model)

    total = len(items)
    st.subheader("진행 상황")
    progress = st.progress(0.0, text=f"0/{total} 채점 중")
    live = st.empty()
    counts = {"correct": 0, "partial": 0, "incorrect": 0}
    _live_counts(live, counts, 0, total)

    st.subheader("문항별 결과")
    done = 0
    traces = {}   # item.id -> 누적 TraceEntry
    results = []

    for ev in run_eval_stream(
        items, context, root_llm, sub_llm, judge_llm,
        max_depth=cfg["max_depth"], max_iterations=cfg["max_iterations"],
        question_field=cfg["question_field"],
    ):
        if ev.kind == "trace":
            traces.setdefault(ev.item.id, []).extend(ev.entries)
        elif ev.kind == "item_done":
            r = ev.result
            results.append(r)
            counts[r.verdict.label] += 1
            done += 1
            progress.progress(done / total, text=f"{done}/{total} 채점 중")
            _live_counts(live, counts, done, total)
            _item_expander(r, cfg, traces.get(r.item.id, []))
        elif ev.kind == "run_done":
            progress.progress(1.0, text=f"{total}/{total} 완료 🎉")
            st.divider()
            st.subheader("종합")
            _summary_cards(ev.aggregate["overall"])
            _difficulty_table(ev.aggregate["by_difficulty"])

            payload = to_payload(
                {**cfg, "judge_model": judge_model}, ev.aggregate, ev.results,
                cfg["question_field"])
            st.download_button(
                "결과 JSON 다운로드", json.dumps(payload, ensure_ascii=False, indent=2),
                file_name="eval_results.json", mime="application/json", icon=":material/download:")


def main():
    ui.page_header("📊", "RLM 평가",
                   "data/의 QA 테스트셋을 골라 라이브로 실행·채점하고 결과를 확인합니다.")
    ui.security_note()

    settings = get_settings()
    api_key_present = bool(settings.openrouter_api_key)

    with st.sidebar:
        st.header("설정")
        set_label = st.selectbox("테스트셋", list(SETS))
        items_all = _load_items(set_label)
        diffs_present = sorted({it.difficulty for it in items_all}, key=_DIFF_ORDER.index)
        difficulties = [d for d in diffs_present if st.checkbox(f"난이도: {d}", True)]
        n = st.number_input("문항 수(n, seed 고정 샘플링)", 1, len(items_all),
                            min(5, len(items_all)))
        seed = st.number_input("seed", 0, 9999, 42)
        question_field = st.radio(
            "질문 형태", ["question", "question_textbook"],
            format_func=lambda k: "대충질문" if k == "question" else "정석")
        with st.expander("모델 설정"):
            root_model = st.text_input("root 모델", settings.rlm_root_model)
            sub_model = st.text_input("sub 모델", settings.rlm_sub_model)
            judge_model = st.text_input("judge 모델", settings.rlm_sub_model)
            max_iterations = st.slider("max_iterations", 4, 20, 12)
            max_depth = st.slider("max_depth", 0, 2, 1)
        ui.api_key_badge(api_key_present)

    if n > 20:
        st.info(f"{n}문항 × (RLM 전체 실행 + judge)라 시간·비용이 큽니다. 작게 시작해 보세요.",
                icon="⏱️")

    if st.button("▶ 평가 실행", type="primary", disabled=not api_key_present,
                 use_container_width=True):
        if not difficulties:
            st.warning("난이도를 하나 이상 선택하세요.")
            return
        items = select_items(items_all, n=int(n), seed=int(seed), difficulties=difficulties)
        if not items:
            st.warning("선택한 조건에 해당하는 문항이 없습니다.")
            return
        with open(CONTEXT_PATH, encoding="utf-8") as f:
            context = f.read()
        cfg = {
            "set": set_label, "n": int(n), "seed": int(seed),
            "difficulty": difficulties, "question_field": question_field,
            "root_model": root_model, "sub_model": sub_model,
            "max_iterations": max_iterations, "max_depth": max_depth,
        }
        _run(items, context, cfg, judge_model)


if __name__ == "__main__":
    main()
