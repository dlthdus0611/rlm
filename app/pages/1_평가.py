"""RLM 평가 페이지 — data/의 QA 테스트셋을 골라 라이브로 실행·채점한다.

멀티페이지: streamlit이 app/pages/를 자동 인식한다. 메인은 playground.py(플레이그라운드).
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
from eval.datasets import CONTEXT_PATH, TESTSETS
from eval.harness import load_testset, question_of, select_items
from rlm import make_llm
from rlm.config import get_settings

load_dotenv()

# 공유 경로(eval.datasets) 위에 화면용 표시 라벨만 얹는다: 표시명 → 짧은 이름.
SETS = {
    "single (qa_testset, 100)": "single",
    "cross (qa_crosssection, 30)": "cross",
    "both": "both",
}
_DIFF_ORDER = ["low", "medium", "high", "expert"]
# 모델 설정은 UI에서 숨기고 환경변수/기본값과 아래 상수를 쓴다.
MAX_ITERATIONS = 12
MAX_DEPTH = 1


@st.cache_data(show_spinner=False)
def _load_items(set_label):
    """테스트셋을 로드한다. set_label(표시명)당 한 번만 파싱하고 이후 rerun은 캐시에서."""
    items = []
    for p in TESTSETS[SETS[set_label]]:
        items += load_testset(p)
    return items


def _render_live(box, counts, done, total):
    """실행 중 실시간 집계 카드를 갱신한다."""
    box.markdown(ui.stat_cards([
        ("진행", f"{done}/{total}", True),
        ("✅ correct", counts["correct"], False),
        ("🟡 partial", counts["partial"], False),
        ("❌ incorrect", counts["incorrect"], False),
    ]), unsafe_allow_html=True)


def _render_summary(overall):
    """최종 종합 지표를 스탯 카드로."""
    st.markdown(ui.stat_cards([
        ("종합 SCORE", f"{overall['score']:.3f}", True),
        ("✅ correct", overall["correct"], False),
        ("🟡 partial", overall["partial"], False),
        ("❌ incorrect", overall["incorrect"], False),
        ("평균 turns", overall["avg_turns"], False),
    ]), unsafe_allow_html=True)


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


def _item_expander(r, question_field, entries):
    """문항 하나의 드릴다운(판정 배지 + 추론 트레이스)."""
    icon = ui.VERDICT_ICON.get(r.verdict.label, "·")
    head = f"{icon}  {r.item.id}  ·  {r.item.difficulty}  ·  turns {r.turns}"
    with st.expander(head):
        st.markdown(
            f'{ui.verdict_badge(r.verdict.label)}'
            f'<span style="opacity:.65"> &nbsp;{r.verdict.reason}</span>',
            unsafe_allow_html=True)
        st.markdown(f"**질문**  {question_of(r.item, question_field)}")
        st.markdown(f"**정답**  {r.item.answer}")
        model_ans = r.model_answer if r.model_answer is not None else "_(미제출)_"
        st.markdown(f"**모델답**  {model_ans}")
        if r.error:
            st.error(r.error)
        st.divider()
        st.caption("추론 트레이스")
        ui.render_trace(entries)


def _run(items, context, cfg):
    root_llm = make_llm(cfg["root_model"])
    sub_llm = make_llm(cfg["sub_model"])
    judge_llm = make_llm(cfg["judge_model"])

    total = len(items)
    st.subheader("진행 상황")
    progress = st.progress(0.0, text=f"0/{total} 채점 중")
    live = st.empty()
    counts = {"correct": 0, "partial": 0, "incorrect": 0}
    _render_live(live, counts, 0, total)

    st.subheader("문항별 결과")
    done = 0
    traces = {}   # item.id -> 누적 TraceEntry

    for ev in run_eval_stream(
        items, context, root_llm, sub_llm, judge_llm,
        max_depth=cfg["max_depth"], max_iterations=cfg["max_iterations"],
        question_field=cfg["question_field"],
    ):
        if ev.kind == "trace":
            traces.setdefault(ev.item.id, []).extend(ev.entries)
        elif ev.kind == "item_done":
            r = ev.result
            counts[r.verdict.label] += 1
            done += 1
            progress.progress(done / total, text=f"{done}/{total} 채점 중")
            _render_live(live, counts, done, total)
            _item_expander(r, cfg["question_field"], traces.get(r.item.id, []))
        elif ev.kind == "run_done":
            progress.progress(1.0, text=f"{total}/{total} 완료 🎉")
            st.divider()
            st.subheader("종합")
            _render_summary(ev.aggregate["overall"])
            _difficulty_table(ev.aggregate["by_difficulty"])

            payload = to_payload(cfg, ev.aggregate, ev.results)
            st.download_button(
                "결과 JSON 다운로드", json.dumps(payload, ensure_ascii=False, indent=2),
                file_name="eval_results.json", mime="application/json",
                icon=":material/download:")


def main():
    ui.hero("📊", "RLM 평가",
            "data/의 QA 테스트셋을 난이도별로 골라 라이브로 실행·채점하고 결과를 확인합니다.",
            chip="⚠️ in-process exec() · 실제 OpenRouter 호출 · 신뢰 환경 전용")

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
            "root_model": settings.rlm_root_model, "sub_model": settings.rlm_sub_model,
            "judge_model": settings.rlm_sub_model,
            "max_iterations": MAX_ITERATIONS, "max_depth": MAX_DEPTH,
        }
        _run(items, context, cfg)
    else:
        picked = select_items(items_all, n=int(n), seed=int(seed), difficulties=difficulties)
        ui.empty_state(
            "▶ 평가 실행을 누르면 여기에 진행 상황과 결과가 나타납니다",
            [
                f"선택: <b>{set_label}</b> · 난이도 {', '.join(difficulties) or '(없음)'} · "
                f"<b>{len(picked)}문항</b> (seed {int(seed)} 고정 샘플링)",
                "문항마다 RLM이 삼성 사업보고서를 context로 두고 코드를 생성·실행해 답하고, "
                "그 답을 judge 모델이 정답과 대조해 채점합니다.",
                "실행 중 실시간 정오답 카운트 → 문항별 판정·추론 트레이스 → 종합·난이도별 집계 "
                "→ 결과 JSON 다운로드.",
                "각 문항이 RLM 전체 실행 + judge 1회라 문항 수에 비례해 시간·비용이 듭니다.",
            ],
        )


if __name__ == "__main__":
    main()
