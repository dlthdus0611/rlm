"""테스트셋을 문항별로 스트리밍 실행·채점해 화면용 이벤트를 내는 순수 오케스트레이션.

streamlit·네트워크 비의존 — trace.py와 대칭. 문항마다 build_rlm_graph().stream()을
돌려 트레이스를 뽑고(trace.format_update), 끝에서 harness.judge로 채점한다.
LLM은 .invoke()/.batch()를 가진 Runnable로 주입하므로 FakeChat으로 네트워크 없이 테스트 가능.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from rlm.graph import build_rlm_graph
from app.trace import format_update
from eval.harness import QAItem, EvalResult, Verdict, judge, aggregate


@dataclass
class EvalEvent:
    kind: str                                     # "trace" | "item_done" | "run_done"
    item: Optional[QAItem] = None
    entries: list = field(default_factory=list)   # kind=="trace": TraceEntry 목록
    result: Optional[EvalResult] = None           # kind=="item_done"
    aggregate: Optional[dict] = None              # kind=="run_done"
    results: list = field(default_factory=list)   # kind=="run_done": EvalResult 전체


def run_eval_stream(items, context, root_llm, sub_llm, judge_llm, *,
                    max_depth: int = 1, max_iterations: int = 10,
                    question_field: str = "question") -> Iterator[EvalEvent]:
    """문항 리스트를 순서대로 실행·채점하며 EvalEvent를 yield한다.

    문항마다: trace(여러 개) → item_done. 전 문항 후: run_done(aggregate).
    문항 단위 예외는 잡아 EvalResult.error에 담고 계속한다(run_one과 동일 정책).
    """
    results: list[EvalResult] = []
    for item in items:
        question = getattr(item, question_field, "") or item.question
        final_answer = None
        turn = 0
        try:
            graph = build_rlm_graph(root_llm, sub_llm, max_depth, max_iterations)
            for update in graph.stream(
                {"question": question, "context": context, "depth": 0},
                config={"recursion_limit": 2 * max_iterations + 10},
                stream_mode="updates",
            ):
                entries, turn, maybe_final = format_update(update, turn)
                if entries:
                    yield EvalEvent("trace", item=item, entries=entries)
                if maybe_final is not None:
                    final_answer = maybe_final
        except Exception as exc:  # noqa: BLE001 - 문항 단위 실패는 기록하고 계속
            result = EvalResult(item, None, turn, Verdict("incorrect", "실행 오류"),
                                error=str(exc))
            results.append(result)
            yield EvalEvent("item_done", item=item, result=result)
            continue
        verdict = judge(question, item.answer, final_answer, judge_llm)
        result = EvalResult(item, final_answer, turn, verdict)
        results.append(result)
        yield EvalEvent("item_done", item=item, result=result)
    yield EvalEvent("run_done", aggregate=aggregate(results), results=results)


def to_payload(config: dict, agg: dict, results, question_field: str) -> dict:
    """다운로드용 JSON payload. runner.py의 저장 구조와 동일하게 맞춘다."""
    return {
        "config": config,
        "aggregate": agg,
        "results": [
            {
                "id": r.item.id, "difficulty": r.item.difficulty,
                "question": getattr(r.item, question_field, "") or r.item.question,
                "gold": r.item.answer, "model_answer": r.model_answer,
                "turns": r.turns, "label": r.verdict.label,
                "reason": r.verdict.reason, "error": r.error,
            }
            for r in results
        ],
    }
