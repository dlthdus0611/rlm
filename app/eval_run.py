"""테스트셋을 문항별로 스트리밍 실행·채점해 화면용 이벤트를 내는 순수 오케스트레이션.

streamlit·네트워크 비의존 — trace.py와 대칭. 문항마다 build_rlm_graph().stream()을
돌려 트레이스를 뽑고(trace.format_update), 끝에서 harness.judge로 채점한다.
LLM은 .invoke()/.batch()를 가진 Runnable로 주입하므로 FakeChat으로 네트워크 없이 테스트 가능.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from rlm.graph import build_rlm_graph, recursion_limit_for
from app.trace import format_update
# to_payload은 eval 계층(harness)에 있고, 여기서 재노출해 페이지가 한 곳에서 import하게 한다.
from eval.harness import (
    QAItem, EvalResult, Verdict, judge, aggregate, question_of, to_payload,
)

__all__ = ["EvalEvent", "run_eval_stream", "to_payload"]


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
    # 그래프는 문항 간 상태를 갖지 않으므로(문항마다 setup에서 새 REPL) 한 번만 컴파일해 재사용한다.
    graph = build_rlm_graph(root_llm, sub_llm, max_depth, max_iterations)
    config = {"recursion_limit": recursion_limit_for(max_iterations)}
    results: list[EvalResult] = []
    for item in items:
        question = question_of(item, question_field)
        final_answer = None
        turn = 0
        try:
            for update in graph.stream(
                {"question": question, "context": context, "depth": 0},
                config=config, stream_mode="updates",
            ):
                entries, turn, maybe_final = format_update(update, turn)
                if entries:
                    yield EvalEvent("trace", item=item, entries=entries)
                if maybe_final is not None:
                    final_answer = maybe_final
        except Exception as exc:  # noqa: BLE001 - 문항 단위 실패는 기록하고 계속
            result = EvalResult(item, None, turn, Verdict("incorrect", "실행 오류"),
                                error=str(exc))
        else:
            verdict = judge(question, item.answer, final_answer, judge_llm)
            result = EvalResult(item, final_answer, turn, verdict)
        results.append(result)
        yield EvalEvent("item_done", item=item, result=result)
    yield EvalEvent("run_done", aggregate=aggregate(results), results=results)
