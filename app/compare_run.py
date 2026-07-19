"""비교를 문항별로 실행·채점하며 화면용 이벤트를 내는 순수 오케스트레이션.

eval_run.py와 대칭. solver.solve()로 각 시스템을 돌려(트레이스 포함) 문항별 record를 만들고
끝에서 aggregate_compare로 집계한다. streamlit·네트워크 비의존(StubSolver/FakeJudge로 테스트).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from eval.compare import run_item, aggregate_compare, to_compare_payload

__all__ = ["CompareEvent", "run_compare_stream", "to_compare_payload"]


@dataclass
class CompareEvent:
    kind: str                          # "item_done" | "run_done"
    item: Optional[object] = None
    record: Optional[dict] = None      # kind=="item_done"
    aggregate: Optional[dict] = None   # kind=="run_done"
    records: list = field(default_factory=list)


def run_compare_stream(items, context, solvers, judge_llm, *,
                       question_field: str = "question") -> Iterator[CompareEvent]:
    names = [s.name for s in solvers]
    records = []
    for item in items:
        rec = run_item(item, context, solvers, judge_llm, question_field)
        records.append(rec)
        yield CompareEvent("item_done", item=item, record=rec)
    yield CompareEvent("run_done", aggregate=aggregate_compare(records, names),
                       records=records)
