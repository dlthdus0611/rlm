"""RLM 평가 계층 — 코어 rlm 패키지를 소비해 정답 정확도를 측정한다.

rlm(실행 엔진)과 분리된 응용 계층. harness는 순수 로직(네트워크 불필요),
runner는 실제 OpenRouter 호출 CLI(`python -m eval`).
"""
from .harness import (
    QAItem, Verdict, EvalResult,
    load_testset, select_items, judge, run_one, aggregate,
)

__all__ = [
    "QAItem", "Verdict", "EvalResult",
    "load_testset", "select_items", "judge", "run_one", "aggregate",
]
