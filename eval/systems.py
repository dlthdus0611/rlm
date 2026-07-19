"""RLM·RAG를 바꿔 끼우는 공통 플러그인 구조.

두 시스템의 공통 껍데기 — (질문, context)+주입 LLM → 계측된 답(토큰·지연·트레이스) — 를
BaseSolver가 소유한다. 내부 제어 흐름(RLM=에이전트 루프, RAG=선형 파이프라인)만 _run으로 갈린다.
계측은 여기 한 곳에만 있어 시스템에 중복되지 않는다.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler

from rlm.graph import build_rlm_graph, recursion_limit_for
from app.trace import format_update
from rag.api import build_retriever
from rag.pipeline import answer as rag_answer


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class SolverOutput:
    answer: Optional[str]
    usage: Usage
    latency_s: float
    trace: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class _UsageCollector(BaseCallbackHandler):
    """모든 하위 LLM 호출의 usage_metadata를 합산(가짜 모델은 미발화 → 0)."""
    def __init__(self):
        self.input = 0
        self.output = 0

    def on_llm_end(self, response, **kwargs):
        for gens in getattr(response, "generations", []) or []:
            for gen in gens:
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None) if msg is not None else None
                if um:
                    self.input += um.get("input_tokens", 0)
                    self.output += um.get("output_tokens", 0)
        out = getattr(response, "llm_output", None) or {}
        tu = out.get("token_usage") or out.get("usage") or {}
        self.input += tu.get("prompt_tokens", 0)
        self.output += tu.get("completion_tokens", 0)


class BaseSolver:
    name: str = "base"

    def solve(self, question: str, context: str) -> SolverOutput:
        collector = _UsageCollector()
        t0 = time.perf_counter()
        try:
            answer, trace, extra = self._run(question, context, [collector])
        except Exception as exc:  # noqa: BLE001 - 문항 단위 실패는 기록하고 계속
            answer, trace, extra = None, [], {"error": str(exc)}
        latency = time.perf_counter() - t0
        usage = Usage(collector.input, collector.output, collector.input + collector.output)
        return SolverOutput(answer, usage, latency, trace, extra)

    def _run(self, question, context, callbacks):
        raise NotImplementedError


class RlmSolver(BaseSolver):
    name = "rlm"

    def __init__(self, root_llm, sub_llm, *, max_depth: int = 1, max_iterations: int = 10):
        self.graph = build_rlm_graph(root_llm, sub_llm, max_depth, max_iterations)
        self.max_iterations = max_iterations

    def _run(self, question, context, callbacks):
        config = {"callbacks": callbacks,
                  "recursion_limit": recursion_limit_for(self.max_iterations)}
        turn, final, trace = 0, None, []
        for update in self.graph.stream(
            {"question": question, "context": context, "depth": 0},
            config=config, stream_mode="updates",
        ):
            entries, turn, maybe = format_update(update, turn)
            trace.extend(entries)
            if maybe is not None:
                final = maybe
        return final, trace, {"turns": turn}


class RagSolver(BaseSolver):
    name = "rag"

    def __init__(self, root_llm, sub_llm, embeddings, settings, *, cache_dir=None):
        self.root_llm = root_llm
        self.sub_llm = sub_llm
        self.embeddings = embeddings
        self.settings = settings
        self.cache_dir = cache_dir
        self.use_hyde = None       # None이면 settings 기본값
        self.use_rerank = None
        self._built_key = None
        self._retriever = None

    def _retriever_for(self, context):
        key = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
        if key != self._built_key:
            self._retriever = build_retriever(
                context, self.sub_llm, self.embeddings, self.settings,
                cache_dir=self.cache_dir, use_hyde=self.use_hyde, use_rerank=self.use_rerank,
            )
            self._built_key = key
        return self._retriever

    def _run(self, question, context, callbacks):
        retriever = self._retriever_for(context)
        result = rag_answer(question, retriever, self.root_llm, callbacks=callbacks)
        trace = [{"kind": "retrieve", "passages": [p.text for p in result.passages]}]
        return result.answer, trace, {"passages": result.passages}


SYSTEMS = {"rlm": RlmSolver, "rag": RagSolver}


def build_solvers(names, *, root_llm, sub_llm, embeddings, settings,
                  max_depth: int = 1, max_iterations: int = 10) -> list[BaseSolver]:
    solvers = []
    for n in names:
        if n == "rlm":
            solvers.append(RlmSolver(root_llm, sub_llm, max_depth=max_depth,
                                     max_iterations=max_iterations))
        elif n == "rag":
            solvers.append(RagSolver(root_llm, sub_llm, embeddings, settings))
        else:
            raise ValueError(f"알 수 없는 시스템: {n} (가능: {list(SYSTEMS)})")
    return solvers
