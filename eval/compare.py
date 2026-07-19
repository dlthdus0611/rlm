"""RLM vs RAG 비교 하니스 — 동일 문항을 등록된 solver 전부에 통과시키고 동일 judge로 채점.

streamlit 비의존 순수 로직. solver는 SolverOutput을 내는 것이면 무엇이든(테스트는 StubSolver).

실행 예:
  python -m eval.compare --systems rlm,rag --set single --n 10
  python -m eval.compare --systems rlm,rag --set cross --n 5 --question-field question_textbook

⚠️ 실제 OpenRouter(+RAG는 OpenAI 임베딩) 호출. .env에 OPENROUTER_API_KEY·OPENAI_API_KEY 필요.
"""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from rlm.config import get_settings
from rlm.llm import make_llm
from rag.embeddings import make_embeddings

from .datasets import CONTEXT_PATH, TESTSETS as SETS
from .harness import load_testset, select_items, judge, question_of, gold_evidence
from .systems import build_solvers


def _evidence_hit(passages, evidence) -> bool:
    joined = "\n".join(getattr(p, "text", "") for p in passages)
    return any(ev and ev in joined for ev in evidence)


def run_item(item, context, solvers, judge_llm, question_field) -> dict:
    q = question_of(item, question_field)
    evidence = gold_evidence(item)
    per = {}
    for s in solvers:
        out = s.solve(q, context)
        verdict = judge(q, item.answer, out.answer, judge_llm)
        per[s.name] = {
            "label": verdict.label, "reason": verdict.reason,
            "answer": out.answer, "usage": out.usage, "latency_s": out.latency_s,
            "evidence_hit": _evidence_hit(out.extra.get("passages", []), evidence),
        }
    return {"item": item, "per_system": per}


def _bucket_system(records, name) -> dict:
    rows = [r["per_system"][name] for r in records if name in r["per_system"]]
    total = len(rows)
    correct = sum(1 for x in rows if x["label"] == "correct")
    partial = sum(1 for x in rows if x["label"] == "partial")
    incorrect = sum(1 for x in rows if x["label"] == "incorrect")
    score = (correct + 0.5 * partial) / total if total else 0.0
    hits = sum(1 for x in rows if x["evidence_hit"])

    def avg(f):
        return round(sum(f(x) for x in rows) / total, 2) if total else 0.0

    return {
        "total": total, "correct": correct, "partial": partial, "incorrect": incorrect,
        "score": round(score, 3),
        "avg_input_tokens": avg(lambda x: x["usage"].input_tokens),
        "avg_output_tokens": avg(lambda x: x["usage"].output_tokens),
        "avg_latency_s": avg(lambda x: x["latency_s"]),
        "evidence_hit_rate": round(hits / total, 3) if total else 0.0,
    }


def aggregate_compare(records, system_names) -> dict:
    return {name: _bucket_system(records, name) for name in system_names}


def to_compare_payload(config, agg, records) -> dict:
    systems = {}
    for name, bucket in agg.items():
        systems[name] = {
            "aggregate": bucket,
            "results": [
                {"id": r["item"].id, "difficulty": r["item"].difficulty,
                 "label": r["per_system"][name]["label"],
                 "answer": r["per_system"][name]["answer"],
                 "evidence_hit": r["per_system"][name]["evidence_hit"],
                 "input_tokens": r["per_system"][name]["usage"].input_tokens,
                 "output_tokens": r["per_system"][name]["usage"].output_tokens,
                 "latency_s": r["per_system"][name]["latency_s"]}
                for r in records if name in r["per_system"]
            ],
        }
    return {
        "config": config,
        "systems": systems,
        "items": [{"id": r["item"].id, "difficulty": r["item"].difficulty,
                   "question": r["item"].question, "gold": r["item"].answer}
                  for r in records],
    }


def _print_table(agg: dict) -> None:
    print("\n===== RLM vs RAG =====")
    print(f"{'system':<6} {'score':>6} {'c/p/i':>9} {'in_tok':>8} {'out_tok':>8} "
          f"{'lat_s':>7} {'ev_hit':>7}")
    for name, b in agg.items():
        cpi = f"{b['correct']}/{b['partial']}/{b['incorrect']}"
        print(f"{name:<6} {b['score']:>6} {cpi:>9} "
              f"{b['avg_input_tokens']:>8} {b['avg_output_tokens']:>8} "
              f"{b['avg_latency_s']:>7} {b['evidence_hit_rate']:>7}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="RLM vs RAG 비교")
    parser.add_argument("--systems", default="rlm,rag")
    parser.add_argument("--set", choices=SETS, default="single")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", action="append", default=None,
                        choices=["low", "medium", "high", "expert"])
    parser.add_argument("--question-field", choices=["question", "question_textbook"],
                        default="question")
    parser.add_argument("--root-model", default=None)
    parser.add_argument("--sub-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--out", default="data/compare_results.json")
    args = parser.parse_args()

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY 없음 — .env에 설정하세요.")
    if "rag" in names and not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY 없음 — RAG 임베딩에 필요합니다.")

    root_model = args.root_model or settings.rlm_root_model
    sub_model = args.sub_model or settings.rlm_sub_model
    judge_model = args.judge_model or sub_model

    with open(CONTEXT_PATH, encoding="utf-8") as f:
        context = f.read()
    items = []
    for path in SETS[args.set]:
        items += load_testset(path)
    items = select_items(items, n=args.n, seed=args.seed, difficulties=args.difficulty)

    root_llm, sub_llm = make_llm(root_model), make_llm(sub_model)
    judge_llm = make_llm(judge_model)
    embeddings = make_embeddings() if "rag" in names else None
    solvers = build_solvers(names, root_llm=root_llm, sub_llm=sub_llm,
                            embeddings=embeddings, settings=settings,
                            max_depth=args.max_depth, max_iterations=args.max_iterations)

    print(f"context {len(context)}자 | {len(items)}문항 | systems={names}")
    records = []
    for idx, item in enumerate(items, 1):
        rec = run_item(item, context, solvers, judge_llm, args.question_field)
        flags = " ".join(f"{n}={rec['per_system'][n]['label']}" for n in names)
        print(f"[{idx}/{len(items)}] {item.id} ({item.difficulty}) {flags}")
        records.append(rec)

    agg = aggregate_compare(records, names)
    _print_table(agg)
    payload = to_compare_payload(
        {"set": args.set, "n": args.n, "seed": args.seed, "systems": names,
         "difficulty": args.difficulty, "question_field": args.question_field,
         "root_model": root_model, "sub_model": sub_model, "judge_model": judge_model},
        agg, records)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
