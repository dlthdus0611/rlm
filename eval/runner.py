"""RLM 평가 CLI — 삼성 사업보고서 QA 테스트셋으로 RLM 정답 정확도를 측정한다.

실행 예:
  python -m eval --set single --n 10 --difficulty low --difficulty medium
  python -m eval --set cross --n 5 --question-field question_textbook

⚠️ 실제 OpenRouter 호출. .env 또는 환경변수에 OPENROUTER_API_KEY 필요.
모델이 생성한 코드를 in-process exec()로 실행한다(샌드박스 없음).
"""
import argparse
import json

from dotenv import load_dotenv

from rlm.config import get_settings
from rlm.llm import make_llm

from .harness import aggregate, load_testset, run_one, select_items

load_dotenv()

SETS = {
    "single": ["data/qa_testset.json"],
    "cross": ["data/qa_crosssection.json"],
    "both": ["data/qa_testset.json", "data/qa_crosssection.json"],
}
CONTEXT_PATH = "data/samsung_2023.txt"


def _print_summary(agg: dict) -> None:
    o = agg["overall"]
    print("\n===== 요약 =====")
    print(f"전체 {o['total']}문항 | score={o['score']} | "
          f"correct={o['correct']} partial={o['partial']} "
          f"incorrect={o['incorrect']} errors={o['errors']} | avg_turns={o['avg_turns']}")
    for diff, b in agg["by_difficulty"].items():
        print(f"  [{diff:<7}] {b['total']:>3}문항 | score={b['score']:<5} "
              f"c={b['correct']} p={b['partial']} i={b['incorrect']} e={b['errors']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RLM 평가 하니스")
    parser.add_argument("--set", choices=SETS, default="single", help="테스트셋")
    parser.add_argument("--n", type=int, default=None, help="샘플 수(미지정=전체)")
    parser.add_argument("--seed", type=int, default=42, help="샘플링 seed")
    parser.add_argument("--difficulty", action="append", default=None,
                        choices=["low", "medium", "high", "expert"],
                        help="난이도 필터(반복 지정 가능)")
    parser.add_argument("--question-field", choices=["question", "question_textbook"],
                        default="question", help="대충질문 vs 정석")
    parser.add_argument("--judge-model", default=None, help="채점 모델(기본 sub)")
    parser.add_argument("--root-model", default=None)
    parser.add_argument("--sub-model", default=None)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--out", default="data/eval_results.json")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY 없음 — .env 또는 환경변수에 설정하세요.")

    root_model = args.root_model or settings.rlm_root_model
    sub_model = args.sub_model or settings.rlm_sub_model
    judge_model = args.judge_model or sub_model

    with open(CONTEXT_PATH, encoding="utf-8") as f:
        context = f.read()

    items = []
    for path in SETS[args.set]:
        items += load_testset(path)
    items = select_items(items, n=args.n, seed=args.seed, difficulties=args.difficulty)
    print(f"context {len(context)}자 | {len(items)}문항 평가 시작 "
          f"(root={root_model}, sub={sub_model}, judge={judge_model})")

    root_llm = make_llm(root_model)
    sub_llm = make_llm(sub_model)
    judge_llm = make_llm(judge_model)

    results = []
    for idx, item in enumerate(items, 1):
        res = run_one(item, context, root_llm, sub_llm, judge_llm,
                      max_depth=args.max_depth, max_iterations=args.max_iterations,
                      question_field=args.question_field)
        flag = res.error or res.verdict.label
        print(f"[{idx}/{len(items)}] {item.id} ({item.difficulty}) "
              f"turns={res.turns} -> {flag}")
        results.append(res)

    agg = aggregate(results)
    _print_summary(agg)

    payload = {
        "config": {
            "set": args.set, "n": args.n, "seed": args.seed,
            "difficulty": args.difficulty, "question_field": args.question_field,
            "root_model": root_model, "sub_model": sub_model, "judge_model": judge_model,
            "max_iterations": args.max_iterations, "max_depth": args.max_depth,
        },
        "aggregate": agg,
        "results": [
            {
                "id": r.item.id, "difficulty": r.item.difficulty,
                "question": getattr(r.item, args.question_field, "") or r.item.question,
                "gold": r.item.answer, "model_answer": r.model_answer,
                "turns": r.turns, "label": r.verdict.label,
                "reason": r.verdict.reason, "error": r.error,
            }
            for r in results
        ],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
