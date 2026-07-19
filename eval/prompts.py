"""LLM-as-judge 프롬프트 — 정답(gold)과 모델답변을 대조해 채점하도록 지시한다.

harness.judge()가 사용한다. rlm/prompts.py(모델 행동 프롬프트)와 대칭으로, 채점자
프롬프트를 채점 로직에서 분리해 문구 수정을 한곳에서 하도록 둔다.
"""

JUDGE_SYSTEM = (
    "너는 QA 채점자다. 주어진 '정답'과 '모델답변'이 핵심적으로 일치하는지 판정한다.\n"
    "- correct: 정답의 핵심 사실과 수치가 모두 일치한다(표현·어순 차이는 무방하나, "
    "숫자는 반올림·단위까지 일치해야 한다).\n"
    "- partial: 여러 부분 중 일부만 맞거나, 핵심 수치는 맞지만 요구된 부가정보가 빠졌다.\n"
    "- incorrect: 틀리거나 핵심이 누락되었거나 답이 없다.\n"
    '반드시 JSON 한 줄로만 답하라: {"label": "correct|partial|incorrect", "reason": "간단한 이유"}'
)


def build_judge_prompt(question: str, gold: str, candidate: str) -> str:
    """채점자에게 줄 사용자 메시지 — 질문·정답·모델답변을 대조 형식으로 담는다."""
    return (
        f"[질문]\n{question}\n\n"
        f"[정답]\n{gold}\n\n"
        f"[모델답변]\n{candidate}\n\n"
        "위 모델답변을 정답 기준으로 채점하라."
    )
