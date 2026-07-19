"""RAG 프롬프트 — 검색된 근거만으로 답하도록 지시(오케스트레이터인 RLM 프롬프트와 대비)."""

RAG_SYSTEM = (
    "너는 주어진 '근거 발췌'만을 바탕으로 한국어로 정확히 답하는 어시스턴트다. "
    "근거에 없는 내용은 추측하지 말고, 수치는 근거의 값을 그대로 인용한다. "
    "답은 간결한 한 문장으로 제시한다."
)


def build_answer_prompt(question: str, passages: list[str]) -> str:
    joined = "\n\n".join(f"[근거 {i + 1}]\n{p}" for i, p in enumerate(passages))
    return f"{joined}\n\n[질문]\n{question}\n\n[답]"


def build_hyde_prompt(question: str) -> str:
    return (
        "다음 질문에 대해, 사업보고서에 있을 법한 이상적인 정답 문장을 한두 문장으로 "
        f"가정해 작성하라(검색용 가설 답변). 모르면 관련 키워드를 나열하라.\n\n질문: {question}"
    )


def build_rerank_prompt(question: str, passage: str) -> str:
    return (
        "아래 '근거 발췌'가 '질문'에 답하는 데 얼마나 관련 있는지 0~10 정수로만 답하라.\n\n"
        f"질문: {question}\n\n근거 발췌:\n{passage}\n\n관련도(0~10):"
    )
