def test_answer_prompt_includes_passages_and_question():
    from rag.prompts import build_answer_prompt
    p = build_answer_prompt("종속기업 몇 개?", ["종속기업은 232개다.", "계열사는 63개다."])
    assert "종속기업 몇 개?" in p
    assert "232개" in p and "63개" in p


def test_rerank_prompt_asks_for_score():
    from rag.prompts import build_rerank_prompt
    p = build_rerank_prompt("q", "어떤 passage")
    assert "어떤 passage" in p and "q" in p
