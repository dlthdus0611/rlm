from rlm.prompts import (
    SYSTEM_PROMPT,
    build_metadata_message, build_turn_prompt,
)


def test_system_prompt_mentions_key_tokens():
    for token in ["context", "llm_query", "rlm_query", "answer", "repl"]:
        assert token in SYSTEM_PROMPT


def test_metadata_message_includes_question_and_length():
    msg = build_metadata_message("환불 몇 건?", "x" * 1234)
    assert "환불 몇 건?" in msg
    assert "1234" in msg


def test_turn_prompt_iteration_zero_has_safeguard():
    msg = build_turn_prompt(0, 10)
    assert "턴 1/10" in msg
    assert "먼저 context" in msg


def test_turn_prompt_later_iteration_is_short():
    msg = build_turn_prompt(3, 10)
    assert "턴 4/10" in msg
    assert "먼저 context" not in msg
