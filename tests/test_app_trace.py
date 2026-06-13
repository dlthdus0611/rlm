from langchain_core.messages import AIMessage, HumanMessage

from app_trace import format_update, TraceEntry


def test_call_model_update_extracts_code_and_increments_turn():
    ai = AIMessage(content='계획을 세웁니다.\n```repl\nprint(1)\n```')
    update = {"call_model": {"messages": [HumanMessage(content="턴 1/12:"), ai]}}
    entries, turn, final = format_update(update, 0)
    assert turn == 1
    assert final is None
    assert len(entries) == 1
    e = entries[0]
    assert e.kind == "model"
    assert e.turn == 1
    assert e.code_blocks == ["print(1)"]
    assert "계획" in e.text


def test_execute_code_update_without_final():
    update = {"execute_code": {
        "messages": [HumanMessage(content="1\n")],
        "iteration": 1, "final_answer": None,
    }}
    entries, turn, final = format_update(update, 1)
    assert final is None
    assert turn == 1
    assert len(entries) == 1
    assert entries[0].kind == "exec"
    assert "1" in entries[0].text


def test_execute_code_update_with_final():
    update = {"execute_code": {
        "messages": [HumanMessage(content="(출력 없음)")],
        "iteration": 2, "final_answer": "정답 42",
    }}
    entries, turn, final = format_update(update, 1)
    assert final == "정답 42"
    assert entries[0].kind == "exec"


def test_unknown_node_yields_no_entries():
    update = {"setup": {"messages": [], "iteration": 0}}
    entries, turn, final = format_update(update, 3)
    assert entries == []
    assert turn == 3
    assert final is None


def test_call_model_with_empty_messages_does_not_crash():
    update = {"call_model": {"messages": []}}
    entries, turn, final = format_update(update, 0)
    assert turn == 1
    assert len(entries) == 1
    assert entries[0].kind == "model"
    assert entries[0].text == ""
    assert entries[0].code_blocks == []


def test_non_str_content_is_coerced_to_str():
    # 멀티모달 등 list content 도 크래시 없이 문자열로 강제돼야 한다.
    ai = AIMessage(content=[{"type": "text", "text": "안녕"}])
    update = {"call_model": {"messages": [ai]}}
    entries, turn, final = format_update(update, 0)
    assert isinstance(entries[0].text, str)
    assert "안녕" in entries[0].text
