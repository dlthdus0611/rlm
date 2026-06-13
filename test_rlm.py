from rlm_graph import parse_code_blocks


def test_parse_no_block():
    assert parse_code_blocks("그냥 텍스트, 코드 없음") == []


def test_parse_single_block():
    text = "계획을 세웁니다.\n```repl\nprint(1)\n```\n끝."
    assert parse_code_blocks(text) == ["print(1)"]


def test_parse_multiple_blocks():
    text = "```repl\na = 1\n```\n사이 텍스트\n```repl\nprint(a)\n```"
    assert parse_code_blocks(text) == ["a = 1", "print(a)"]


def test_parse_ignores_other_languages():
    text = "```python\nprint('x')\n```\n```repl\nprint('y')\n```"
    assert parse_code_blocks(text) == ["print('y')"]


from rlm_graph import REPL


def _make_repl(context="컨텍스트"):
    return REPL(
        context=context,
        llm_query=lambda p: f"LLM:{p}",
        llm_query_batched=lambda ps: [f"LLM:{p}" for p in ps],
        rlm_query=lambda q, c: f"RLM:{q}",
    )


def test_repl_captures_stdout():
    repl = _make_repl()
    out = repl.run("print('안녕')")
    assert "안녕" in out


def test_repl_context_available():
    repl = _make_repl(context="비밀값")
    out = repl.run("print(context)")
    assert "비밀값" in out


def test_repl_exception_becomes_traceback():
    repl = _make_repl()
    out = repl.run("raise ValueError('펑')")
    assert "ValueError" in out and "펑" in out
    assert "ok" in repl.run("print('ok')")


def test_repl_persists_variables_across_runs():
    repl = _make_repl()
    repl.run("x = 41")
    out = repl.run("print(x + 1)")
    assert "42" in out


def test_repl_answer_ready_sets_final_answer():
    repl = _make_repl()
    assert repl.final_answer is None
    repl.run('answer["content"] = "최종 답"\nanswer["ready"] = True')
    assert repl.final_answer == "최종 답"


def test_repl_answer_not_ready_keeps_none():
    repl = _make_repl()
    repl.run('answer["content"] = "아직"')
    assert repl.final_answer is None


def test_repl_llm_query_injected():
    repl = _make_repl()
    out = repl.run("print(llm_query('질문'))")
    assert "LLM:질문" in out


def test_repl_show_vars():
    repl = _make_repl()
    repl.run("myvar = 1")
    out = repl.run("print(SHOW_VARS())")
    assert "myvar" in out
    assert "context" not in out


from rlm_graph import _truncate, MAX_OUTPUT_CHARS


def test_truncate_short_passthrough():
    assert _truncate("짧은 문자열") == "짧은 문자열"


def test_truncate_long_marks_remainder():
    s = "a" * (MAX_OUTPUT_CHARS + 100)
    out = _truncate(s)
    assert out.startswith("a" * MAX_OUTPUT_CHARS)
    assert "+100 chars" in out
    assert len(out) < len(s)
