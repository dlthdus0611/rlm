from rlm.repl import REPL


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
