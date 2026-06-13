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
