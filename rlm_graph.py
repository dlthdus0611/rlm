import re

CODE_BLOCK_RE = re.compile(r"```repl[ \t]*\n(.*?)```", re.DOTALL)


def parse_code_blocks(text: str) -> list[str]:
    """모델 응답에서 ```repl ... ``` 블록의 코드만 추출한다."""
    return [block.strip("\n") for block in CODE_BLOCK_RE.findall(text)]
