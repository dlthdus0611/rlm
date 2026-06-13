import re

CODE_BLOCK_RE = re.compile(r"```repl[ \t]*\n(.*?)```", re.DOTALL)


def parse_code_blocks(text: str) -> list[str]:
    """모델 응답에서 ```repl ... ``` 블록의 코드만 추출한다."""
    return [block.strip("\n") for block in CODE_BLOCK_RE.findall(text)]


MAX_OUTPUT_CHARS = 8000


def _truncate(s: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """REPL 출력이 한도를 넘으면 잘라내고 남은 글자 수를 표기한다."""
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[+{len(s) - limit} chars truncated]"
