import io
import re
import traceback
from contextlib import redirect_stdout
from typing import Callable

CODE_BLOCK_RE = re.compile(r"```repl[ \t]*\n(.*?)```", re.DOTALL)


def parse_code_blocks(text: str) -> list[str]:
    """모델 응답에서 ```repl ... ``` 블록의 코드만 추출한다."""
    return [block.strip("\n") for block in CODE_BLOCK_RE.findall(text)]


class _AnswerDict(dict):
    """answer["ready"] = True 가 설정되면 콜백으로 content를 캡처하는 dict."""

    def __init__(self, on_ready: Callable[[str], None]):
        super().__init__()
        super().__setitem__("content", "")
        super().__setitem__("ready", False)
        self._on_ready = on_ready

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "ready" and value:
            self._on_ready(self.get("content", ""))


_RESERVED = {
    "context", "llm_query", "llm_query_batched", "rlm_query",
    "answer", "SHOW_VARS", "__builtins__",
}


class REPL:
    """모델이 생성한 코드를 실행하는 영속 Python 네임스페이스.

    ⚠️ in-process exec(). 신뢰된 입력에만 사용.
    """

    def __init__(self, context, llm_query, llm_query_batched, rlm_query):
        self.final_answer: str | None = None
        self.ns: dict = {
            "context": context,
            "llm_query": llm_query,
            "llm_query_batched": llm_query_batched,
            "rlm_query": rlm_query,
            "answer": _AnswerDict(self._capture_answer),
            "SHOW_VARS": self._show_vars,
        }

    def _capture_answer(self, content) -> None:
        self.final_answer = str(content)

    def _show_vars(self) -> str:
        names = [
            k for k in self.ns
            if not k.startswith("_") and k not in _RESERVED
        ]
        return ", ".join(sorted(names))

    def run(self, code: str) -> str:
        """code를 실행하고 stdout(또는 예외 traceback)을 문자열로 반환."""
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, self.ns)
        except Exception:
            buf.write(traceback.format_exc())
        return buf.getvalue()
