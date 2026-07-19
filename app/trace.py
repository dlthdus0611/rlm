"""graph.stream(stream_mode="updates") 업데이트를 화면용 트레이스 항목으로 변환.

streamlit에 의존하지 않는 순수 모듈 — 단위 테스트가 쉽다.
"""
from dataclasses import dataclass, field
from typing import Optional

from rlm.parsing import parse_code_blocks


@dataclass
class TraceEntry:
    kind: str                       # "model" | "exec"
    turn: int                       # 1-based 턴 번호
    text: str                       # 모델 산문/코드 또는 REPL 출력
    code_blocks: list = field(default_factory=list)  # kind=="model"의 ```repl``` 코드


def _content_str(msg) -> str:
    content = getattr(msg, "content", "")
    return content if isinstance(content, str) else str(content)


def format_update(
    update: dict, turn: int
) -> tuple[list[TraceEntry], int, Optional[str]]:
    """업데이트 dict 하나를 (entries, new_turn, final_answer|None)로 변환한다.

    update 예:
      {"call_model": {"messages": [turn_msg, ai_msg]}}
      {"execute_code": {"messages": [human_msg], "iteration": k, "final_answer": ...}}
    그 외 노드(setup 등)는 빈 목록을 낸다.
    """
    entries: list[TraceEntry] = []
    final_answer = None
    for node, payload in update.items():
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        last = messages[-1] if messages else None
        if node == "call_model":
            turn += 1
            text = _content_str(last) if last is not None else ""
            entries.append(TraceEntry("model", turn, text, parse_code_blocks(text)))
        elif node == "execute_code":
            text = _content_str(last) if last is not None else ""
            entries.append(TraceEntry("exec", turn, text))
            if isinstance(payload, dict) and payload.get("final_answer") is not None:
                final_answer = payload["final_answer"]
    return entries, turn, final_answer
