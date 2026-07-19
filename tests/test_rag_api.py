from tests.test_rag_index import FakeEmbedder
from langchain_core.messages import AIMessage


class FakeChat:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, prompt, config=None):
        return AIMessage(content=self.reply)


def test_run_end_to_end_with_fakes(tmp_path):
    from rag.api import run
    ctx = "삼성전자 종속기업은 232개다. " * 20
    out = run("종속기업 몇 개?", ctx,
              root_llm=FakeChat("232개"), sub_llm=FakeChat("5"),
              embeddings=FakeEmbedder(),
              use_hyde=False, use_rerank=False, cache_dir=str(tmp_path))
    assert out == "232개"
