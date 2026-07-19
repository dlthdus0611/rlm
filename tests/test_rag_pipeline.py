from langchain_core.messages import AIMessage
from rag.retriever import Passage


class FakeRetriever:
    def __init__(self, passages):
        self._p = passages

    def retrieve(self, question):
        return list(self._p)


class FakeGen:
    def __init__(self, reply="종속기업은 232개다."):
        self.reply = reply
        self.last_config = "unset"

    def invoke(self, messages, config=None):
        self.last_config = config
        return AIMessage(content=self.reply)


def test_answer_uses_retrieved_passages():
    from rag.pipeline import answer
    retr = FakeRetriever([Passage("종속기업은 232개다.", 0, 9.0)])
    gen = FakeGen()
    res = answer("종속기업 몇 개?", retr, gen)
    assert res.answer == "종속기업은 232개다."
    assert res.passages[0].text == "종속기업은 232개다."


def test_answer_propagates_callbacks():
    from rag.pipeline import answer
    gen = FakeGen()
    marker = ["cb"]
    answer("q", FakeRetriever([Passage("p", 0)]), gen, callbacks=marker)
    assert gen.last_config == {"callbacks": marker}
