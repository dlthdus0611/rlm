import pytest


def test_make_embeddings_requires_key(monkeypatch):
    from rag.embeddings import make_embeddings
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        make_embeddings()


def test_make_embeddings_uses_settings_model(monkeypatch):
    from rag.embeddings import make_embeddings
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    emb = make_embeddings()
    assert emb.model == "text-embedding-3-large"
