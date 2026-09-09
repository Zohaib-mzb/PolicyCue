from backend.app.retrieval.embeddings import create_embeddings


def test_embedding_dimensions():
    embeddings = create_embeddings(["Privacy Policy"])

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768