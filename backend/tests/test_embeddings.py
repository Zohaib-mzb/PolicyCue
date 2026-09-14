from backend.app.retrieval.embeddings import (
    create_document_embeddings,
    create_query_embedding,
)


def test_embedding_dimensions():
    documents = create_document_embeddings(
        ["Privacy Policy"]
    )

    query = create_query_embedding(
        "What does the privacy policy say?"
    )

    assert len(documents) == 1
    assert len(documents[0]) == 768
    assert len(query) == 768