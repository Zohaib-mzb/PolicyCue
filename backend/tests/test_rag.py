from unittest.mock import patch

from backend.app.retrieval.vector_store import answer_question


def test_answer_question():
    mock_results = [
        {
            "score": 0.95,
            "text": "We collect your email to provide our services.",
            "document_id": "test-document",
            "chunk_index": 0,
        }
    ]

    with patch(
        "backend.app.retrieval.vector_store.search_chunks",
        return_value=mock_results,
    ), patch(
        "backend.app.analysis.answer_generator.generate_answer",
        return_value="The policy says your email is collected to provide services.",
    ):
        result = answer_question(
            "Why is my email collected?"
        )

    assert "email" in result["answer"]
    assert len(result["sources"]) == 1