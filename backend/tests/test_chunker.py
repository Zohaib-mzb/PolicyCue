from backend.app.ingestion.chunker import chunk_text


def test_chunk_text():
    text = "This is a test. " * 200

    chunks = chunk_text(text)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)