from pinecone import Pinecone

from backend.app.core.config import get_settings
from backend.app.retrieval.embeddings import create_embeddings


settings = get_settings()

pinecone = Pinecone(api_key=settings.pinecone_api_key)
index = pinecone.Index(settings.pinecone_index_name)


def store_chunks(
    document_id: str,
    chunks: list[str],
    source: str = "",
    filename: str | None = None,
) -> None:
    if not chunks:
        return

    embeddings = create_embeddings(chunks)

    vectors = [
        {
            "id": f"{document_id}-{i}",
            "values": embedding,
            "metadata": {
                "document_id": document_id,
                "text": chunk,
                "chunk_index": i,
                "source": source,
                "filename": filename or "",
            },
        }
        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        )
    ]

    index.upsert(vectors=vectors)


def search_chunks(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    query_embedding = create_embeddings([query])[0]

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        {
            "score": match["score"],
            "text": match["metadata"]["text"],
            "document_id": match["metadata"]["document_id"],
            "chunk_index": match["metadata"]["chunk_index"],
            "source": match["metadata"].get("source", ""),
            "filename": match["metadata"].get("filename", ""),
        }
        for match in results["matches"]
    ]


def answer_question(
    question: str,
    top_k: int = 5,
) -> dict:
    from backend.app.analysis.answer_generator import generate_answer

    results = search_chunks(question, top_k)

    answer = generate_answer(
        question,
        results,
    )

    return {
        "answer": answer,
        "sources": results,
    }