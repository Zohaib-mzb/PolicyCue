from pinecone import Pinecone

from backend.app.core.config import get_settings
from backend.app.core.external_retry import retry_external
from backend.app.retrieval.embeddings import (
    create_document_embeddings,
    create_query_embedding,
)


settings = get_settings()

pinecone = Pinecone(
    api_key=settings.pinecone_api_key,
)

index = pinecone.Index(
    settings.pinecone_index_name,
)


def store_chunks(
    document_id: str,
    chunks: list[str],
    source: str = "",
    filename: str | None = None,
    *,
    owner_id: str | None = None,
    chunk_metadata: list[dict] | None = None,
    chunk_index_offset: int = 0,
) -> None:
    if not chunks:
        return

    if chunk_metadata is not None and len(chunk_metadata) != len(chunks):
        raise ValueError("Chunk metadata count must match chunk count.")

    embeddings = create_document_embeddings(chunks)

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count mismatch: "
            f"expected {len(chunks)}, "
            f"received {len(embeddings)}."
        )

    vectors = [
        {
            "id": f"{document_id}-{i + chunk_index_offset}",
            "values": embedding,
            "metadata": {
                "document_id": document_id,
                "owner_id": owner_id or "",
                "text": chunk,
                "chunk_index": i + chunk_index_offset,
                "source": source,
                "filename": filename or "",
            },
        }
        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        )
    ]

    if chunk_metadata is not None:
        for vector, metadata in zip(vectors, chunk_metadata):
            for key in ("source_url", "source_urls", "policy_categories", "content_hash"):
                if key in metadata:
                    vector["metadata"][key] = metadata[key]
            vector["metadata"]["source"] = metadata.get("source_url", source)

    retry_external(
        lambda: index.upsert(
            vectors=vectors,
        )
    )


def delete_document_vectors(document_id: str, owner_id: str) -> None:
    retry_external(
        lambda: index.delete(
            filter={
                "$and": [
                    {
                        "document_id": {
                            "$eq": document_id,
                        }
                    },
                    {
                        "owner_id": {
                            "$eq": owner_id,
                        }
                    },
                ]
            },
        )
    )


def search_chunks(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    owner_id: str | None = None,
) -> list[dict]:
    query_embedding = create_query_embedding(query)

    query_args = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
    }

    if document_id and owner_id:
        query_args["filter"] = {
            "$and": [
                {
                    "document_id": {
                        "$eq": document_id,
                    }
                },
                {
                    "owner_id": {
                        "$eq": owner_id,
                    }
                },
            ]
        }
    elif document_id:
        query_args["filter"] = {
            "document_id": {
                "$eq": document_id,
            }
        }

    results = retry_external(
        lambda: index.query(
            **query_args,
        )
    )

    return [
        {
            "score": match["score"],
            "text": match["metadata"]["text"],
            "document_id": match["metadata"]["document_id"],
            "owner_id": match["metadata"].get("owner_id", ""),
            "chunk_index": match["metadata"]["chunk_index"],
            "source": match["metadata"].get(
                "source",
                "",
            ),
            "filename": match["metadata"].get(
                "filename",
                "",
            ),
            **{
                key: match["metadata"][key]
                for key in ("source_url", "source_urls", "policy_categories", "content_hash")
                if key in match["metadata"]
            },
        }
        for match in results["matches"]
    ]


def answer_question(
    question: str,
    top_k: int = 5,
    document_id: str | None = None,
    owner_id: str | None = None,
) -> dict:
    from backend.app.analysis.answer_generator import (
        generate_answer,
    )

    results = search_chunks(
        query=question,
        top_k=top_k,
        document_id=document_id,
        owner_id=owner_id,
    )

    answer = generate_answer(
        question,
        results,
    )

    return {
        "answer": answer,
        "sources": results,
    }
