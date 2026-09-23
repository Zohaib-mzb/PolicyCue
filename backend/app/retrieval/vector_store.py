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

VECTOR_STORAGE_BATCH_SIZE = 32


class VectorStorageError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        total_chunks: int,
        batch_number: int,
        batch_size: int,
    ) -> None:
        super().__init__(f"Vector storage failed during {stage}.")
        self.stage = stage
        self.total_chunks = total_chunks
        self.batch_number = batch_number
        self.batch_size = batch_size


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

    for start in range(0, len(chunks), VECTOR_STORAGE_BATCH_SIZE):
        batch = chunks[start:start + VECTOR_STORAGE_BATCH_SIZE]
        batch_number = start // VECTOR_STORAGE_BATCH_SIZE + 1
        try:
            embeddings = create_document_embeddings(batch)
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    "Embedding count mismatch: "
                    f"expected {len(batch)}, "
                    f"received {len(embeddings)}."
                )
        except Exception as exc:
            raise VectorStorageError(
                stage="embedding",
                total_chunks=len(chunks),
                batch_number=batch_number,
                batch_size=len(batch),
            ) from exc

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
                zip(batch, embeddings),
                start=start,
            )
        ]

        if chunk_metadata is not None:
            batch_metadata = chunk_metadata[start:start + len(batch)]
            for vector, metadata in zip(vectors, batch_metadata):
                for key in ("source_url", "source_urls", "policy_categories", "content_hash", "source_type", "title"):
                    if key in metadata:
                        vector["metadata"][key] = metadata[key]
                vector["metadata"]["source"] = metadata.get("source_url", source)

        try:
            retry_external(lambda: index.upsert(vectors=vectors))
        except Exception as exc:
            raise VectorStorageError(
                stage="upsert",
                total_chunks=len(chunks),
                batch_number=batch_number,
                batch_size=len(batch),
            ) from exc


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
                for key in ("source_url", "source_urls", "policy_categories", "content_hash", "source_type", "title")
                if key in match["metadata"]
            },
        }
        for match in results["matches"]
    ]


def source_attributions(chunks: list[dict]) -> list[dict]:
    attributions = []
    seen = set()

    for chunk in chunks:
        source_url = chunk.get("source_url") or chunk.get("source") or ""
        filename = chunk.get("filename") or ""
        categories = chunk.get("policy_categories") or []
        if isinstance(categories, str):
            categories = [categories]

        key = (source_url, filename, tuple(categories))
        if key in seen:
            continue
        seen.add(key)

        title = chunk.get("title") or ""
        source_type = chunk.get("source_type") or ""
        item = {
            "source_url": source_url,
            "filename": filename,
            "policy_categories": categories,
        }
        if title:
            item["title"] = title
        if source_type:
            item["source_type"] = source_type
        if "source_urls" in chunk:
            item["source_urls"] = chunk["source_urls"]
        attributions.append(item)

    return attributions


def answer_question(
    question: str,
    top_k: int = 5,
    document_id: str | None = None,
    owner_id: str | None = None,
) -> dict:
    from backend.app.analysis.answer_generator import (
        NO_ANSWER_MESSAGE,
        generate_answer,
    )

    results = search_chunks(
        query=question,
        top_k=top_k,
        document_id=document_id,
        owner_id=owner_id,
    )

    document_found = bool(results)

    answer = generate_answer(
        question,
        results,
    )

    if answer == NO_ANSWER_MESSAGE:
        results = []

    return {
        "answer": answer,
        "sources": results,
        "source_attributions": source_attributions(results),
        "document_found": document_found,
    }
