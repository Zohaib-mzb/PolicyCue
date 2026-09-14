import asyncio
import logging
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field, HttpUrl
from google.genai.errors import ClientError

from backend.app.core.session import get_or_create_owner_id, require_owner_id
from backend.app.ingestion.chunker import chunk_text
from backend.app.ingestion.pdf_processor import (
    PDFProcessingError,
    extract_pdf_text,
)
from backend.app.ingestion.policy_discovery import (
    discover_website_policies,
)
from backend.app.ingestion.processor import process_text
from backend.app.ingestion.url_fetcher import URLFetchError
from backend.app.retrieval.vector_store import (
    answer_question,
    delete_document_vectors,
    store_chunks,
)


logger = logging.getLogger(__name__)


app = FastAPI(
    title="PolicyLens API",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500_000)


class URLRequest(BaseModel):
    url: HttpUrl


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/ingest/text")
async def ingest_text(
    request: TextRequest,
    http_request: Request,
    response: Response,
):
    document = process_text(request.text)
    chunks = chunk_text(document["text"])

    document_id = str(uuid4())
    owner_id = get_or_create_owner_id(http_request, response)

    try:
        store_chunks(
            document_id,
            chunks,
            source="text",
            owner_id=owner_id,
        )
    except Exception as exc:
        _cleanup_failed_document(document_id, owner_id)
        raise HTTPException(
            status_code=503,
            detail="Document storage failed. Please retry ingestion.",
        ) from exc

    return {
        "status": "success",
        "document_id": document_id,
        "chunks": len(chunks),
    }


# Keep URL embedding/upsert requests small without changing the PDF path.
URL_CHUNK_BATCH_SIZE = 32
URL_RATE_LIMIT_RETRIES = 1
URL_RATE_LIMIT_RETRY_SECONDS = 60


def _cleanup_failed_document(document_id: str, owner_id: str) -> None:
    try:
        delete_document_vectors(document_id, owner_id)
    except Exception:
        logger.warning(
            "Failed to clean up vectors after ingestion failure for document_id=%s.",
            document_id,
        )


@app.post("/api/v1/ingest/url")
async def ingest_url(
    request: URLRequest,
    http_request: Request,
    response: Response,
):
    try:
        document = await discover_website_policies(str(request.url))
    except URLFetchError as exc:
        raise HTTPException(status_code=400, detail="Website URL is not allowed.") from exc

    document_id = str(uuid4())
    owner_id = get_or_create_owner_id(http_request, response)
    chunks = []
    metadata = []
    policies = {}
    accepted = []
    for page in document["pages"]:
        page_chunks = chunk_text(page["text"])
        if not page_chunks:
            continue
        categories = [category.value for category in page["categories"]]
        page_metadata = {
            "source_url": page["url"],
            "source_urls": page["source_urls"],
            "policy_categories": categories,
            "content_hash": page["content_hash"],
        }
        chunks.extend(page_chunks)
        metadata.extend([page_metadata] * len(page_chunks))
        accepted.append({**page_metadata, "chunks": len(page_chunks)})
        for category in categories:
            policies.setdefault(category, []).append(page["url"])

    try:
        for start in range(0, len(chunks), URL_CHUNK_BATCH_SIZE):
            for attempt in range(URL_RATE_LIMIT_RETRIES + 1):
                try:
                    await asyncio.to_thread(
                        store_chunks,
                        document_id,
                        chunks[start:start + URL_CHUNK_BATCH_SIZE],
                        owner_id=owner_id,
                        chunk_metadata=metadata[start:start + URL_CHUNK_BATCH_SIZE],
                        chunk_index_offset=start,
                    )
                    break
                except ClientError as exc:
                    if exc.code != 429 or attempt == URL_RATE_LIMIT_RETRIES:
                        raise
                    await asyncio.sleep(URL_RATE_LIMIT_RETRY_SECONDS)
    except Exception as exc:
        _cleanup_failed_document(document_id, owner_id)
        raise HTTPException(
            status_code=503,
            detail="Policy storage failed. Please retry ingestion.",
        ) from exc

    return {
        "status": "success" if chunks else "no_policies_found",
        "document_id": document_id,
        "url": document["url"],
        "chunks": len(chunks),
        "policies": policies,
        "accepted_policies": accepted,
        "warnings": document["warnings"],
        "skipped": document["skipped"],
        "candidates": document["candidates"],
    }


@app.post("/api/v1/ingest/pdf")
async def ingest_pdf(
    http_request: Request,
    response: Response,
    file: UploadFile = File(...),
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    content = await file.read()

    try:
        text = extract_pdf_text(content)
    except PDFProcessingError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    chunks = chunk_text(text)

    document_id = str(uuid4())
    owner_id = get_or_create_owner_id(http_request, response)

    try:
        store_chunks(
            document_id,
            chunks,
            filename=file.filename,
            owner_id=owner_id,
        )
    except Exception as exc:
        _cleanup_failed_document(document_id, owner_id)
        raise HTTPException(
            status_code=503,
            detail="Document storage failed. Please retry ingestion.",
        ) from exc

    return {
        "status": "success",
        "document_id": document_id,
        "filename": file.filename,
        "chunks": len(chunks),
    }


@app.post("/api/v1/ask")
async def ask_question(request: QuestionRequest, http_request: Request):
    owner_id = require_owner_id(http_request)
    if owner_id is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    result = answer_question(
        request.question,
        request.top_k,
        request.document_id,
        owner_id,
    )
    if not result["sources"]:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return result
