from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, HttpUrl

from backend.app.ingestion.chunker import chunk_text
from backend.app.ingestion.pdf_processor import (
    PDFProcessingError,
    extract_pdf_text,
)
from backend.app.ingestion.policy_fetcher import fetch_policy
from backend.app.ingestion.policy_validator import is_valid_policy
from backend.app.ingestion.processor import process_text, process_url
from backend.app.retrieval.vector_store import answer_question, store_chunks

app = FastAPI(
    title="PolicyLens API",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500_000)


class URLRequest(BaseModel):
    url: HttpUrl


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/ingest/text")
async def ingest_text(request: TextRequest):
    document = process_text(request.text)
    chunks = chunk_text(document["text"])

    document_id = str(uuid4())

    store_chunks(
        document_id,
        chunks,
        source="text",
    )

    return {
        "status": "success",
        "document_id": document_id,
        "chunks": len(chunks),
    }


@app.post("/api/v1/ingest/url")
async def ingest_url(request: URLRequest):
    try:
        document = await process_url(str(request.url))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    document_id = str(uuid4())
    total_chunks = 0
    fetched_policies = {}

    for category, urls in document["policies"].items():
        fetched_policies[category.value] = []

        for url in urls:
            try:
                text = await fetch_policy(url)
                if not is_valid_policy(text, category):
                    continue

                chunks = chunk_text(text)
                
                if not chunks:
                    continue

                store_chunks(
                    document_id,
                    chunks,
                    source=url,
                )

                total_chunks += len(chunks)
                fetched_policies[category.value].append(url)

            except Exception:
                continue

    return {
        "status": "success",
        "document_id": document_id,
        "url": document["url"],
        "chunks": total_chunks,
        "policies": fetched_policies,
    }


@app.post("/api/v1/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
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

    store_chunks(
        document_id,
        chunks,
        filename=file.filename,
    )

    return {
        "status": "success",
        "document_id": document_id,
        "filename": file.filename,
        "chunks": len(chunks),
    }


@app.post("/api/v1/ask")
async def ask_question(request: QuestionRequest):
    return answer_question(
        request.question,
        request.top_k,
    )