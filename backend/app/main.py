from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from uuid import uuid4
from backend.app.ingestion.chunker import chunk_text
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
    store_chunks(document_id, chunks)

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

    chunks = chunk_text(document["text"])

    document_id = str(uuid4())
    store_chunks(document_id, chunks)

    return {
        "status": "success",
        "document_id": document_id,
        "url": document["url"],
        "chunks": len(chunks),
        "policies": {
            category.value: urls
            for category, urls in document["policies"].items()
        },
    }


@app.post("/api/v1/ask")
async def ask_question(request: QuestionRequest):
    return answer_question(
        request.question,
        request.top_k,
    )