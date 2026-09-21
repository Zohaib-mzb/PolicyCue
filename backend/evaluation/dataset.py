"""Versioned fixtures and strict, auditable chunk labels."""
import hashlib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_-]+$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Document(StrictModel):
    id: Identifier
    text: NonBlank
    chunk_sha256: list[Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]] = Field(min_length=1)


class Example(StrictModel):
    id: Identifier
    document_id: Identifier
    question: NonBlank
    expected_answer: NonBlank | None
    relevant_chunk_ids: list[NonBlank]
    should_answer: bool
    notes: str | None = None
    category: str | None = None

    @model_validator(mode="after")
    def validate_labels(self):
        if self.should_answer != bool(self.expected_answer and self.relevant_chunk_ids):
            raise ValueError("Answerable examples require a reference answer and relevant chunks")
        if not self.should_answer and (self.expected_answer is not None or self.relevant_chunk_ids):
            raise ValueError("Unanswerable examples require null reference and empty relevant chunks")
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("Duplicate relevant chunk IDs")
        return self


class Dataset(StrictModel):
    name: NonBlank
    version: NonBlank
    description: NonBlank
    synthetic: bool
    documents: list[Document] = Field(min_length=1)
    examples: list[Example] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self):
        for items in (self.documents, self.examples):
            if len({item.id for item in items}) != len(items):
                raise ValueError("Duplicate document or example IDs")
        documents = {doc.id: doc for doc in self.documents}
        for example in self.examples:
            if example.document_id not in documents:
                raise ValueError(f"Unknown document: {example.document_id}")
            doc = documents[example.document_id]
            valid = {f"{doc.id}:{i}" for i in range(len(doc.chunk_sha256))}
            if not set(example.relevant_chunk_ids) <= valid:
                raise ValueError(f"Invalid chunk labels: {example.id}")
        return self


def prepare_chunks(dataset: Dataset) -> dict[str, list[str]]:
    from backend.app.ingestion.chunker import chunk_text

    prepared = {}
    for doc in dataset.documents:
        chunks = chunk_text(doc.text)
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in chunks]
        if hashes != doc.chunk_sha256:
            raise ValueError(f"Chunk manifest changed for {doc.id}; manually review labels and version")
        prepared[doc.id] = chunks
    return prepared


def load_dataset(path: Path) -> Dataset:
    dataset = Dataset.model_validate_json(path.read_text(encoding="utf-8"))
    prepare_chunks(dataset)
    return dataset
