from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class DocumentSource(str, Enum):
    WEBSITE = "website"
    URL = "url"
    PDF = "pdf"
    TEXT = "text"


class DocumentInput(BaseModel):
    source: DocumentSource
    url: HttpUrl | None = None
    text: str | None = Field(default=None, min_length=1)