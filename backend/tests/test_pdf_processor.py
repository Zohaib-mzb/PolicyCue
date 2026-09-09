import pytest

from backend.app.ingestion.pdf_processor import (
    PDFProcessingError,
    extract_pdf_text,
)


def test_invalid_pdf_is_rejected():
    with pytest.raises(PDFProcessingError):
        extract_pdf_text(b"not a pdf")