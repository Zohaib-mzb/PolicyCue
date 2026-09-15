from unittest.mock import patch

import pytest

from backend.app.ingestion import pdf_processor
from backend.app.ingestion.pdf_processor import (
    MAX_EXTRACTED_TEXT_CHARACTERS,
    MAX_PDF_PAGES,
    PDFProcessingError,
    extract_pdf_text,
)


class FakePage:
    def __init__(self, text: str):
        self.text = text
        self.extract_count = 0

    def extract_text(self):
        self.extract_count += 1
        return self.text


class FakePages:
    def __init__(self, pages, reported_length=None):
        self.pages = pages
        self.reported_length = reported_length if reported_length is not None else len(pages)
        self.iterated = False

    def __len__(self):
        return self.reported_length

    def __iter__(self):
        self.iterated = True
        return iter(self.pages)


class FakeReader:
    def __init__(self, pages, *, encrypted=False, reported_length=None):
        self.is_encrypted = encrypted
        self.pages = FakePages(pages, reported_length=reported_length)


class ExplodingPages:
    def __len__(self):
        return MAX_PDF_PAGES + 1

    def __iter__(self):
        raise AssertionError("pages should not be extracted after page-limit rejection")


class ExplodingReader:
    is_encrypted = False
    pages = ExplodingPages()


class MalformedReader:
    def __init__(self, _stream):
        raise RuntimeError("broken pdf internals")


def test_valid_normal_pdf_extracts_text():
    reader = FakeReader([FakePage("Privacy policy text")])

    with patch.object(pdf_processor, "PdfReader", return_value=reader):
        assert extract_pdf_text(b"%PDF-1.7\n...") == "Privacy policy text"


def test_invalid_pdf_is_rejected():
    with pytest.raises(PDFProcessingError):
        extract_pdf_text(b"not a pdf")


def test_invalid_pdf_signature_is_rejected_before_pypdf():
    with patch.object(pdf_processor, "PdfReader") as reader:
        with pytest.raises(PDFProcessingError) as caught:
            extract_pdf_text(b"html pretending to be pdf")

    assert caught.value.args[0] == "Uploaded file is not a valid PDF."
    reader.assert_not_called()


def test_encrypted_pdf_is_rejected():
    with patch.object(pdf_processor, "PdfReader", return_value=FakeReader([], encrypted=True)):
        with pytest.raises(PDFProcessingError) as caught:
            extract_pdf_text(b"%PDF-1.7\n...")

    assert caught.value.args[0] == "Encrypted PDFs are not supported."


def test_excessive_page_count_is_rejected_before_full_extraction():
    with patch.object(pdf_processor, "PdfReader", return_value=ExplodingReader()):
        with pytest.raises(PDFProcessingError) as caught:
            extract_pdf_text(b"%PDF-1.7\n...")

    assert caught.value.args[0] == "PDF has too many pages."


def test_excessive_extracted_text_is_rejected_without_truncation():
    first = FakePage("a" * MAX_EXTRACTED_TEXT_CHARACTERS)
    second = FakePage("b")
    reader = FakeReader([first, second])

    with patch.object(pdf_processor, "PdfReader", return_value=reader):
        with pytest.raises(PDFProcessingError) as caught:
            extract_pdf_text(b"%PDF-1.7\n...")

    assert caught.value.args[0] == "PDF extracted text is too large."
    assert first.extract_count == 1
    assert second.extract_count == 1


def test_malformed_pdf_returns_controlled_error():
    with patch.object(pdf_processor, "PdfReader", side_effect=RuntimeError("bad internals")):
        with pytest.raises(PDFProcessingError) as caught:
            extract_pdf_text(b"%PDF-1.7\n...")

    assert caught.value.args[0] == "Unable to process PDF."
