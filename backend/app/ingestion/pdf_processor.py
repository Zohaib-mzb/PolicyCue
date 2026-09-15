from io import BytesIO

from pypdf import PdfReader


MAX_PDF_SIZE = 10 * 1024 * 1024
MAX_PDF_PAGES = 150
MAX_EXTRACTED_TEXT_CHARACTERS = 500_000
PDF_SIGNATURE = b"%PDF-"


class PDFProcessingError(Exception):
    pass


def validate_pdf_signature(content: bytes) -> None:
    if not content.startswith(PDF_SIGNATURE):
        raise PDFProcessingError("Uploaded file is not a valid PDF.")


def extract_pdf_text(content: bytes) -> str:
    if len(content) > MAX_PDF_SIZE:
        raise PDFProcessingError("PDF is too large.")

    validate_pdf_signature(content)

    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise PDFProcessingError("Encrypted PDFs are not supported.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise PDFProcessingError("PDF has too many pages.")

        parts = []
        total_characters = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            total_characters += len(page_text)
            if total_characters > MAX_EXTRACTED_TEXT_CHARACTERS:
                raise PDFProcessingError("PDF extracted text is too large.")
            parts.append(page_text)
        text = "\n".join(parts)
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(
            "Unable to process PDF."
        ) from exc

    text = text.strip()

    if not text:
        raise PDFProcessingError(
            "PDF contains no extractable text."
        )

    return text
