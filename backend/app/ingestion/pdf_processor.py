from io import BytesIO

from pypdf import PdfReader


MAX_PDF_SIZE = 10 * 1024 * 1024


class PDFProcessingError(Exception):
    pass


def extract_pdf_text(content: bytes) -> str:
    if len(content) > MAX_PDF_SIZE:
        raise PDFProcessingError("PDF is too large.")

    try:
        reader = PdfReader(BytesIO(content))
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
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