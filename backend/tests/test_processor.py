from backend.app.ingestion.processor import process_text


def test_process_text():
    result = process_text("  Privacy Policy content  ")

    assert result["text"] == "Privacy Policy content"