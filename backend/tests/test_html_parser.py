from backend.app.ingestion.html_parser import extract_text


def test_extract_text_removes_unwanted_elements():
    html = """
    <html>
        <body>
            <h1>Privacy Policy</h1>
            <p>Your privacy matters.</p>
            <script>alert("secret");</script>
            <style>body { color: red; }</style>
        </body>
    </html>
    """

    text = extract_text(html)

    assert "Privacy Policy" in text
    assert "Your privacy matters." in text
    assert "alert" not in text
    assert "color: red" not in text