from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(text: str) -> list[str]:
    text = text.strip()

    if not text:
        return []

    return _splitter.split_text(text)