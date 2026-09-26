import re
from agents.schemas import DocumentChunk


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120
) -> list[str]:

    words = clean_text(text).split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(pages: list[dict]) -> list[DocumentChunk]:

    output = []

    for page in pages:

        chunks = chunk_text(page["text"])

        for chunk_id, chunk in enumerate(chunks):

            output.append(
                DocumentChunk(
                    text=chunk,
                    source=page["source"],
                    page=page["page"],
                    chunk_id=chunk_id,
                    token_count=len(chunk.split())
                )
            )

    return output