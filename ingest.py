from pathlib import Path
from pypdf import PdfReader

from agents.schemas import DocumentChunk


def load_pdf(path: str) -> list[dict]:
    reader = PdfReader(path)

    pages = []

    for i, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():

            pages.append({
                "text": text.strip(),
                "source": Path(path).name,
                "page": i,
            })

    return pages


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 120
) -> list[str]:

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunks.append(
            " ".join(words[start:end])
        )

        if end == len(words):
            break

        start = max(
            end - overlap,
            start + 1
        )

    return chunks


def build_chunks(
    pages: list[dict]
) -> list[DocumentChunk]:

    output = []

    for page in pages:

        chunks = chunk_text(
            page["text"]
        )

        for idx, chunk in enumerate(chunks):

            output.append(
                DocumentChunk(
                    text=chunk,
                    source=page["source"],
                    page=page["page"],
                    chunk_id=idx
                )
            )

    return output