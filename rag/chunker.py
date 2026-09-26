from agents.schemas import DocumentChunk


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100
    ):
        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(
        self,
        text: str,
        source: str = "unknown",
        page: int = 1
    ) -> list[DocumentChunk]:

        text = " ".join(
            text.split()
        )

        if not text:
            return []

        chunks = []

        start = 0
        chunk_id = 0

        while start < len(text):

            end = min(
                start + self.chunk_size,
                len(text)
            )

            chunk_text = text[start:end]

            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source=source,
                    page=page,
                    chunk_id=chunk_id
                )
            )

            chunk_id += 1

            if end >= len(text):
                break

            start = end - self.overlap

        return chunks