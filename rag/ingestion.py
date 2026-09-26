from rag.chunker import TextChunker


class DocumentIngestion:

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
    ):
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def ingest(
        self,
        text: str,
        source: str = "unknown",
        page: int = 1,
    ):

        if not text or not text.strip():
            return []

        return self.chunker.split(
            text=text,
            source=source,
            page=page,
        )