from agents.schemas import DocumentChunk


class MultimodalDocumentAdapter:

    @staticmethod
    def to_chunk(
        description: str,
        source: str,
        page: int = 1,
        chunk_id: int = 0
    ) -> DocumentChunk:

        return DocumentChunk(
            text=description,
            source=source,
            page=page,
            chunk_id=chunk_id
        )