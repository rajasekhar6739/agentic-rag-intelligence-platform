from rag.ingestion import DocumentIngestion


class RAGIndexManager:

    def __init__(self, bm25_store=None, vector_store=None):
        self.bm25_store = bm25_store
        self.vector_store = vector_store

        self.ingestion = DocumentIngestion(
            chunk_size=500,
            overlap=100
        )

    def add_text(
        self,
        text: str,
        source: str = "unknown",
        page: int = 1
    ):

        chunks = self.ingestion.ingest(
            text=text,
            source=source,
            page=page
        )

        if not chunks:
            return {
                "success": False,
                "chunks": 0
            }

        # BM25
        if self.bm25_store:
            method = getattr(
                self.bm25_store,
                "add_documents",
                None
            )

            if method:
                method(chunks)

        # Vector
        if self.vector_store:
            method = getattr(
                self.vector_store,
                "add_documents",
                None
            )

            if method:
                method(chunks)

        return {
            "success": True,
            "chunks": len(chunks),
            "documents": chunks
        }