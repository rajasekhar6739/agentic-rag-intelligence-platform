from rag.ingestion import DocumentIngestion


class RAGIndexer:

    def __init__(
        self,
        bm25_store=None,
        vector_store=None,
        chunk_size=500,
        overlap=100,
    ):
        self.bm25_store = bm25_store
        self.vector_store = vector_store

        self.ingestion = DocumentIngestion(
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def prepare(
        self,
        text,
        source="unknown",
        page=1,
    ):
        """
        Convert raw document text into
        normalized DocumentChunk objects.
        """

        return self.ingestion.ingest(
            text=text,
            source=source,
            page=page,
        )

    def index(
        self,
        text,
        source="unknown",
        page=1,
    ):
        """
        Prepare chunks and index them into
        whichever stores are available.
        """

        chunks = self.prepare(
            text=text,
            source=source,
            page=page,
        )

        if not chunks:
            return {
                "success": False,
                "chunks": 0,
                "message": "No content to index.",
            }

        indexed = {
            "bm25": False,
            "vector": False,
        }

        # -----------------------------
        # BM25
        # -----------------------------

        if self.bm25_store is not None:

            if hasattr(
                self.bm25_store,
                "add_documents"
            ):
                self.bm25_store.add_documents(
                    chunks
                )

                indexed["bm25"] = True

        # -----------------------------
        # VECTOR STORE
        # -----------------------------

        if self.vector_store is not None:

            if hasattr(
                self.vector_store,
                "add_documents"
            ):
                self.vector_store.add_documents(
                    chunks
                )

                indexed["vector"] = True

        return {
            "success": True,
            "chunks": len(chunks),
            "indexed": indexed,
            "documents": chunks,
        }