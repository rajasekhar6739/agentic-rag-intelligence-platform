from rag.vector_store import VectorStore
from rag.bm25 import BM25Store
from agents.schemas import DocumentChunk


class HybridRetriever:

    def __init__(self):

        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()

    def build(
        self,
        documents: list[DocumentChunk]
    ):

        if not documents:
            raise ValueError(
                "No documents provided."
            )

        self.vector_store.build(
            documents
        )

        self.bm25_store.build(
            documents
        )

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:

        vector_results = (
            self.vector_store.search(
                query,
                top_k=top_k * 2
            )
        )

        bm25_results = (
            self.bm25_store.search(
                query,
                top_k=top_k * 2
            )
        )

        combined = {}

        # Vector results
        for result in vector_results:

            key = (
                result["source"],
                result["page"],
                result["chunk_id"]
            )

            combined[key] = {
                **result,
                "vector_score": result["score"],
                "bm25_score": 0.0
            }

        # BM25 results
        for result in bm25_results:

            key = (
                result["source"],
                result["page"],
                result["chunk_id"]
            )

            if key not in combined:

                combined[key] = {
                    **result,
                    "vector_score": 0.0,
                    "bm25_score": result["score"]
                }

            else:

                combined[key][
                    "bm25_score"
                ] = result["score"]

        # Normalize BM25
        max_bm25 = max(
            [
                item["bm25_score"]
                for item in combined.values()
            ],
            default=1.0
        )

        if max_bm25 <= 0:
            max_bm25 = 1.0

        # Fusion
        for item in combined.values():

            vector_score = (
                item["vector_score"]
            )

            bm25_score = (
                item["bm25_score"]
                / max_bm25
            )

            item["hybrid_score"] = (
                0.65 * vector_score
                +
                0.35 * bm25_score
            )

        ranked = sorted(
            combined.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return ranked[:top_k]