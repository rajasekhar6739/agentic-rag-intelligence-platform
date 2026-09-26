from rag.reranker import Reranker


class HybridRetriever:

    def __init__(
        self,
        bm25_store,
        vector_store,
        rrf_k: int = 60,
        reranker=None
    ):
        self.bm25_store = bm25_store
        self.vector_store = vector_store
        self.rrf_k = rrf_k

        self.reranker = reranker or Reranker()

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10
    ):

        bm25_results = self._search(
            self.bm25_store,
            query,
            candidate_k
        )

        vector_results = self._search(
            self.vector_store,
            query,
            candidate_k
        )

        fused = self._rrf_fusion(
            bm25_results,
            vector_results
        )

        # Retrieve more candidates before reranking
        candidates = fused[:candidate_k]

        # Cross-encoder reranking
        reranked = self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=top_k
        )

        return reranked

    def _search(
        self,
        store,
        query,
        top_k
    ):

        if store is None:
            return []

        search = getattr(
            store,
            "search",
            None
        )

        if search is None:
            return []

        return search(
            query,
            top_k=top_k
        )

    def _rrf_fusion(
        self,
        bm25_results,
        vector_results
    ):

        scores = {}
        documents = {}

        for rank, document in enumerate(
            bm25_results,
            start=1
        ):

            key = self._document_key(
                document
            )

            scores[key] = (
                scores.get(key, 0.0)
                + 1.0 / (
                    self.rrf_k + rank
                )
            )

            documents[key] = document

        for rank, document in enumerate(
            vector_results,
            start=1
        ):

            key = self._document_key(
                document
            )

            scores[key] = (
                scores.get(key, 0.0)
                + 1.0 / (
                    self.rrf_k + rank
                )
            )

            documents[key] = document

        ranked = sorted(
            documents.items(),
            key=lambda item: scores[
                item[0]
            ],
            reverse=True
        )

        return [
            {
                "document": document,
                "rrf_score": scores[key]
            }
            for key, document in ranked
        ]

    @staticmethod
    def _document_key(document):

        if isinstance(
            document,
            dict
        ):

            return (
                document.get("source"),
                document.get("page"),
                document.get("chunk_id")
            )

        return str(document)