from rank_bm25 import BM25Okapi

from agents.schemas import DocumentChunk


class BM25Store:

    def __init__(self):
        self.documents: list[DocumentChunk] = []
        self.bm25 = None

    def build(
        self,
        documents: list[DocumentChunk]
    ):

        if not documents:
            raise ValueError(
                "No documents provided."
            )

        self.documents = documents

        tokenized_documents = [
            document.text.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index is empty."
            )

        query_tokens = (
            query.lower().split()
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:

            document = self.documents[index]

            results.append({
                "text": document.text,
                "source": document.source,
                "page": document.page,
                "chunk_id": document.chunk_id,
                "score": float(scores[index])
            })

        return results