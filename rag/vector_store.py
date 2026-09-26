import faiss
import numpy as np

from rag.embeddings import EmbeddingModel
from agents.schemas import DocumentChunk


class VectorStore:

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.index = None
        self.documents = []

    def build(self, documents: list[DocumentChunk]):

        if not documents:
            raise ValueError("No documents provided.")

        texts = [
            document.text
            for document in documents
        ]

        embeddings = self.embedding_model.encode(texts)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.documents = documents

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:

        if self.index is None:
            raise RuntimeError(
                "Vector store is empty. Build the index first."
            )

        query_embedding = (
            self.embedding_model
            .encode_query(query)
        )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.documents))
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            document = self.documents[int(index)]

            results.append({
                "text": document.text,
                "source": document.source,
                "page": document.page,
                "chunk_id": document.chunk_id,
                "score": float(score)
            })

        return results