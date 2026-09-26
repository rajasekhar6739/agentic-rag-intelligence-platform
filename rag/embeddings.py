from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return np.asarray(
            embeddings,
            dtype="float32"
        )

    def encode_query(self, query: str) -> np.ndarray:

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        return np.asarray(
            embedding,
            dtype="float32"
        )
