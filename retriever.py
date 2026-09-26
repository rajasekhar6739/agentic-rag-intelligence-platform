import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.docs = []
        self.index = None
        self.bm25 = None

    def build(self, docs: list[dict]):
        self.docs = docs
        texts = [d["text"] for d in docs]
        vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.docs:
            return []
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(qv, min(k * 2, len(self.docs)))

        dense = {}
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                dense[int(idx)] = float(score)

        bm_scores = self.bm25.get_scores(query.lower().split())
        candidates = set(dense)
        candidates.update(np.argsort(bm_scores)[-k * 2:])

        ranked = []
        for idx in candidates:
            # Simple fusion for MVP; replace with learned reranker later.
            d = dense.get(idx, 0.0)
            b = float(bm_scores[idx])
            score = 0.65 * d + 0.35 * (b / (1.0 + b) if b > 0 else 0.0)
            item = dict(self.docs[idx])
            item["retrieval_score"] = round(score, 4)
            ranked.append(item)

        return sorted(ranked, key=lambda x: x["retrieval_score"], reverse=True)[:k]
