from __future__ import annotations

from typing import Any

from rag.vector_store import VectorStore
from rag.bm25 import BM25Store
from agents.schemas import DocumentChunk


class HybridRetriever:
    """
    Hybrid RAG retriever.

    Combines:
        1. Dense vector retrieval
        2. BM25 lexical retrieval

    All retrieval outputs are normalized into dictionaries so that
    downstream agents/tools never receive unexpected string objects.
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()
        self.documents = []

    # ============================================================
    # BUILD INDEX
    # ============================================================

    def build(self, documents: list[DocumentChunk]):
        if not documents:
            raise ValueError("No documents provided.")

        self.documents = list(documents)

        self.vector_store.build(self.documents)
        self.bm25_store.build(self.documents)

        return {
            "success": True,
            "documents": len(self.documents),
        }

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if not query or not str(query).strip():
            return []

        top_k = max(1, int(top_k))

        # --------------------------------------------------------
        # VECTOR SEARCH
        # --------------------------------------------------------

        try:
            vector_results = self.vector_store.search(
                str(query),
                top_k=top_k * 2,
            )
        except Exception as exc:
            print("VECTOR SEARCH WARNING:", repr(exc))
            vector_results = []

        # --------------------------------------------------------
        # BM25 SEARCH
        # --------------------------------------------------------

        try:
            bm25_results = self.bm25_store.search(
                str(query),
                top_k=top_k * 2,
            )
        except Exception as exc:
            print("BM25 SEARCH WARNING:", repr(exc))
            bm25_results = []

        # --------------------------------------------------------
        # NORMALIZE RESULTS
        # --------------------------------------------------------

        vector_results = self._normalize_results(
            vector_results,
            default_score=0.0,
        )

        bm25_results = self._normalize_results(
            bm25_results,
            default_score=0.0,
        )

        # --------------------------------------------------------
        # COMBINE
        # --------------------------------------------------------

        combined: dict[tuple, dict] = {}

        # VECTOR RESULTS
        for result in vector_results:

            key = self._result_key(result)

            result["vector_score"] = self._safe_float(
                result.get("score", 0.0)
            )

            result["bm25_score"] = 0.0

            combined[key] = result

        # BM25 RESULTS
        for result in bm25_results:

            key = self._result_key(result)

            bm25_score = self._safe_float(
                result.get("score", 0.0)
            )

            if key not in combined:

                result["vector_score"] = 0.0
                result["bm25_score"] = bm25_score

                combined[key] = result

            else:

                combined[key]["bm25_score"] = bm25_score

        # --------------------------------------------------------
        # NORMALIZE BM25
        # --------------------------------------------------------

        max_bm25 = max(
            (
                self._safe_float(
                    item.get("bm25_score", 0.0)
                )
                for item in combined.values()
            ),
            default=1.0,
        )

        if max_bm25 <= 0:
            max_bm25 = 1.0

        # --------------------------------------------------------
        # HYBRID FUSION
        # --------------------------------------------------------

        for item in combined.values():

            vector_score = self._safe_float(
                item.get("vector_score", 0.0)
            )

            bm25_score = (
                self._safe_float(
                    item.get("bm25_score", 0.0)
                )
                / max_bm25
            )

            item["hybrid_score"] = (
                0.65 * vector_score
                + 0.35 * bm25_score
            )

            # Keep generic score for downstream tools.
            item["score"] = item["hybrid_score"]

        # --------------------------------------------------------
        # FINAL RANKING
        # --------------------------------------------------------

        ranked = sorted(
            combined.values(),
            key=lambda item: self._safe_float(
                item.get("hybrid_score", 0.0)
            ),
            reverse=True,
        )

        return ranked[:top_k]

    # ============================================================
    # RESULT NORMALIZATION
    # ============================================================

    def _normalize_results(
        self,
        results,
        default_score: float = 0.0,
    ) -> list[dict]:

        if results is None:
            return []

        if not isinstance(results, (list, tuple)):
            results = [results]

        normalized = []

        for index, item in enumerate(results):

            # ----------------------------------------------------
            # Already a dictionary
            # ----------------------------------------------------

            if isinstance(item, dict):

                result = dict(item)

                result.setdefault(
                    "text",
                    result.get(
                        "content",
                        result.get(
                            "document",
                            "",
                        ),
                    ),
                )

                result.setdefault(
                    "source",
                    "unknown",
                )

                result.setdefault(
                    "page",
                    1,
                )

                result.setdefault(
                    "chunk_id",
                    index,
                )

                result.setdefault(
                    "score",
                    default_score,
                )

                normalized.append(result)
                continue

            # ----------------------------------------------------
            # DocumentChunk / object
            # ----------------------------------------------------

            if hasattr(item, "text"):

                result = {
                    "text": str(
                        getattr(
                            item,
                            "text",
                            "",
                        )
                    ),
                    "source": getattr(
                        item,
                        "source",
                        "unknown",
                    ),
                    "page": getattr(
                        item,
                        "page",
                        1,
                    ),
                    "chunk_id": getattr(
                        item,
                        "chunk_id",
                        index,
                    ),
                    "score": getattr(
                        item,
                        "score",
                        default_score,
                    ),
                }

                normalized.append(result)
                continue

            # ----------------------------------------------------
            # STRING RESULT
            # ----------------------------------------------------

            if isinstance(item, str):

                normalized.append(
                    {
                        "text": item,
                        "source": "unknown",
                        "page": 1,
                        "chunk_id": index,
                        "score": default_score,
                    }
                )

                continue

            # ----------------------------------------------------
            # UNKNOWN OBJECT
            # ----------------------------------------------------

            normalized.append(
                {
                    "text": str(item),
                    "source": "unknown",
                    "page": 1,
                    "chunk_id": index,
                    "score": default_score,
                }
            )

        return normalized

    # ============================================================
    # RESULT KEY
    # ============================================================

    @staticmethod
    def _result_key(
        result: dict,
    ) -> tuple:

        return (
            str(
                result.get(
                    "source",
                    "unknown",
                )
            ),
            str(
                result.get(
                    "page",
                    1,
                )
            ),
            str(
                result.get(
                    "chunk_id",
                    0,
                )
            ),
        )

    # ============================================================
    # SAFE FLOAT
    # ============================================================

    @staticmethod
    def _safe_float(value) -> float:

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    # ============================================================
    # DOCUMENT LOADING COMPATIBILITY
    # ============================================================

    def load_documents(self, documents):

        documents = list(documents or [])

        if not documents:
            return {
                "success": False,
                "loaded": 0,
                "error": "No documents provided.",
            }

        self.build(documents)

        return {
            "success": True,
            "loaded": len(documents),
        }

    def add_documents(self, documents):

        documents = list(documents or [])

        if not documents:
            return {
                "success": False,
                "loaded": 0,
            }

        self.documents.extend(documents)

        self.build(self.documents)

        return {
            "success": True,
            "loaded": len(documents),
            "total": len(self.documents),
        }

    def __repr__(self):

        return (
            f"HybridRetriever("
            f"documents={len(self.documents)}"
            f")"
        )