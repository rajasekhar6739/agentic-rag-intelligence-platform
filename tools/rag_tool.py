from __future__ import annotations

from typing import Any, Optional


class RAGTool:
    """
    RAG retrieval tool used by the AgentOrchestrator.

    Supports:
        - HybridRetriever
        - Vector retrievers
        - callable retrievers
        - local fallback retrieval
    """

    description = (
        "Search indexed documents and return relevant "
        "evidence with source, page and score metadata."
    )

    def __init__(
        self,
        retriever: Optional[Any] = None,
    ):

        self.retriever = retriever
        self.documents = []

    # ============================================================
    # CONFIGURE
    # ============================================================

    def configure(
        self,
        retriever: Any,
    ):

        self.retriever = retriever

        return self

    # ============================================================
    # LOAD DOCUMENTS
    # ============================================================

    def load_documents(
        self,
        documents,
    ):

        documents = list(
            documents or []
        )

        self.documents = documents

        if self.retriever is not None:

            if hasattr(
                self.retriever,
                "build",
            ):

                self.retriever.build(
                    documents
                )

            elif hasattr(
                self.retriever,
                "load_documents",
            ):

                self.retriever.load_documents(
                    documents
                )

            elif hasattr(
                self.retriever,
                "add_documents",
            ):

                self.retriever.add_documents(
                    documents
                )

            return {
                "success": True,
                "loaded": len(
                    documents
                ),
                "backend": type(
                    self.retriever
                ).__name__,
            }

        return {
            "success": True,
            "loaded": len(
                documents
            ),
            "backend": "local",
        }

    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):

        if not query:
            return []

        top_k = max(
            1,
            int(top_k),
        )

        # --------------------------------------------------------
        # RETRIEVER
        # --------------------------------------------------------

        if self.retriever is not None:

            if hasattr(
                self.retriever,
                "search",
            ):

                results = (
                    self.retriever.search(
                        query,
                        top_k=top_k,
                    )
                )

            elif callable(
                self.retriever
            ):

                results = self.retriever(
                    query,
                    top_k=top_k,
                )

            else:

                raise RuntimeError(
                    "Configured retriever "
                    "does not support search()."
                )

            return self._normalize_results(
                results,
                top_k,
            )

        # --------------------------------------------------------
        # LOCAL FALLBACK
        # --------------------------------------------------------

        query_words = set(
            str(query)
            .lower()
            .split()
        )

        scored = []

        for index, document in enumerate(
            self.documents
        ):

            text = self._get_text(
                document
            )

            if not text:
                continue

            words = set(
                text.lower().split()
            )

            overlap = (
                query_words
                .intersection(words)
            )

            score = len(
                overlap
            )

            if score <= 0:
                continue

            result = self._to_result(
                document,
                score,
                index,
            )

            scored.append(
                result
            )

        scored.sort(
            key=lambda item: float(
                item.get(
                    "score",
                    0,
                )
            ),
            reverse=True,
        )

        return scored[:top_k]

    # ============================================================
    # NORMALIZE RESULTS
    # ============================================================

    @classmethod
    def _normalize_results(
        cls,
        results,
        top_k,
    ):

        if results is None:
            return []

        if isinstance(
            results,
            dict,
        ):

            if "results" in results:

                results = results.get(
                    "results",
                    [],
                )

            else:

                results = [
                    results
                ]

        if isinstance(
            results,
            str,
        ):

            results = [
                results
            ]

        if not isinstance(
            results,
            (list, tuple),
        ):

            results = [
                results
            ]

        normalized = []

        for index, item in enumerate(
            results
        ):

            if isinstance(
                item,
                dict,
            ):

                result = dict(item)

            elif isinstance(
                item,
                str,
            ):

                result = {
                    "text": item,
                    "source": "unknown",
                    "page": 1,
                    "chunk_id": (
                        f"chunk_{index}"
                    ),
                    "score": 0.0,
                }

            else:

                result = (
                    cls._to_result(
                        item,
                        getattr(
                            item,
                            "score",
                            0.0,
                        ),
                        index,
                    )
                )

            result.setdefault(
                "text",
                result.get(
                    "content",
                    "",
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
                f"chunk_{index}",
            )

            result.setdefault(
                "score",
                0.0,
            )

            normalized.append(
                result
            )

        return normalized[:top_k]

    # ============================================================
    # TEXT EXTRACTION
    # ============================================================

    @staticmethod
    def _get_text(
        document,
    ):

        if isinstance(
            document,
            dict,
        ):

            return str(
                document.get(
                    "text",
                    document.get(
                        "content",
                        "",
                    ),
                )
            )

        if hasattr(
            document,
            "text",
        ):

            return str(
                document.text
            )

        if hasattr(
            document,
            "content",
        ):

            return str(
                document.content
            )

        return str(
            document
        )

    # ============================================================
    # RESULT CONVERSION
    # ============================================================

    @classmethod
    def _to_result(
        cls,
        document,
        score=0.0,
        index=0,
    ):

        text = cls._get_text(
            document
        )

        if isinstance(
            document,
            dict,
        ):

            result = dict(
                document
            )

            result["text"] = text
            result.setdefault(
                "score",
                score,
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
                f"chunk_{index}",
            )

            return result

        return {
            "text": text,
            "score": score,
            "source": getattr(
                document,
                "source",
                "unknown",
            ),
            "page": getattr(
                document,
                "page",
                1,
            ),
            "chunk_id": getattr(
                document,
                "chunk_id",
                f"chunk_{index}",
            ),
        }

    # ============================================================
    # REPRESENTATION
    # ============================================================

    def __repr__(self):

        return (
            "RAGTool("
            f"retriever={type(self.retriever).__name__ "
            "if self.retriever else 'None'}"
            ")"
        )