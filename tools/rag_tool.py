from __future__ import annotations

from typing import Any, Optional


class RAGTool:
    """
    Lightweight RAG tool with injectable retriever.

    Supports:
        - retriever.search(query, top_k)
        - callable retrievers
        - local lexical retrieval when no backend is configured
    """

    description = (
        "Search indexed documents and return relevant evidence "
        "with source and page metadata."
    )

    def __init__(self, retriever: Optional[Any] = None):
        self.retriever = retriever
        self.documents = []

    def configure(self, retriever: Any):
        self.retriever = retriever
        return self

    # ============================================================
    # DOCUMENT LOADING
    # ============================================================

    def load_documents(self, documents):
        documents = list(documents or [])

        if self.retriever is not None:

            if hasattr(self.retriever, "load_documents"):
                result = self.retriever.load_documents(documents)

                return {
                    "success": True,
                    "loaded": len(documents),
                    "backend": type(self.retriever).__name__,
                    "result": result,
                }

            if hasattr(self.retriever, "add_documents"):
                result = self.retriever.add_documents(documents)

                return {
                    "success": True,
                    "loaded": len(documents),
                    "backend": type(self.retriever).__name__,
                    "result": result,
                }

            if hasattr(self.retriever, "upsert"):
                result = self.retriever.upsert(documents)

                return {
                    "success": True,
                    "loaded": len(documents),
                    "backend": type(self.retriever).__name__,
                    "result": result,
                }

        self.documents = documents

        return {
            "success": True,
            "loaded": len(documents),
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

        top_k = max(1, int(top_k))

        # --------------------------------------------------------
        # EXTERNAL RETRIEVER
        # --------------------------------------------------------

        if self.retriever is not None:

            if hasattr(self.retriever, "search"):
                result = self.retriever.search(
                    query,
                    top_k=top_k,
                )

            elif callable(self.retriever):
                result = self.retriever(
                    query,
                    top_k=top_k,
                )

            else:
                raise RuntimeError(
                    "Configured RAG retriever does not support search()."
                )

            if result is None:
                return []

            if isinstance(result, list):
                return result[:top_k]

            return result

        # --------------------------------------------------------
        # LOCAL LEXICAL RETRIEVAL
        # --------------------------------------------------------

        query_words = set(
            str(query).lower().split()
        )

        scored = []

        for document in self.documents:

            text = self._get_text(document)

            if not text:
                continue

            words = set(
                text.lower().split()
            )

            overlap = query_words.intersection(words)

            score = len(overlap)

            if score <= 0:
                continue

            result = self._to_result(
                document=document,
                score=score,
            )

            scored.append(result)

        scored.sort(
            key=lambda item: item.get("score", 0),
            reverse=True,
        )

        return scored[:top_k]

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _get_text(document):
        if isinstance(document, dict):
            return str(
                document.get(
                    "text",
                    document.get(
                        "content",
                        "",
                    ),
                )
            )

        if hasattr(document, "text"):
            return str(document.text)

        return str(document)

    @classmethod
    def _to_result(
        cls,
        document,
        score,
    ):
        text = cls._get_text(document)

        if isinstance(document, dict):

            result = dict(document)

            result["text"] = text
            result["score"] = score

            return result

        result = {
            "text": text,
            "score": score,
        }

        source = getattr(
            document,
            "source",
            None,
        )

        page = getattr(
            document,
            "page",
            None,
        )

        chunk_id = getattr(
            document,
            "chunk_id",
            None,
        )

        if source is not None:
            result["source"] = source

        if page is not None:
            result["page"] = page

        if chunk_id is not None:
            result["chunk_id"] = chunk_id

        return result