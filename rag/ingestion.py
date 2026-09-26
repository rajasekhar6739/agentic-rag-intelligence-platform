from __future__ import annotations

from pathlib import Path
from typing import Any, List

from rag.chunker import TextChunker


class DocumentIngestion:
    """
    Document ingestion layer for the Agentic RAG system.

    Supports:
        - ingest()
        - load()
        - load_documents()
        - chunk_documents()
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
    ):
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

    # ============================================================
    # INGEST TEXT
    # ============================================================

    def ingest(
        self,
        text: str,
        source: str = "unknown",
        page: int = 1,
    ) -> List[Any]:

        if text is None:
            return []

        text = str(text).strip()

        if not text:
            return []

        return self.chunker.split(
            text=text,
            source=source,
            page=page,
        )

    # ============================================================
    # LOAD FILE / DIRECTORY
    # ============================================================

    def load(
        self,
        path: str | Path,
    ) -> List[Any]:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document path does not exist: {path}"
            )

        # --------------------------------------------------------
        # DIRECTORY
        # --------------------------------------------------------

        if path.is_dir():

            all_chunks = []

            for file_path in sorted(
                path.rglob("*")
            ):

                if not file_path.is_file():
                    continue

                try:

                    chunks = self._load_file(
                        file_path
                    )

                    all_chunks.extend(
                        chunks
                    )

                except Exception as exc:

                    print(
                        f"WARNING: Failed to load "
                        f"{file_path}: {exc}"
                    )

            return all_chunks

        # --------------------------------------------------------
        # SINGLE FILE
        # --------------------------------------------------------

        return self._load_file(path)

    # ============================================================
    # LOAD SINGLE FILE
    # ============================================================

    def _load_file(
        self,
        path: Path,
    ) -> List[Any]:

        suffix = path.suffix.lower()

        # --------------------------------------------------------
        # TEXT FILES
        # --------------------------------------------------------

        if suffix in {
            ".txt",
            ".md",
            ".markdown",
            ".text",
        }:

            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            return self.ingest(
                text=text,
                source=str(path),
                page=1,
            )

        # --------------------------------------------------------
        # PDF
        # --------------------------------------------------------

        if suffix == ".pdf":

            return self._load_pdf(path)

        # --------------------------------------------------------
        # UNSUPPORTED
        # --------------------------------------------------------

        return []

    # ============================================================
    # PDF
    # ============================================================

    def _load_pdf(
        self,
        path: Path,
    ) -> List[Any]:

        try:

            from pypdf import PdfReader

        except ImportError as exc:

            raise ImportError(
                "pypdf is required for PDF loading. "
                "Run: pip install pypdf"
            ) from exc

        reader = PdfReader(
            str(path)
        )

        all_chunks = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = page.extract_text() or ""

            if not text.strip():
                continue

            chunks = self.ingest(
                text=text,
                source=str(path),
                page=page_number,
            )

            all_chunks.extend(
                chunks
            )

        return all_chunks

    # ============================================================
    # LOAD MULTIPLE DOCUMENTS
    # ============================================================

    def load_documents(
        self,
        documents: Any,
    ) -> List[Any]:

        if documents is None:
            return []

        if isinstance(
            documents,
            (str, Path),
        ):

            return self.load(
                documents
            )

        all_chunks = []

        if isinstance(
            documents,
            (list, tuple),
        ):

            for document in documents:

                if isinstance(
                    document,
                    (str, Path),
                ):

                    all_chunks.extend(
                        self.load(
                            document
                        )
                    )

                else:

                    all_chunks.extend(
                        self.chunk_documents(
                            document
                        )
                    )

            return all_chunks

        return self.chunk_documents(
            documents
        )

    # ============================================================
    # CHUNK DOCUMENT OBJECTS
    # ============================================================

    def chunk_documents(
        self,
        documents: Any,
    ) -> List[Any]:

        if documents is None:
            return []

        # --------------------------------------------------------
        # STRING
        # --------------------------------------------------------

        if isinstance(
            documents,
            str,
        ):

            return self.ingest(
                text=documents,
                source="unknown",
                page=1,
            )

        # --------------------------------------------------------
        # DICT
        # --------------------------------------------------------

        if isinstance(
            documents,
            dict,
        ):

            text = (
                documents.get("text")
                or documents.get("content")
                or documents.get("page_content")
                or ""
            )

            source = (
                documents.get("source")
                or documents.get("filename")
                or "unknown"
            )

            page = documents.get(
                "page",
                1,
            )

            if not text:
                return []

            return self.ingest(
                text=str(text),
                source=str(source),
                page=int(page),
            )

        # --------------------------------------------------------
        # LIST / TUPLE
        # --------------------------------------------------------

        if isinstance(
            documents,
            (list, tuple),
        ):

            all_chunks = []

            for index, document in enumerate(
                documents
            ):

                if isinstance(
                    document,
                    str,
                ):

                    all_chunks.extend(
                        self.ingest(
                            text=document,
                            source=(
                                f"document_{index + 1}"
                            ),
                            page=1,
                        )
                    )

                elif isinstance(
                    document,
                    dict,
                ):

                    text = (
                        document.get("text")
                        or document.get("content")
                        or document.get("page_content")
                        or ""
                    )

                    source = (
                        document.get("source")
                        or document.get("filename")
                        or (
                            f"document_{index + 1}"
                        )
                    )

                    page = document.get(
                        "page",
                        1,
                    )

                    if text:

                        all_chunks.extend(
                            self.ingest(
                                text=str(text),
                                source=str(source),
                                page=int(page),
                            )
                        )

                elif hasattr(
                    document,
                    "page_content",
                ):

                    text = getattr(
                        document,
                        "page_content",
                        "",
                    )

                    metadata = getattr(
                        document,
                        "metadata",
                        {},
                    )

                    if not isinstance(
                        metadata,
                        dict,
                    ):
                        metadata = {}

                    source = metadata.get(
                        "source",
                        f"document_{index + 1}",
                    )

                    page = metadata.get(
                        "page",
                        1,
                    )

                    if text:

                        all_chunks.extend(
                            self.ingest(
                                text=str(text),
                                source=str(source),
                                page=int(page),
                            )
                        )

            return all_chunks

        # --------------------------------------------------------
        # OBJECT WITH TEXT
        # --------------------------------------------------------

        if hasattr(
            documents,
            "text",
        ):

            text = getattr(
                documents,
                "text",
                "",
            )

            source = getattr(
                documents,
                "source",
                "unknown",
            )

            page = getattr(
                documents,
                "page",
                1,
            )

            return self.ingest(
                text=str(text),
                source=str(source),
                page=int(page),
            )

        return []

    # ============================================================
    # REPRESENTATION
    # ============================================================

    def __repr__(self) -> str:

        return (
            "DocumentIngestion("
            f"chunk_size={getattr(self.chunker, 'chunk_size', '?')}, "
            f"overlap={getattr(self.chunker, 'overlap', '?')}"
            ")"
        )