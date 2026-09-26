from pathlib import Path

from pypdf import PdfReader

from agents.schemas import DocumentChunk


class PDFLoader:

    def load(
        self,
        file_path: str
    ) -> list[DocumentChunk]:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF not found: {file_path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        reader = PdfReader(
            str(path)
        )

        chunks = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            text = text.strip()

            if not text:
                continue

            chunks.append(
                DocumentChunk(
                    text=text,
                    source=path.name,
                    page=page_number,
                    chunk_id=len(chunks)
                )
            )

        return chunks