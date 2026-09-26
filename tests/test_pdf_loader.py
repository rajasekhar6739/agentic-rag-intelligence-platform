import sys

from multimodal.pdf_loader import PDFLoader


def main(pdf_path: str) -> int:

    loader = PDFLoader()

    chunks = loader.load(
        pdf_path
    )

    print("\n==============================")
    print("PDF INGESTION")
    print("==============================")

    print(
        "Chunks:",
        len(chunks),
    )

    for chunk in chunks[:3]:

        print("\n------------------------------")

        print(
            "Source:",
            chunk.source,
        )

        print(
            "Page:",
            chunk.page,
        )

        print(
            "Chunk ID:",
            chunk.chunk_id,
        )

        print(
            "Text:",
            chunk.text[:500],
        )

    return 0


def test_pdf_loader_import():
    """
    Pytest smoke test.

    Actual PDF ingestion is performed through main()
    because this module is also a CLI utility.
    """
    assert PDFLoader is not None


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: "
            "python -m tests.test_pdf_loader <pdf_path>"
        )

        raise SystemExit(1)

    raise SystemExit(
        main(sys.argv[1])
    )