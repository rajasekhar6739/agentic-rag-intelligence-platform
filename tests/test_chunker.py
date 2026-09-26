from rag.chunker import TextChunker


text = """
Retrieval Augmented Generation combines
information retrieval with language models.
Documents are divided into chunks.
Embeddings represent semantic meaning.
Vector search retrieves relevant chunks.
BM25 provides lexical retrieval.
Hybrid retrieval combines both approaches.
"""


chunker = TextChunker(
    chunk_size=100,
    overlap=20
)


chunks = chunker.split(
    text,
    source="rag_notes.txt",
    page=1
)


print("\n==============================")
print("INTELLIGENT CHUNKING")
print("==============================")


print(
    "Total chunks:",
    len(chunks)
)


for chunk in chunks:

    print("\n------------------------------")

    print(
        "Chunk ID:",
        chunk.chunk_id
    )

    print(
        "Source:",
        chunk.source
    )

    print(
        "Page:",
        chunk.page
    )

    print(
        "Text:",
        chunk.text
    )