from rag.ingestion import DocumentIngestion


text = """
Retrieval Augmented Generation improves
LLM responses by retrieving relevant external
knowledge before generating an answer.

Hybrid retrieval combines semantic vector
search with lexical BM25 retrieval.

A reranker then improves the ordering of
retrieved documents.
"""


ingestion = DocumentIngestion(
    chunk_size=120,
    overlap=30
)


chunks = ingestion.ingest(
    text=text,
    source="genai_notes.txt",
    page=1
)


print("\n==============================")
print("DOCUMENT INGESTION")
print("==============================")

print(
    "Total chunks:",
    len(chunks)
)


for chunk in chunks:

    print("\n------------------------------")

    print("ID:", chunk.chunk_id)
    print("Source:", chunk.source)
    print("Page:", chunk.page)
    print("Text:", chunk.text)