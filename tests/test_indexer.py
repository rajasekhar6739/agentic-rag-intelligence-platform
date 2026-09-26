from rag.indexer import RAGIndexer


text = """
Large language models use transformer
architectures based on attention mechanisms.

Retrieval Augmented Generation improves
answers by retrieving relevant external
knowledge.

Hybrid retrieval combines BM25 lexical
search with semantic vector search.

A reranker can improve the ordering of
retrieved documents before generation.
"""


indexer = RAGIndexer()


result = indexer.index(
    text=text,
    source="genai_architecture.txt",
    page=1,
)


print("\n==============================")
print("RAG INDEXER")
print("==============================")

print(
    "Success:",
    result["success"]
)

print(
    "Chunks:",
    result["chunks"]
)

print(
    "Indexed:",
    result["indexed"]
)


for chunk in result["documents"]:

    print("\n------------------------------")

    print(
        "ID:",
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