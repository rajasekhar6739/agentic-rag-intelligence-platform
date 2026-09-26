from rag.index_manager import RAGIndexManager


manager = RAGIndexManager()

result = manager.add_text(
    """
    Transformers use self-attention to model
    relationships between tokens.

    RAG retrieves external knowledge before
    an LLM generates the final answer.

    Hybrid retrieval combines BM25 and
    semantic vector search.
    """,
    source="genai.txt",
    page=1
)

print("\n==============================")
print("INDEX MANAGER")
print("==============================")

print("Success:", result["success"])
print("Chunks:", result["chunks"])

for chunk in result["documents"]:
    print(
        chunk.chunk_id,
        chunk.source,
        chunk.page,
        chunk.text
    )