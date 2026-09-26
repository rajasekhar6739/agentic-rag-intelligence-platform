from agents.tool_agent import ToolAgent
from agents.schemas import DocumentChunk


def test_e2e_evaluation():
    documents = [
        DocumentChunk(
            text=(
                "RAG stands for Retrieval-Augmented Generation. "
                "It retrieves relevant information from a knowledge base "
                "and provides that context to an LLM."
            ),
            source="rag_notes.txt",
            page=1,
            chunk_id=0,
        ),
        DocumentChunk(
            text=(
                "Hybrid retrieval combines semantic vector search "
                "with lexical keyword retrieval such as BM25."
            ),
            source="rag_notes.txt",
            page=2,
            chunk_id=1,
        ),
        DocumentChunk(
            text=(
                "Reranking reorders retrieved candidates according "
                "to their relevance to the user query."
            ),
            source="rag_notes.txt",
            page=3,
            chunk_id=2,
        ),
    ]

    agent = ToolAgent()

    load_result = agent.load_documents(documents)
    assert load_result is not None

    queries = [
        "What is RAG?",
        "What is hybrid retrieval?",
        "What does reranking do?",
    ]

    successful = 0

    for query in queries:
        result = agent.run(query)

        assert result["success"] is True
        assert result["answer"]

        successful += 1

        print("\n==============================")
        print("QUERY:", query)
        print("SUCCESS:", result["success"])
        print("ANSWER:", result["answer"])
        print("TOOLS:", result.get("tools_used"))

    success_rate = successful / len(queries)

    print("\n==============================")
    print("E2E SUCCESS RATE:", success_rate)
    print("==============================")

    assert success_rate == 1.0